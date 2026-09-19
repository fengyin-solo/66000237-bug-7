/**
 * ECG 信号生成与分析共享算法（前端本地判定）。
 *
 * 本模块与后端 app/services/ecg_service.py 使用完全一致的：
 *  - 确定性伪随机信号生成（相同参数生成同一段波形）
 *  - 5-15Hz Butterworth 带通 + 微分 + 平方 + 滑动积分的 Pan-Tompkins 检测
 *  - HRV 指标、心律失常判定规则
 * 从而保证「同一段数据」在本地与后端得到相同的心跳数、心率与结论。
 */
import type { ArrhythmiaEvent, ECGLead, HRVData, RPeak } from '../types';

// ---------------------------------------------------------------------------
// 确定性随机数（mulberry32 + Box-Muller），与后端实现逐位对齐
// ---------------------------------------------------------------------------

function hashSeed(lead: string, hr: number, duration: number, sr: number): number {
  const key = `${lead}|${hr}|${duration}|${sr}`;
  let h = 0x811c9dc5;
  for (let i = 0; i < key.length; i++) {
    h = Math.imul(h ^ key.charCodeAt(i), 0x01000193) >>> 0;
  }
  return h >>> 0;
}

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function makeGaussian(rng: () => number): () => number {
  let spare: number | null = null;
  return () => {
    if (spare !== null) {
      const v = spare;
      spare = null;
      return v;
    }
    let u = 0;
    let v = 0;
    let s = 0;
    do {
      u = rng() * 2 - 1;
      v = rng() * 2 - 1;
      s = u * u + v * v;
    } while (s >= 1 || s === 0);
    const mul = Math.sqrt((-2 * Math.log(s)) / s);
    spare = v * mul;
    return u * mul;
  };
}

// ---------------------------------------------------------------------------
// 12 导联 PQRST 形态配置
// ---------------------------------------------------------------------------

export interface LeadConfig {
  pAmplitude: number;
  qAmplitude: number;
  rAmplitude: number;
  sAmplitude: number;
  tAmplitude: number;
  stElevation: number;
}

export const LEAD_CONFIGS: Record<string, LeadConfig> = {
  I: { pAmplitude: 0.12, qAmplitude: -0.05, rAmplitude: 0.8, sAmplitude: -0.1, tAmplitude: 0.25, stElevation: 0.0 },
  II: { pAmplitude: 0.15, qAmplitude: -0.1, rAmplitude: 1.2, sAmplitude: -0.2, tAmplitude: 0.3, stElevation: 0.0 },
  III: { pAmplitude: 0.1, qAmplitude: -0.08, rAmplitude: 0.9, sAmplitude: -0.15, tAmplitude: 0.2, stElevation: 0.0 },
  aVR: { pAmplitude: -0.1, qAmplitude: 0.05, rAmplitude: -0.8, sAmplitude: 0.1, tAmplitude: -0.2, stElevation: 0.0 },
  aVL: { pAmplitude: 0.1, qAmplitude: -0.03, rAmplitude: 0.6, sAmplitude: -0.05, tAmplitude: 0.2, stElevation: 0.0 },
  aVF: { pAmplitude: 0.13, qAmplitude: -0.09, rAmplitude: 1.0, sAmplitude: -0.18, tAmplitude: 0.28, stElevation: 0.0 },
  V1: { pAmplitude: 0.08, qAmplitude: 0.0, rAmplitude: 0.3, sAmplitude: -0.8, tAmplitude: 0.15, stElevation: 0.0 },
  V2: { pAmplitude: 0.1, qAmplitude: -0.02, rAmplitude: 0.6, sAmplitude: -0.6, tAmplitude: 0.25, stElevation: 0.0 },
  V3: { pAmplitude: 0.1, qAmplitude: -0.05, rAmplitude: 0.9, sAmplitude: -0.4, tAmplitude: 0.3, stElevation: 0.0 },
  V4: { pAmplitude: 0.12, qAmplitude: -0.08, rAmplitude: 1.3, sAmplitude: -0.25, tAmplitude: 0.35, stElevation: 0.0 },
  V5: { pAmplitude: 0.12, qAmplitude: -0.1, rAmplitude: 1.1, sAmplitude: -0.15, tAmplitude: 0.3, stElevation: 0.0 },
  V6: { pAmplitude: 0.1, qAmplitude: -0.08, rAmplitude: 0.9, sAmplitude: -0.1, tAmplitude: 0.25, stElevation: 0.0 },
};

function gaussian(x: number, amplitude: number, center: number, width: number): number {
  return amplitude * Math.exp(-((x - center) ** 2) / (2 * width ** 2));
}

function pqrstCycle(tNorm: number, cfg: LeadConfig): number {
  const p = gaussian(tNorm, cfg.pAmplitude, 0.12, 0.035);
  const q = gaussian(tNorm, cfg.qAmplitude, 0.22, 0.012);
  const r = gaussian(tNorm, cfg.rAmplitude, 0.26, 0.012);
  const s = gaussian(tNorm, cfg.sAmplitude, 0.3, 0.015);
  const tWave = gaussian(tNorm, cfg.tAmplitude, 0.48, 0.055);
  const st = tNorm > 0.32 && tNorm < 0.42 ? cfg.stElevation : 0.0;
  return p + q + r + s + tWave + st;
}

function round4(x: number): number {
  // 与 Python round(float, 4) 对齐（噪声极少落在精确的 .5 边界）
  return Math.round(x * 10000) / 10000;
}

/**
 * 生成确定性的 ECG 波形：相同的导联/心率/时长/采样率总是产生同一段信号，
 * 因此本地分析与后端分析（同样的种子）心跳数与结论一致。
 */
export function generateECGWaveform(
  leadName: string,
  duration: number,
  samplingRate: number,
  heartRate: number,
): ECGLead {
  const totalSamples = Math.floor(duration * samplingRate);
  const cfg = LEAD_CONFIGS[leadName] || LEAD_CONFIGS.II;
  const cycleDuration = 60.0 / heartRate;

  const rng = mulberry32(hashSeed(leadName, heartRate, duration, samplingRate));
  const gauss = makeGaussian(rng);
  // 每拍一个幅度抖动因子（先于逐点噪声抽取，顺序与后端一致）
  const beatCount = Math.ceil(duration / cycleDuration);
  const beatFactors = new Array<number>(beatCount);
  for (let b = 0; b < beatCount; b++) {
    beatFactors[b] = 1.0 + gauss() * 0.02;
  }

  const samples = new Array<number>(totalSamples);
  for (let i = 0; i < totalSamples; i++) {
    const time = i / samplingRate;
    const beatIndex = Math.floor(time / cycleDuration);
    const tNorm = (time % cycleDuration) / cycleDuration;
    const value =
      pqrstCycle(tNorm, cfg) * beatFactors[beatIndex] +
      0.03 * Math.sin(2 * Math.PI * 0.15 * time) +
      0.02 * gauss();
    samples[i] = round4(value);
  }

  return {
    leadName,
    samplingRate,
    duration,
    samples,
    rPeaks: [],
  };
}

// ---------------------------------------------------------------------------
// 数字滤波：与 scipy.signal.butter(2, [5,15], btype='band') + filtfilt 对齐
// ---------------------------------------------------------------------------

interface Cplx {
  re: number;
  im: number;
}

function cAdd(a: Cplx, b: Cplx): Cplx {
  return { re: a.re + b.re, im: a.im + b.im };
}
function cSub(a: Cplx, b: Cplx): Cplx {
  return { re: a.re - b.re, im: a.im - b.im };
}
function cMul(a: Cplx, b: Cplx): Cplx {
  return { re: a.re * b.re - a.im * b.im, im: a.re * b.im + a.im * b.re };
}
function cDiv(a: Cplx, b: Cplx): Cplx {
  const d = b.re * b.re + b.im * b.im;
  return { re: (a.re * b.re + a.im * b.im) / d, im: (a.im * b.re - a.re * b.im) / d };
}
function cSqrt(a: Cplx): Cplx {
  const r = Math.hypot(a.re, a.im);
  const re = Math.sqrt((r + a.re) / 2);
  const im = (a.im >= 0 ? 1 : -1) * Math.sqrt((r - a.re) / 2);
  return { re, im };
}
const cOne: Cplx = { re: 1, im: 0 };

/** 2 阶 Butterworth 带通零极点增益（等价 scipy butter(2, ..., 'band', output='zpk')） */
function butterBandpassZpk(low: number, high: number): { z: Cplx[]; p: Cplx[]; k: number } {
  // 1) N=2 Butterworth 低通原型
  const protoP: Cplx[] = [
    { re: -Math.SQRT1_2, im: Math.SQRT1_2 },
    { re: -Math.SQRT1_2, im: -Math.SQRT1_2 },
  ];
  const degree = 2;

  // 2) 预畸（scipy fs=1：wa = 2*tan(pi*wd/2)）
  const warped = (wd: number) => 2 * Math.tan((Math.PI * wd) / 2);
  const waLow = warped(low);
  const waHigh = warped(high);
  const wo2 = waLow * waHigh; // 中心频率平方
  const bwW = waHigh - waLow;

  // lp2bp_zpk 的极点公式：p' = p*bw/2 ± sqrt((p*bw/2)^2 - wo^2)
  // lp2bp_zpk：带通有 N 个 s=0 有限零点（4 极点中相对阶数为 N）
  const zBP: Cplx[] = [];
  for (let i = 0; i < degree; i++) zBP.push({ re: 0, im: 0 });
  const pBP: Cplx[] = [];
  for (const p of protoP) {
    const pScaled: Cplx = { re: p.re * (bwW / 2), im: p.im * (bwW / 2) };
    const root = cSqrt(cSub(cMul(pScaled, pScaled), { re: wo2, im: 0 }));
    pBP.push(cAdd(pScaled, root));
    pBP.push(cSub(pScaled, root));
  }
  let kBP: number = Math.pow(bwW, degree); // 原型 k=1

  // 3) bilinear_zpk（fs=1，fs2=2）：z = (2+s)/(2-s)
  const fs2 = 2;
  const bilinearMap = (sp: Cplx): Cplx =>
    cDiv({ re: fs2 + sp.re, im: sp.im }, { re: fs2 - sp.re, im: -sp.im });
  const zD: Cplx[] = zBP.map(bilinearMap);
  // 相对阶数 degree 个无穷远零点移到奈奎斯特频率 z=-1
  for (let i = 0; i < degree; i++) zD.push({ re: -1, im: 0 });
  const pD = pBP.map(bilinearMap);

  // k = k * prod(fs2 - z) / prod(fs2 - p)
  let num: Cplx = { re: 1, im: 0 };
  let den: Cplx = { re: 1, im: 0 };
  for (const z of zBP) num = cMul(num, { re: fs2 - z.re, im: -z.im });
  for (const p of pBP) den = cMul(den, { re: fs2 - p.re, im: -p.im });
  const kD = kBP * cDiv(num, den).re;

  return { z: zD, p: pD, k: kD };
}

/** 共轭根展开为实系数多项式（降幂，与 numpy.poly / scipy zpk2tf 一致） */
function rootsToPoly(roots: Cplx[]): number[] {
  // 因子按「高次在前」累积：实根 (x-r) -> [-r, 1]，共轭对 -> [|r|^2, -2Re, 1]
  let coeffs: number[] = [1];
  const mulFactor = (f: number[]) => {
    const next = new Array<number>(coeffs.length + f.length - 1).fill(0);
    for (let i = 0; i < coeffs.length; i++) {
      for (let j = 0; j < f.length; j++) {
        next[i + j] += coeffs[i] * f[j];
      }
    }
    coeffs = next;
  };
  const used = new Array<boolean>(roots.length).fill(false);
  for (let i = 0; i < roots.length; i++) {
    if (used[i]) continue;
    const r = roots[i];
    if (Math.abs(r.im) < 1e-10) {
      mulFactor([1, -r.re]);
      used[i] = true;
      continue;
    }
    let j = -1;
    for (let q = i + 1; q < roots.length; q++) {
      if (!used[q] && Math.abs(roots[q].im + r.im) < 1e-9 && Math.abs(roots[q].re - r.re) < 1e-9) {
        j = q;
        break;
      }
    }
    if (j >= 0) {
      mulFactor([1, -2 * r.re, r.re * r.re + r.im * r.im]);
      used[i] = true;
      used[j] = true;
    } else {
      mulFactor([1, -r.re]);
      used[i] = true;
    }
  }
  return coeffs;
}

/** 零极点增益 -> 传递函数多项式系数（降幂，b[0] 对应最高次） */
function zpk2tf(z: Cplx[], p: Cplx[], k: number): { b: number[]; a: number[] } {
  return { b: rootsToPoly(z).map((v) => v * k), a: rootsToPoly(p) };
}

/** Butterworth 带通系数，结果与 scipy.signal.butter 一致 */
export function bandpassCoeffs(samplingRate: number): { b: number[]; a: number[] } {
  const nyquist = samplingRate / 2;
  const { z, p, k } = butterBandpassZpk(5.0 / nyquist, 15.0 / nyquist);
  return zpk2tf(z, p, k);
}

/** 与 scipy.signal.filtfilt（默认 padtype='odd', padlen=3*max(len(a),len(b))）等价 */
export function filtfilt(b: number[], a: number[], x: number[]): number[] {
  const n = x.length;
  const padlen = 3 * Math.max(a.length, b.length);
  const edge = padlen;

  const lfilter = (bb: number[], aa: number[], sig: number[]): number[] => {
    // 归一化 a[0]=1，直接 I 型差分方程，与 scipy.signal.lfilter 等价：
    //   y[n] = b[0]x[n] + Σ_{k>=1}(b[k]x[n-k] - a[k]y[n-k])
    const nb = bb.map((v) => v / aa[0]);
    const na = aa.map((v) => v / aa[0]);
    const out = new Array<number>(sig.length).fill(0);
    for (let i = 0; i < sig.length; i++) {
      let y = nb[0] * sig[i];
      for (let k = 1; k < nb.length; k++) {
        if (i - k >= 0) y += nb[k] * sig[i - k];
      }
      for (let k = 1; k < na.length; k++) {
        if (i - k >= 0) y -= na[k] * out[i - k];
      }
      out[i] = y;
    }
    return out;
  };

  // 奇延拓（odd reflection），与 scipy.signal.filtfilt 的延拓序列一致
  const ext: number[] = new Array(n + 2 * edge);
  // 左侧：2*x[0] - x[edge..1]
  for (let i = 0; i < edge; i++) {
    ext[i] = 2 * x[0] - x[edge - i];
  }
  for (let i = 0; i < n; i++) ext[edge + i] = x[i];
  // 右侧：2*x[n-1] - x[n-2 .. n-1-edge]
  for (let i = 0; i < edge; i++) {
    ext[n + edge + i] = 2 * x[n - 1] - x[n - 2 - i];
  }

  let y = lfilter(b, a, ext);
  y.reverse();
  y = lfilter(b, a, y);
  y.reverse();
  return y.slice(edge, edge + n);
}

// ---------------------------------------------------------------------------
// Pan-Tompkins R 峰值检测（与后端同算法，aVR 等负向导联同样准确）
// ---------------------------------------------------------------------------

export function detectRPeaks(samples: number[], sr: number): RPeak[] {
  if (samples.length === 0) return [];

  const { b, a } = bandpassCoeffs(sr);
  const filtered = filtfilt(b, a, samples);

  // 微分 -> 平方
  const squared = new Array<number>(filtered.length - 1);
  for (let i = 0; i < squared.length; i++) {
    const d = filtered[i + 1] - filtered[i];
    squared[i] = d * d;
  }

  // 150ms 滑动窗口积分（居中、零填充，等价 np.convolve(x, ones/w, 'same')）
  const windowSize = Math.max(1, Math.floor(0.15 * sr));
  const offset = Math.floor((windowSize - 1) / 2);
  const m = squared.length;
  const integrated = new Array<number>(m).fill(0);
  // 前缀和（含零填充）：pref[k+1] = sum(squared[0..k-1])，越界按 0
  const pref = new Array<number>(m + 1).fill(0);
  for (let i = 0; i < m; i++) pref[i + 1] = pref[i] + squared[i];
  const rangeSum = (lo: number, hi: number): number => {
    const a = Math.max(0, lo);
    const b = Math.min(m, hi);
    return a < b ? pref[b] - pref[a] : 0;
  };
  for (let i = 0; i < m; i++) {
    // full[i+offset] 对应 squared 下标 [i+offset-(w-1), i+offset]
    integrated[i] = rangeSum(i + offset - (windowSize - 1), i + offset + 1) / windowSize;
  }

  let mean = 0;
  for (const v of integrated) mean += v;
  mean /= integrated.length;
  let variance = 0;
  for (const v of integrated) variance += (v - mean) ** 2;
  const stdDev = Math.sqrt(variance / integrated.length);
  const threshold = mean + 0.5 * stdDev;
  const minDistance = Math.floor(0.2 * sr);

  // 主波方向按整条带通信号判定（aVR 负向、V1/V2 深 S 波等导联同样准确）
  let fMax = -Infinity;
  let fMin = Infinity;
  for (const v of filtered) {
    if (v > fMax) fMax = v;
    if (v < fMin) fMin = v;
  }
  const positiveOriented = fMax >= -fMin;

  // 每个连续超阈值段只取积分能量最高的一个触发点，
  // 避免同一 QRS 的 R/S 两个能量团或宽段两端被重复计数。
  // 短于 30ms 的阈值间隙视为同一拍（滤波数值噪声不应把一个 QRS 劈成两段）
  const mergeGap = Math.max(1, Math.floor(0.03 * sr));
  const triggers: number[] = [];
  let i = 0;
  while (i < integrated.length) {
    if (integrated[i] <= threshold) {
      i++;
      continue;
    }
    let j = i;
    while (j + 1 < integrated.length) {
      if (integrated[j + 1] > threshold) {
        j++;
        continue;
      }
      // 向前看一个短间隙，若间隙后仍是超阈值区则并入本段
      let gap = 0;
      let k = j + 1;
      while (k < integrated.length && integrated[k] <= threshold && gap < mergeGap) {
        k++;
        gap++;
      }
      if (gap < mergeGap && k < integrated.length && integrated[k] > threshold) {
        j = k;
      } else {
        break;
      }
    }
    let best = i;
    for (let q = i; q <= j; q++) {
      if (integrated[q] > integrated[best]) best = q;
    }
    triggers.push(best);
    i = j + 1;
  }

  const peaks: RPeak[] = [];
  for (const trig of triggers) {
    const searchStart = Math.max(0, trig - Math.floor(windowSize / 2));
    const searchEnd = Math.min(samples.length, trig + Math.floor(windowSize / 2));
    let localPeak = searchStart;
    for (let q = searchStart + 1; q < searchEnd; q++) {
      const better = positiveOriented ? samples[q] > samples[localPeak] : samples[q] < samples[localPeak];
      if (better) localPeak = q;
    }

    // 按细化后的真实峰位去重（200ms 内不应有两次心跳）
    if (peaks.every((rp) => Math.abs(localPeak - rp.index) >= minDistance)) {
      peaks.push({ index: localPeak, time: localPeak / sr, amplitude: samples[localPeak] });
    }
  }

  peaks.sort((p, q) => p.index - q.index);
  return peaks;
}

// ---------------------------------------------------------------------------
// HRV 指标
// ---------------------------------------------------------------------------

export function calculateHRV(rPeaks: RPeak[], sr: number): HRVData {
  // 心跳太少（少于 3 拍、不足 2 个 RR 间期）不计算心率
  if (rPeaks.length < 3) {
    return { heartRate: 0, sdnn: 0, rmssd: 0, pnn50: 0, nnIntervals: [] };
  }

  const nnIntervals: number[] = [];
  for (let i = 1; i < rPeaks.length; i++) {
    nnIntervals.push(((rPeaks[i].index - rPeaks[i - 1].index) / sr) * 1000);
  }

  const n = nnIntervals.length;
  const meanRR = nnIntervals.reduce((x, y) => x + y, 0) / n;
  const hr = meanRR > 0 ? 60000 / meanRR : 0;

  const sdnn = Math.sqrt(nnIntervals.reduce((s, x) => s + (x - meanRR) ** 2, 0) / n);

  let sumSqDiff = 0;
  let nn50 = 0;
  for (let i = 1; i < n; i++) {
    const d = nnIntervals[i] - nnIntervals[i - 1];
    sumSqDiff += d * d;
    if (Math.abs(d) > 50) nn50++;
  }
  const rmssd = Math.sqrt(sumSqDiff / (n - 1));
  const pnn50 = (nn50 / (n - 1)) * 100;

  const round1 = (x: number) => Math.round(x * 10) / 10;
  const round2 = (x: number) => Math.round(x * 100) / 100;
  return {
    heartRate: round1(hr),
    sdnn: round2(sdnn),
    rmssd: round2(rmssd),
    pnn50: round2(pnn50),
    nnIntervals: nnIntervals.map((x) => round2(x)),
  };
}

// ---------------------------------------------------------------------------
// 心律失常判定
// ---------------------------------------------------------------------------

export const MIN_BEATS_FOR_RHYTHM = 3;

export function detectArrhythmias(
  rPeaks: RPeak[],
  hrv: HRVData,
  samples: number[],
  sr: number,
): ArrhythmiaEvent[] {
  const events: ArrhythmiaEvent[] = [];
  const ts = rPeaks[0]?.time ?? 0;

  // 心跳太少时不给心率过快/过慢等任何结论
  if (rPeaks.length < MIN_BEATS_FOR_RHYTHM) {
    return [
      {
        eventType: 'insufficient_data',
        confidence: 1.0,
        description: `采集时间内仅检测到 ${rPeaks.length} 次心跳，数据不足以判断心律`,
        timestamp: ts,
      },
    ];
  }

  const hr = hrv.heartRate;

  if (hr > 100) {
    events.push({
      eventType: 'tachycardia',
      confidence: Math.min(1.0, (hr - 100) / 50 + 0.6),
      description: `心率过快 (${hr.toFixed(0)} BPM)，检测到心动过速`,
      timestamp: ts,
    });
  }

  if (hr < 60 && hr > 0) {
    events.push({
      eventType: 'bradycardia',
      confidence: Math.min(1.0, (60 - hr) / 30 + 0.6),
      description: `心率过慢 (${hr.toFixed(0)} BPM)，检测到心动过缓`,
      timestamp: ts,
    });
  }

  // ST 段抬高：心率过快（>=100）时 ST 段与 T 波重叠，固定窗口不可靠，此时不下结论
  if (hr > 0 && hr < 100) {
    let stElevationCount = 0;
    let stUsable = 0;
    for (let n = 0; n < rPeaks.length; n++) {
      const rp = rPeaks[n];
      const stStart = rp.index + Math.floor(0.1 * sr);
      const stEnd = rp.index + Math.floor(0.16 * sr);
      const midpoint = n + 1 < rPeaks.length ? Math.floor((rp.index + rPeaks[n + 1].index) / 2) : samples.length;
      if (stEnd < samples.length && stEnd <= midpoint) {
        stUsable++;
        let stSum = 0;
        for (let q = stStart; q < stEnd; q++) stSum += samples[q];
        const blStart = Math.max(0, rp.index - Math.floor(0.2 * sr));
        const blEnd = Math.max(blStart, rp.index - Math.floor(0.12 * sr));
        let blSum = 0;
        for (let q = blStart; q < blEnd; q++) blSum += samples[q];
        const stLevel = stSum / (stEnd - stStart);
        const baseline = blEnd > blStart ? blSum / (blEnd - blStart) : 0;
        if (stLevel - baseline > 0.15) stElevationCount++;
      }
    }
    if (stUsable > 0 && stElevationCount > stUsable * 0.5) {
      events.push({
        eventType: 'st_elevation',
        confidence: Math.min(1.0, stElevationCount / stUsable),
        description: '检测到 ST 段抬高，可能提示心肌梗死',
        timestamp: ts,
      });
    }
  }

  // RR 间期不规则（疑似房颤），与后端一致
  if (hrv.nnIntervals.length > 3) {
    const n = hrv.nnIntervals.length;
    const mean = hrv.nnIntervals.reduce((x, y) => x + y, 0) / n;
    const sd = Math.sqrt(hrv.nnIntervals.reduce((s, x) => s + (x - mean) ** 2, 0) / n);
    const cv = mean > 0 ? sd / mean : 0;
    if (cv > 0.15) {
      events.push({
        eventType: 'atrial_fibrillation',
        confidence: Math.min(1.0, cv * 2),
        description: 'RR 间期不规则，可能提示房颤',
        timestamp: ts,
      });
    }
  }

  if (events.length === 0) {
    events.push({
      eventType: 'normal',
      confidence: 1.0,
      description: '正常窦性心律',
      timestamp: ts,
    });
  }

  return events;
}

export function getRhythmDiagnosis(events: ArrhythmiaEvent[], hrv: HRVData, rPeakCount: number): string {
  const types = events.map((e) => e.eventType);

  if (rPeakCount < MIN_BEATS_FOR_RHYTHM) {
    return `数据不足（仅 ${rPeakCount} 次心跳），无法判断心律`;
  }
  if (types.includes('st_elevation')) return 'ST 段抬高 - 建议立即就医检查';
  if (types.includes('tachycardia') && types.includes('atrial_fibrillation')) {
    return '快速房颤 - 建议进一步心脏评估';
  }
  if (types.includes('tachycardia')) return '窦性心动过速 - 请结合临床症状判断';
  if (types.includes('bradycardia')) return '窦性心动过缓 - 建议关注心率变化';
  if (types.includes('atrial_fibrillation')) return '心律不规则 - 疑似房颤，建议 Holter 监测';
  return `正常窦性心律 | HR: ${hrv.heartRate.toFixed(0)} BPM | SDNN: ${hrv.sdnn.toFixed(1)} ms`;
}
