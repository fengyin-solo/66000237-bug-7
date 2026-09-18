import { defineStore } from 'pinia';
import { ref, computed } from 'vue';
import type { ECGLead, HRVData, RPeak, ArrhythmiaEvent, ECGAnalysisResponse } from '../types';

// Gaussian function for PQRST wave simulation
function gaussian(x: number, amplitude: number, center: number, width: number): number {
  return amplitude * Math.exp(-((x - center) ** 2) / (2 * width ** 2));
}

// Lead-specific PQRST configuration
interface LeadConfig {
  pAmplitude: number;
  qAmplitude: number;
  rAmplitude: number;
  sAmplitude: number;
  tAmplitude: number;
  stElevation: number;
}

const LEAD_CONFIGS: Record<string, LeadConfig> = {
  'I': { pAmplitude: 0.12, qAmplitude: -0.05, rAmplitude: 0.8, sAmplitude: -0.1, tAmplitude: 0.25, stElevation: 0.0 },
  'II': { pAmplitude: 0.15, qAmplitude: -0.1, rAmplitude: 1.2, sAmplitude: -0.2, tAmplitude: 0.3, stElevation: 0.0 },
  'III': { pAmplitude: 0.10, qAmplitude: -0.08, rAmplitude: 0.9, sAmplitude: -0.15, tAmplitude: 0.2, stElevation: 0.0 },
  'aVR': { pAmplitude: -0.10, qAmplitude: 0.05, rAmplitude: -0.8, sAmplitude: 0.1, tAmplitude: -0.2, stElevation: 0.0 },
  'aVL': { pAmplitude: 0.10, qAmplitude: -0.03, rAmplitude: 0.6, sAmplitude: -0.05, tAmplitude: 0.2, stElevation: 0.0 },
  'aVF': { pAmplitude: 0.13, qAmplitude: -0.09, rAmplitude: 1.0, sAmplitude: -0.18, tAmplitude: 0.28, stElevation: 0.0 },
  'V1': { pAmplitude: 0.08, qAmplitude: 0.0, rAmplitude: 0.3, sAmplitude: -0.8, tAmplitude: 0.15, stElevation: 0.0 },
  'V2': { pAmplitude: 0.10, qAmplitude: -0.02, rAmplitude: 0.6, sAmplitude: -0.6, tAmplitude: 0.25, stElevation: 0.0 },
  'V3': { pAmplitude: 0.10, qAmplitude: -0.05, rAmplitude: 0.9, sAmplitude: -0.4, tAmplitude: 0.3, stElevation: 0.0 },
  'V4': { pAmplitude: 0.12, qAmplitude: -0.08, rAmplitude: 1.3, sAmplitude: -0.25, tAmplitude: 0.35, stElevation: 0.0 },
  'V5': { pAmplitude: 0.12, qAmplitude: -0.1, rAmplitude: 1.1, sAmplitude: -0.15, tAmplitude: 0.3, stElevation: 0.0 },
  'V6': { pAmplitude: 0.10, qAmplitude: -0.08, rAmplitude: 0.9, sAmplitude: -0.1, tAmplitude: 0.25, stElevation: 0.0 },
};

// Generate a single PQRST cycle at normalized time t (0 to 1)
function generatePQRSTCycle(tNorm: number, config: LeadConfig): number {
  const p = gaussian(tNorm, config.pAmplitude, 0.12, 0.035);
  const q = gaussian(tNorm, config.qAmplitude, 0.22, 0.012);
  const r = gaussian(tNorm, config.rAmplitude, 0.26, 0.012);
  const s = gaussian(tNorm, config.sAmplitude, 0.30, 0.015);
  const tWave = gaussian(tNorm, config.tAmplitude, 0.48, 0.055);
  const st = (tNorm > 0.32 && tNorm < 0.42) ? config.stElevation : 0.0;
  return p + q + r + s + tWave + st;
}

// Np.convolve(x, kernel, mode='same') with zero padding, used to mirror the
// backend's moving-window integration so local and backend results agree.
function convolveSame(x: number[], kernel: number[]): number[] {
  const n = x.length;
  const k = kernel.length;
  const out = new Array<number>(n).fill(0);
  for (let i = 0; i < n; i++) {
    let sum = 0;
    for (let j = 0; j < k; j++) {
      const xi = i + j - Math.floor((k - 1) / 2);
      if (xi >= 0 && xi < n) {
        sum += x[xi] * kernel[j];
      }
    }
    out[i] = sum;
  }
  return out;
}

export const useECGStore = defineStore('ecg', () => {
  // State
  const selectedLead = ref<string>('II');
  const heartRate = ref<number>(72);
  const samplingRate = ref<number>(500);
  const duration = ref<number>(10);
  const isMonitoring = ref<boolean>(false);
  const ecgData = ref<ECGLead | null>(null);
  const hrvData = ref<HRVData | null>(null);
  const arrhythmiaEvents = ref<ArrhythmiaEvent[]>([]);
  const rhythmDiagnosis = ref<string>('');
  const isLoading = ref<boolean>(false);
  const useBackend = ref<boolean>(false);
  const backendUrl = ref<string>('http://localhost:8000');

  let animationTimer: ReturnType<typeof setInterval> | null = null;
  let scrollOffset = ref<number>(0);

  // Getters
  const currentSamples = computed(() => ecgData.value?.samples ?? []);
  const currentRPeaks = computed(() => ecgData.value?.rPeaks ?? []);
  const currentHeartRate = computed(() => hrvData.value?.heartRate ?? 0);

  // Actions

  /**
   * Generate realistic 12-lead ECG waveform data with PQRST morphology
   */
  function generateECGWaveform(): ECGLead {
    const totalSamples = Math.floor(duration.value * samplingRate.value);
    const samples: number[] = new Array(totalSamples);
    const config = LEAD_CONFIGS[selectedLead.value] || LEAD_CONFIGS['II'];
    const cycleDuration = 60.0 / heartRate.value;
    const samplesPerCycle = Math.floor(cycleDuration * samplingRate.value);

    for (let i = 0; i < totalSamples; i++) {
      const time = i / samplingRate.value;
      const cyclePosition = (time % cycleDuration) / cycleDuration;

      // Add slight HRV variation per beat
      const beatIndex = Math.floor(time / cycleDuration);
      const hrvFactor = 1.0 + Math.sin(beatIndex * 0.7) * 0.02;

      samples[i] = generatePQRSTCycle(cyclePosition, config) * hrvFactor;

      // Add baseline wander
      samples[i] += 0.03 * Math.sin(2 * Math.PI * 0.15 * time);
      // Add small noise
      samples[i] += (Math.random() - 0.5) * 0.02;
    }

    return {
      leadName: selectedLead.value,
      samplingRate: samplingRate.value,
      duration: duration.value,
      samples,
      rPeaks: [],
    };
  }

  /**
   * Polarity-independent R-peak detection (simplified Pan-Tompkins).
   * Mirrors the backend pipeline: baseline removal -> differentiate ->
   * square -> moving-window integration -> adaptive threshold.
   *
   * QRS complexes are sharp regardless of lead polarity, so the
   * derivative energy detects positive-R leads (I/II/aVF/...) and
   * negative-R leads (aVR) alike. Each detected heartbeat is refined to
   * the largest absolute deflection in the original signal.
   */
  function detectRPeaks(samples: number[], sr: number): RPeak[] {
    const n = samples.length;
    if (n < 3) return [];

    // 1. Remove DC offset / baseline wander
    const mean = samples.reduce((a, b) => a + b, 0) / n;
    const centered = samples.map((s) => s - mean);

    // 2. Differentiate, 3. square (derivative energy, polarity independent)
    const squared: number[] = new Array(n - 1);
    for (let i = 0; i < n - 1; i++) {
      const d = centered[i + 1] - centered[i];
      squared[i] = d * d;
    }

    // 4. Moving-window integration (150 ms window), same as np.convolve(mode='same')
    const windowSize = Math.max(1, Math.floor(0.15 * sr));
    const kernel = new Array<number>(windowSize).fill(1 / windowSize);
    const integrated = convolveSame(squared, kernel);

    // 5. Adaptive threshold
    const intMean = integrated.reduce((a, b) => a + b, 0) / integrated.length;
    const intStd = Math.sqrt(
      integrated.reduce((sum, v) => sum + (v - intMean) ** 2, 0) / integrated.length
    );
    const detectionThreshold = intMean + 0.5 * intStd;
    const minDistance = Math.floor(0.2 * sr); // at most 300 BPM

    const rPeaks: RPeak[] = [];
    const usedIndices = new Set<number>();
    let lastPeak = -minDistance;

    for (let i = 1; i < integrated.length - 1; i++) {
      if (integrated[i] <= detectionThreshold) continue;
      if (i - lastPeak < minDistance) continue;

      // Refine to the largest absolute deflection (true R-peak tip,
      // positive or negative) in the original signal within ±half window
      const searchStart = Math.max(0, i - Math.floor(windowSize / 2));
      const searchEnd = Math.min(n, i + Math.floor(windowSize / 2));
      let localPeak = searchStart;
      let localMax = Math.abs(centered[searchStart]);
      for (let j = searchStart + 1; j < searchEnd; j++) {
        if (Math.abs(centered[j]) > localMax) {
          localMax = Math.abs(centered[j]);
          localPeak = j;
        }
      }

      if (!usedIndices.has(localPeak)) {
        usedIndices.add(localPeak);
        rPeaks.push({
          index: localPeak,
          time: localPeak / sr,
          amplitude: samples[localPeak],
        });
      }
      lastPeak = i;
    }

    return rPeaks;
  }

  /**
   * Calculate HRV metrics from R-peak positions
   * SDNN, RMSSD, pNN50
   */
  function calculateHRV(rPeaks: RPeak[], sr: number): HRVData {
    // Fewer than 3 peaks means fewer than 2 RR intervals: heart rate cannot
    // be estimated. Return zeroes and let the caller report insufficient
    // data instead of pretending it equals the configured simulation rate.
    if (rPeaks.length < 3) {
      return { heartRate: 0, sdnn: 0, rmssd: 0, pnn50: 0, nnIntervals: [] };
    }

    const nnIntervals: number[] = [];
    for (let i = 1; i < rPeaks.length; i++) {
      const rr = ((rPeaks[i].index - rPeaks[i - 1].index) / sr) * 1000;
      nnIntervals.push(rr);
    }

    const meanRR = nnIntervals.reduce((a, b) => a + b, 0) / nnIntervals.length;
    const hr = meanRR > 0 ? 60000 / meanRR : 0;

    // SDNN
    const variance = nnIntervals.reduce((sum, x) => sum + (x - meanRR) ** 2, 0) / nnIntervals.length;
    const sdnn = Math.sqrt(variance);

    // RMSSD
    let sumSquaredDiffs = 0;
    for (let i = 1; i < nnIntervals.length; i++) {
      sumSquaredDiffs += (nnIntervals[i] - nnIntervals[i - 1]) ** 2;
    }
    const rmssd = Math.sqrt(sumSquaredDiffs / (nnIntervals.length - 1));

    // pNN50
    let nn50Count = 0;
    for (let i = 1; i < nnIntervals.length; i++) {
      if (Math.abs(nnIntervals[i] - nnIntervals[i - 1]) > 50) {
        nn50Count++;
      }
    }
    const pnn50 = (nn50Count / (nnIntervals.length - 1)) * 100;

    return {
      heartRate: Math.round(hr * 10) / 10,
      sdnn: Math.round(sdnn * 100) / 100,
      rmssd: Math.round(rmssd * 100) / 100,
      pnn50: Math.round(pnn50 * 100) / 100,
      nnIntervals,
    };
  }

  /**
   * Arrhythmia detection: tachycardia, bradycardia, ST-elevation,
   * irregular rhythm. When too few beats were counted, no rate-based
   * conclusion (too fast / too slow) is emitted.
   */
  function detectArrhythmias(hrv: HRVData, rPeaks: RPeak[], samples: number[], sr: number): ArrhythmiaEvent[] {
    const events: ArrhythmiaEvent[] = [];
    const hr = hrv.heartRate;

    // Not enough heartbeats to judge rhythm — do not guess tachy/brady.
    if (rPeaks.length < 3 || hr <= 0) {
      return [
        {
          eventType: 'insufficient_data',
          confidence: 1.0,
          description: `仅检测到 ${rPeaks.length} 次心跳，数据不足，无法判断心率是否过快或过慢`,
          timestamp: rPeaks[0]?.time ?? 0,
        },
      ];
    }

    if (hr > 100) {
      events.push({
        eventType: 'tachycardia',
        confidence: Math.min(1.0, (hr - 100) / 50 + 0.6),
        description: `心率过快 (${hr.toFixed(0)} BPM)，检测到心动过速`,
        timestamp: rPeaks[0]?.time ?? 0,
      });
    }

    if (hr < 60) {
      events.push({
        eventType: 'bradycardia',
        confidence: Math.min(1.0, (60 - hr) / 30 + 0.6),
        description: `心率过慢 (${hr.toFixed(0)} BPM)，检测到心动过缓`,
        timestamp: rPeaks[0]?.time ?? 0,
      });
    }

    // ST-segment elevation detection. Windows scale with the beat-to-beat
    // cycle length so they land on the ST segment (normalized 0.36-0.40)
    // and PR baseline (normalized 0.06-0.11) at every heart rate; fixed
    // millisecond windows would sample the T wave during tachycardia.
    // Per-beat cycle length (samples); fall back to the mean RR for the
    // first and last beat which have no interval on one side.
    const meanPeriod =
      rPeaks.length > 1
        ? (rPeaks[rPeaks.length - 1].index - rPeaks[0].index) / (rPeaks.length - 1)
        : 0;
    const rrPeriods = rPeaks.map((rp, k) => {
      const next = rPeaks[k + 1];
      return next ? next.index - rp.index : meanPeriod;
    });
    let rrIdx = 0;
    let stElevationCount = 0;
    for (const rp of rPeaks) {
      const beatPeriod = rrPeriods[rrIdx++] ?? 0;
      if (beatPeriod <= 0) continue;
      const stStart = rp.index + Math.floor(0.1 * beatPeriod);
      const stEnd = rp.index + Math.floor(0.14 * beatPeriod);
      const blStart = Math.max(0, rp.index - Math.floor(0.2 * beatPeriod));
      const blEnd = Math.max(0, rp.index - Math.floor(0.15 * beatPeriod));
      if (stEnd < samples.length && blEnd > blStart) {
        const stLevel = samples.slice(stStart, stEnd).reduce((a, b) => a + b, 0) / (stEnd - stStart);
        const baseline = samples.slice(blStart, blEnd).reduce((a, b) => a + b, 0) / (blEnd - blStart);
        if (stLevel - baseline > 0.1) {
          stElevationCount++;
        }
      }
    }
    if (stElevationCount > rPeaks.length * 0.5) {
      events.push({
        eventType: 'st_elevation',
        confidence: Math.min(1.0, stElevationCount / Math.max(1, rPeaks.length)),
        description: '检测到 ST 段抬高，可能提示心肌梗死',
        timestamp: rPeaks[0]?.time ?? 0,
      });
    }

    // Irregular rhythm detection (high SDNN relative to mean RR)
    if (hrv.nnIntervals.length > 3) {
      const meanNN = hrv.nnIntervals.reduce((a, b) => a + b, 0) / hrv.nnIntervals.length;
      const sdNN = Math.sqrt(
        hrv.nnIntervals.reduce((sum, x) => sum + (x - meanNN) ** 2, 0) / hrv.nnIntervals.length
      );
      const cv = meanNN > 0 ? sdNN / meanNN : 0;
      if (cv > 0.15) {
        events.push({
          eventType: 'atrial_fibrillation',
          confidence: Math.min(1.0, cv * 2),
          description: 'RR 间期不规则，可能提示房颤',
          timestamp: rPeaks[0]?.time ?? 0,
        });
      }
    }

    if (events.length === 0) {
      events.push({
        eventType: 'normal',
        confidence: 1.0,
        description: '正常窦性心律',
        timestamp: rPeaks[0]?.time ?? 0,
      });
    }

    return events;
  }

  /**
   * Overall rhythm diagnosis — same wording and ordering as the backend's
   * get_rhythm_diagnosis so both analysis paths agree.
   */
  function getRhythmDiagnosis(events: ArrhythmiaEvent[], hrv: HRVData): string {
    const types = events.map((e) => e.eventType);

    if (types.includes('insufficient_data')) {
      return '数据不足，无法给出心律诊断';
    }
    if (types.includes('st_elevation')) {
      return 'ST 段抬高 - 建议立即就医检查';
    }
    if (types.includes('tachycardia') && types.includes('atrial_fibrillation')) {
      return '快速房颤 - 建议进一步心脏评估';
    }
    if (types.includes('tachycardia')) {
      return '窦性心动过速 - 请结合临床症状判断';
    }
    if (types.includes('bradycardia')) {
      return '窦性心动过缓 - 建议关注心率变化';
    }
    if (types.includes('atrial_fibrillation')) {
      return '心律不规则 - 疑似房颤，建议 Holter 监测';
    }
    return `正常窦性心律 | HR: ${hrv.heartRate.toFixed(0)} BPM | SDNN: ${hrv.sdnn.toFixed(1)} ms`;
  }

  /**
   * Run full ECG analysis (frontend simulation)
   */
  async function analyzeECG() {
    // Clear previous run up front so no stale conclusion or numbers remain
    // visible on the panel if the new run yields little/no data.
    ecgData.value = null;
    hrvData.value = null;
    arrhythmiaEvents.value = [];
    rhythmDiagnosis.value = '';
    isLoading.value = true;

    if (useBackend.value) {
      // Use backend API
      try {
        const response = await fetch(`${backendUrl.value}/ecg/analyze`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            lead_name: selectedLead.value,
            duration: duration.value,
            sampling_rate: samplingRate.value,
            heart_rate: heartRate.value,
          }),
        });
        const data = await response.json() as ECGAnalysisResponse;
        ecgData.value = {
          leadName: data.lead.lead_name,
          samplingRate: data.lead.sampling_rate,
          duration: data.lead.duration,
          samples: data.lead.samples,
          rPeaks: data.lead.r_peaks.map((rp) => ({
            index: rp.index,
            time: rp.time,
            amplitude: rp.amplitude,
          })),
        };
        hrvData.value = {
          heartRate: data.hrv.heart_rate,
          sdnn: data.hrv.sdnn,
          rmssd: data.hrv.rmssd,
          pnn50: data.hrv.pnn50,
          nnIntervals: data.hrv.nn_intervals,
        };
        arrhythmiaEvents.value = data.arrhythmia_events.map((evt) => ({
          eventType: evt.event_type,
          confidence: evt.confidence,
          description: evt.description,
          timestamp: evt.timestamp,
        }));
        rhythmDiagnosis.value = data.rhythm_diagnosis;
      } catch (error) {
        console.error('Backend API error:', error);
        // Fallback to frontend simulation
        runFrontendAnalysis();
      }
    } else {
      runFrontendAnalysis();
    }

    isLoading.value = false;
  }

  function runFrontendAnalysis() {
    const lead = generateECGWaveform();
    const peaks = detectRPeaks(lead.samples, lead.samplingRate);
    lead.rPeaks = peaks;

    const hrv = calculateHRV(peaks, lead.samplingRate);
    const events = detectArrhythmias(hrv, peaks, lead.samples, lead.samplingRate);

    ecgData.value = lead;
    hrvData.value = hrv;
    arrhythmiaEvents.value = events;
    rhythmDiagnosis.value = getRhythmDiagnosis(events, hrv);
  }

  /**
   * Start real-time monitoring simulation
   */
  function startMonitoring() {
    isMonitoring.value = true;
    analyzeECG();
    animationTimer = setInterval(() => {
      scrollOffset.value += 5;
      // Regenerate data every full cycle
      if (scrollOffset.value >= currentSamples.value.length) {
        scrollOffset.value = 0;
        analyzeECG();
      }
    }, 50);
  }

  /**
   * Stop monitoring
   */
  function stopMonitoring() {
    isMonitoring.value = false;
    if (animationTimer) {
      clearInterval(animationTimer);
      animationTimer = null;
    }
  }

  /**
   * Select a different ECG lead
   */
  function selectLead(lead: string) {
    selectedLead.value = lead;
    if (isMonitoring.value) {
      analyzeECG();
    }
  }

  /**
   * Update heart rate setting
   */
  function setHeartRate(hr: number) {
    heartRate.value = hr;
    if (isMonitoring.value) {
      analyzeECG();
    }
  }

  return {
    // State
    selectedLead,
    heartRate,
    samplingRate,
    duration,
    isMonitoring,
    ecgData,
    hrvData,
    arrhythmiaEvents,
    rhythmDiagnosis,
    isLoading,
    useBackend,
    backendUrl,
    scrollOffset,
    // Getters
    currentSamples,
    currentRPeaks,
    currentHeartRate,
    // Actions
    analyzeECG,
    startMonitoring,
    stopMonitoring,
    selectLead,
    setHeartRate,
    generateECGWaveform,
    detectRPeaks,
    calculateHRV,
    detectArrhythmias,
    getRhythmDiagnosis,
  };
});
