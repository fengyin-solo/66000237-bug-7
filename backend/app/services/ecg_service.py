import math
from typing import List, Dict, Any, Tuple

import numpy as np
from scipy import signal


# ---------------------------------------------------------------------------
# 确定性随机数：与前端 src/utils/ecgUtils.ts 中的 mulberry32 + Box-Muller 对齐，
# 相同参数总是生成同一段信号，保证本地/后端分析结果一致
# ---------------------------------------------------------------------------

def _hash_seed(lead: str, heart_rate: float, duration: float, sampling_rate: int) -> int:
    def fmt(v: Any) -> str:
        # 整数不带 .0，与前端 JS 字符串插值保持一致
        if isinstance(v, float) and v.is_integer():
            return str(int(v))
        return str(v)

    key = f"{lead}|{fmt(heart_rate)}|{fmt(duration)}|{fmt(sampling_rate)}"
    h = 0x811C9DC5
    for ch in key:
        h = ((h ^ ord(ch)) * 0x01000193) & 0xFFFFFFFF
    return h


def _mulberry32(seed: int):
    a = seed & 0xFFFFFFFF

    def rng() -> float:
        nonlocal a
        a = (a + 0x6D2B79F5) & 0xFFFFFFFF
        t = a
        t = ((t ^ (t >> 15)) * (t | 1)) & 0xFFFFFFFF
        t ^= (t + (((t ^ (t >> 7)) * (t | 61)) & 0xFFFFFFFF)) & 0xFFFFFFFF
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296

    return rng


def _gaussian(rng):
    spare = None

    def nxt() -> float:
        nonlocal spare
        if spare is not None:
            v = spare
            spare = None
            return v
        while True:
            u = rng() * 2 - 1
            v = rng() * 2 - 1
            s = u * u + v * v
            if 0 < s < 1:
                break
        mul = math.sqrt((-2 * math.log(s)) / s)
        spare = v * mul
        return u * mul

    return nxt


# ---------------------------------------------------------------------------
# 12 导联 PQRST 形态配置
# ---------------------------------------------------------------------------

def get_lead_config(lead_name: str) -> Dict[str, float]:
    configs = {
        "I": {"p_amplitude": 0.12, "q_amplitude": -0.05, "r_amplitude": 0.8, "s_amplitude": -0.1, "t_amplitude": 0.25, "st_elevation": 0.0},
        "II": {"p_amplitude": 0.15, "q_amplitude": -0.1, "r_amplitude": 1.2, "s_amplitude": -0.2, "t_amplitude": 0.3, "st_elevation": 0.0},
        "III": {"p_amplitude": 0.10, "q_amplitude": -0.08, "r_amplitude": 0.9, "s_amplitude": -0.15, "t_amplitude": 0.2, "st_elevation": 0.0},
        "aVR": {"p_amplitude": -0.10, "q_amplitude": 0.05, "r_amplitude": -0.8, "s_amplitude": 0.1, "t_amplitude": -0.2, "st_elevation": 0.0},
        "aVL": {"p_amplitude": 0.10, "q_amplitude": -0.03, "r_amplitude": 0.6, "s_amplitude": -0.05, "t_amplitude": 0.2, "st_elevation": 0.0},
        "aVF": {"p_amplitude": 0.13, "q_amplitude": -0.09, "r_amplitude": 1.0, "s_amplitude": -0.18, "t_amplitude": 0.28, "st_elevation": 0.0},
        "V1": {"p_amplitude": 0.08, "q_amplitude": 0.0, "r_amplitude": 0.3, "s_amplitude": -0.8, "t_amplitude": 0.15, "st_elevation": 0.0},
        "V2": {"p_amplitude": 0.10, "q_amplitude": -0.02, "r_amplitude": 0.6, "s_amplitude": -0.6, "t_amplitude": 0.25, "st_elevation": 0.0},
        "V3": {"p_amplitude": 0.10, "q_amplitude": -0.05, "r_amplitude": 0.9, "s_amplitude": -0.4, "t_amplitude": 0.3, "st_elevation": 0.0},
        "V4": {"p_amplitude": 0.12, "q_amplitude": -0.08, "r_amplitude": 1.3, "s_amplitude": -0.25, "t_amplitude": 0.35, "st_elevation": 0.0},
        "V5": {"p_amplitude": 0.12, "q_amplitude": -0.1, "r_amplitude": 1.1, "s_amplitude": -0.15, "t_amplitude": 0.3, "st_elevation": 0.0},
        "V6": {"p_amplitude": 0.10, "q_amplitude": -0.08, "r_amplitude": 0.9, "s_amplitude": -0.1, "t_amplitude": 0.25, "st_elevation": 0.0},
    }
    return configs.get(lead_name, configs["II"])


def _gaussian_wave(x: float, amplitude: float, center: float, width: float) -> float:
    return amplitude * math.exp(-((x - center) ** 2) / (2 * width ** 2))


def _pqrst_cycle(t_norm: float, cfg: Dict[str, float]) -> float:
    return (
        _gaussian_wave(t_norm, cfg["p_amplitude"], 0.12, 0.035)
        + _gaussian_wave(t_norm, cfg["q_amplitude"], 0.22, 0.012)
        + _gaussian_wave(t_norm, cfg["r_amplitude"], 0.26, 0.012)
        + _gaussian_wave(t_norm, cfg["s_amplitude"], 0.30, 0.015)
        + _gaussian_wave(t_norm, cfg["t_amplitude"], 0.48, 0.055)
        + (cfg["st_elevation"] if 0.32 < t_norm < 0.42 else 0.0)
    )


def _round_half_up(x: float, ndigits: int) -> float:
    factor = 10 ** ndigits
    # 与前端 Math.round 一致（正数四舍五入；此处信号取值为正偏移场景）
    return math.floor(x * factor + 0.5) / factor


def generate_ecg_signal(
    lead_name: str = "II",
    duration: float = 10.0,
    sampling_rate: int = 500,
    heart_rate: float = 72.0,
    noise_level: float = 0.02,
    include_arrhythmia: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成确定性 ECG 信号：相同参数总是产生同一段波形，
    与前端 generateECGWaveform 逐样本一致。
    """
    total_samples = int(duration * sampling_rate)
    cfg = get_lead_config(lead_name)
    cycle_duration = 60.0 / heart_rate

    rng = _mulberry32(_hash_seed(lead_name, heart_rate, duration, sampling_rate))
    gauss = _gaussian(rng)

    # 先按拍抽取幅度抖动因子（顺序必须与前端一致）
    beat_count = math.ceil(duration / cycle_duration)
    beat_factors = [1.0 + gauss() * 0.02 for _ in range(beat_count)]

    t = np.arange(total_samples) / sampling_rate
    samples = np.zeros(total_samples)
    for i in range(total_samples):
        time = i / sampling_rate
        beat_index = int(time // cycle_duration)
        t_norm = (time % cycle_duration) / cycle_duration
        value = (
            _pqrst_cycle(t_norm, cfg) * beat_factors[beat_index]
            + 0.03 * math.sin(2 * math.pi * 0.15 * time)
            + noise_level * gauss()
        )
        samples[i] = _round_half_up(value, 4)

    return t, samples


def pan_tompkins_r_peak_detection(
    ecg_signal: np.ndarray, sampling_rate: int = 500
) -> List[Dict[str, Any]]:
    """
    Pan-Tompkins 算法检测 R 峰：
    5-15Hz 带通 -> 微分 -> 平方 -> 150ms 滑动积分 -> 自适应阈值。
    与前端 detectRPeaks 使用同一套系数与流程。
    """
    if len(ecg_signal) == 0:
        return []

    nyquist = sampling_rate / 2
    low = 5.0 / nyquist
    high = 15.0 / nyquist
    b, a = signal.butter(2, [low, high], btype="band")
    filtered = signal.filtfilt(b, a, ecg_signal)

    diff_signal = np.diff(filtered)
    squared = diff_signal ** 2

    window_size = max(1, int(0.15 * sampling_rate))
    kernel = np.ones(window_size) / window_size
    integrated = np.convolve(squared, kernel, mode="same")

    threshold = float(np.mean(integrated) + 0.5 * np.std(integrated))
    min_distance = int(0.2 * sampling_rate)

    # 主波方向按整条带通信号判定（aVR 负向、V1/V2 深 S 波等导联同样准确）
    positive_oriented = float(np.max(filtered)) >= -float(np.min(filtered))

    # 每个连续超阈值段只取积分能量最高的一个触发点，
    # 避免同一 QRS 的 R/S 两个能量团或宽段两端被重复计数。
    # 短于 30ms 的阈值间隙视为同一拍（滤波数值噪声不应把一个 QRS 劈成两段）
    above_threshold = integrated > threshold
    merge_gap = max(1, int(0.03 * sampling_rate))
    triggers: List[int] = []
    above_idx = np.where(above_threshold)[0]
    if len(above_idx) > 0:
        seg_start = int(above_idx[0])
        prev = int(above_idx[0])
        for q in above_idx[1:]:
            q = int(q)
            if q - prev - 1 > merge_gap:
                seg = integrated[seg_start:prev + 1]
                triggers.append(seg_start + int(np.argmax(seg)))
                seg_start = q
            prev = q
        seg = integrated[seg_start:prev + 1]
        triggers.append(seg_start + int(np.argmax(seg)))

    r_peaks: List[Dict[str, Any]] = []
    for i in triggers:
        search_start = max(0, i - window_size // 2)
        search_end = min(len(ecg_signal), i + window_size // 2)
        window = ecg_signal[search_start:search_end]
        offset = int(np.argmax(window)) if positive_oriented else int(np.argmin(window))
        local_peak = search_start + offset

        # 按细化后的真实峰位去重（200ms 不应出现两次心跳）
        if all(abs(local_peak - rp["index"]) >= min_distance for rp in r_peaks):
            r_peaks.append({
                "index": int(local_peak),
                "time": float(local_peak / sampling_rate),
                "amplitude": float(ecg_signal[local_peak]),
            })

    r_peaks.sort(key=lambda rp: rp["index"])
    return r_peaks


MIN_BEATS_FOR_RHYTHM = 3


def calculate_hrv(r_peaks: List[Dict[str, Any]], sampling_rate: int = 500) -> Dict[str, Any]:
    """
    计算 HRV 指标（HR / SDNN / RMSSD / pNN50）。
    心跳少于 3 次（不足 2 个 RR 间期）时不给出心率。
    """
    if len(r_peaks) < MIN_BEATS_FOR_RHYTHM:
        return {
            "heart_rate": 0.0,
            "sdnn": 0.0,
            "rmssd": 0.0,
            "pnn50": 0.0,
            "nn_intervals": [],
        }

    rr_intervals = [
        (r_peaks[i]["index"] - r_peaks[i - 1]["index"]) / sampling_rate * 1000
        for i in range(1, len(r_peaks))
    ]
    rr_array = np.array(rr_intervals)

    mean_rr = float(np.mean(rr_array))
    heart_rate = 60000.0 / mean_rr if mean_rr > 0 else 0.0
    sdnn = float(np.std(rr_array))
    successive_diffs = np.diff(rr_array)
    rmssd = float(np.sqrt(np.mean(successive_diffs ** 2))) if len(successive_diffs) > 0 else 0.0
    pnn50 = float(np.sum(np.abs(successive_diffs) > 50) / len(successive_diffs) * 100) if len(successive_diffs) > 0 else 0.0

    return {
        "heart_rate": round(heart_rate, 1),
        "sdnn": round(sdnn, 2),
        "rmssd": round(rmssd, 2),
        "pnn50": round(pnn50, 2),
        "nn_intervals": [round(float(x), 2) for x in rr_intervals],
    }


def detect_arrhythmia(
    r_peaks: List[Dict[str, Any]],
    hrv: Dict[str, Any],
    ecg_signal: np.ndarray,
    sampling_rate: int = 500,
) -> List[Dict[str, Any]]:
    """
    心律失常判定。心跳太少（<3 次）时返回 insufficient_data，
    不做心动过速/过缓等任何结论。
    """
    events = []
    timestamp = r_peaks[0]["time"] if r_peaks else 0.0

    if len(r_peaks) < MIN_BEATS_FOR_RHYTHM:
        events.append({
            "event_type": "insufficient_data",
            "confidence": 1.0,
            "description": f"采集时间内仅检测到 {len(r_peaks)} 次心跳，数据不足以判断心律",
            "timestamp": timestamp,
        })
        return events

    heart_rate = hrv["heart_rate"]

    if heart_rate > 100:
        events.append({
            "event_type": "tachycardia",
            "confidence": min(1.0, (heart_rate - 100) / 50 + 0.6),
            "description": f"心率过快 ({heart_rate:.0f} BPM)，检测到心动过速",
            "timestamp": timestamp,
        })

    if 0 < heart_rate < 60:
        events.append({
            "event_type": "bradycardia",
            "confidence": min(1.0, (60 - heart_rate) / 30 + 0.6),
            "description": f"心率过慢 ({heart_rate:.0f} BPM)，检测到心动过缓",
            "timestamp": timestamp,
        })

    # ST 段抬高：心率过快（>=100）时 ST 段与 T 波重叠，固定窗口不可靠，此时不下结论
    if 0 < heart_rate < 100 and len(r_peaks) > 0:
        st_elevation_count = 0
        st_usable = 0
        peak_indices = [rp["index"] for rp in r_peaks]
        for n, rp in enumerate(r_peaks):
            idx = rp["index"]
            # ST 段取 R 后 100-160ms；不得越过本拍与下一拍的中点（之后是 T 波/下一个 QRS）
            st_start = idx + int(0.10 * sampling_rate)
            st_end = idx + int(0.16 * sampling_rate)
            midpoint = (idx + peak_indices[n + 1]) // 2 if n + 1 < len(r_peaks) else len(ecg_signal)
            if st_end < len(ecg_signal) and st_end <= midpoint:
                st_usable += 1
                st_level = float(np.mean(ecg_signal[st_start:st_end]))
                baseline_start = max(0, idx - int(0.20 * sampling_rate))
                baseline_end = max(baseline_start, idx - int(0.12 * sampling_rate))
                baseline = float(np.mean(ecg_signal[baseline_start:baseline_end]))
                if st_level - baseline > 0.15:
                    st_elevation_count += 1

        if st_usable > 0 and st_elevation_count > st_usable * 0.5:
            events.append({
                "event_type": "st_elevation",
                "confidence": min(1.0, st_elevation_count / st_usable),
                "description": "检测到 ST 段抬高，可能提示心肌梗死",
                "timestamp": timestamp,
            })

    # RR 间期不规则
    nn_intervals = hrv.get("nn_intervals", [])
    if len(nn_intervals) > 3:
        nn_array = np.array(nn_intervals)
        mean = float(np.mean(nn_array))
        cv = float(np.std(nn_array) / mean) if mean > 0 else 0.0
        if cv > 0.15:
            events.append({
                "event_type": "atrial_fibrillation",
                "confidence": min(1.0, cv * 2),
                "description": "RR 间期不规则，可能提示房颤",
                "timestamp": timestamp,
            })

    if not events:
        events.append({
            "event_type": "normal",
            "confidence": 1.0,
            "description": "正常窦性心律",
            "timestamp": timestamp,
        })

    return events


def get_rhythm_diagnosis(
    arrhythmia_events: List[Dict[str, Any]],
    hrv: Dict[str, Any],
    r_peak_count: int = 0,
) -> str:
    """根据检测事件生成整体结论。"""
    if r_peak_count < MIN_BEATS_FOR_RHYTHM:
        return f"数据不足（仅 {r_peak_count} 次心跳），无法判断心律"

    event_types = [e["event_type"] for e in arrhythmia_events]
    if "st_elevation" in event_types:
        return "ST 段抬高 - 建议立即就医检查"
    if "tachycardia" in event_types and "atrial_fibrillation" in event_types:
        return "快速房颤 - 建议进一步心脏评估"
    if "tachycardia" in event_types:
        return "窦性心动过速 - 请结合临床症状判断"
    if "bradycardia" in event_types:
        return "窦性心动过缓 - 建议关注心率变化"
    if "atrial_fibrillation" in event_types:
        return "心律不规则 - 疑似房颤，建议 Holter 监测"

    hr = hrv.get("heart_rate", 0)
    sdnn = hrv.get("sdnn", 0)
    return f"正常窦性心律 | HR: {hr:.0f} BPM | SDNN: {sdnn:.1f} ms"
