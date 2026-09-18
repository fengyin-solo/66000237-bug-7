import numpy as np
from typing import List, Tuple, Dict, Any
import math


def gaussian(x: np.ndarray, amplitude: float, center: float, width: float) -> np.ndarray:
    """Generate a Gaussian function for ECG wave simulation."""
    return amplitude * np.exp(-((x - center) ** 2) / (2 * width ** 2))


def generate_pqrst_cycle(
    t_norm: np.ndarray,
    lead_config: Dict[str, float] = None,
) -> np.ndarray:
    """
    Generate one PQRST cycle from normalized phase.

    Args:
        t_norm: time normalized to the cycle length, 0..1 within one beat.

    Each wave (P, Q, R, S, T) is modeled as a Gaussian with specific
    amplitude, center position, and width for realistic morphology.
    """
    if lead_config is None:
        lead_config = {
            "p_amplitude": 0.15,
            "q_amplitude": -0.1,
            "r_amplitude": 1.0,
            "s_amplitude": -0.2,
            "t_amplitude": 0.3,
            "st_elevation": 0.0,
        }

    # P wave: atrial depolarization (starts at ~0ms, peaks at ~80ms)
    p_wave = gaussian(t_norm, lead_config["p_amplitude"], 0.12, 0.035)

    # Q wave: initial ventricular depolarization (~160ms)
    q_wave = gaussian(t_norm, lead_config["q_amplitude"], 0.22, 0.012)

    # R wave: main ventricular depolarization (~200ms, tallest peak)
    r_wave = gaussian(t_norm, lead_config["r_amplitude"], 0.26, 0.012)

    # S wave: late ventricular depolarization (~240ms)
    s_wave = gaussian(t_norm, lead_config["s_amplitude"], 0.30, 0.015)

    # T wave: ventricular repolarization (~360ms, broader)
    t_wave = gaussian(t_norm, lead_config["t_amplitude"], 0.48, 0.055)

    # ST segment elevation (if any)
    st_segment = lead_config.get("st_elevation", 0.0) * np.where(
        (t_norm > 0.32) & (t_norm < 0.42), 1.0, 0.0
    )

    return p_wave + q_wave + r_wave + s_wave + t_wave + st_segment


def get_lead_config(lead_name: str) -> Dict[str, float]:
    """Get lead-specific configuration for realistic 12-lead ECG simulation."""
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


def generate_ecg_signal(
    lead_name: str = "II",
    duration: float = 10.0,
    sampling_rate: int = 500,
    heart_rate: float = 72.0,
    noise_level: float = 0.02,
    include_arrhythmia: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a realistic ECG signal for a specified lead.
    
    Args:
        lead_name: ECG lead name (I, II, III, aVR, aVL, aVF, V1-V6)
        duration: Signal duration in seconds
        sampling_rate: Sampling rate in Hz
        heart_rate: Heart rate in BPM
        noise_level: Baseline noise amplitude
        include_arrhythmia: Whether to simulate arrhythmia events
        
    Returns:
        Tuple of (time_array, ecg_signal)
    """
    total_samples = int(duration * sampling_rate)
    t = np.arange(total_samples) / sampling_rate
    ecg = np.zeros(total_samples)

    lead_config = get_lead_config(lead_name)
    cycle_duration = 60.0 / heart_rate

    # Build the waveform on the same regular sample grid as the frontend
    # generator: each sample's beat index and phase come from integer
    # arithmetic, which keeps every PQRST feature (esp. the sharp R spike)
    # inside its own cycle. Masking float time boundaries would let rounding
    # spill R energy into adjacent windows and create phantom deflections.
    beat_index = np.floor(t / cycle_duration).astype(int)
    phase = ((t % cycle_duration) / cycle_duration)
    hrv_factor = 1.0 + 0.02 * np.sin(beat_index * 0.7)

    # Optional premature ventricular contraction beats (10% after beat 2)
    cycle_config = lead_config.copy()
    if include_arrhythmia:
        n_beats = int(np.ceil(duration / cycle_duration))
        pvc_beats = {
            b for b in range(3, n_beats) if np.random.random() < 0.1
        }
        if pvc_beats:
            is_pvc = np.isin(beat_index, list(pvc_beats))
            pvc_config = cycle_config.copy()
            pvc_config["r_amplitude"] *= 1.8
            pvc_config["t_amplitude"] *= -0.5
            pvc_config["q_amplitude"] *= 0.5
            ecg += np.where(
                is_pvc,
                generate_pqrst_cycle(phase, pvc_config),
                generate_pqrst_cycle(phase, cycle_config),
            ) * hrv_factor
        else:
            ecg += generate_pqrst_cycle(phase, cycle_config) * hrv_factor
    else:
        ecg += generate_pqrst_cycle(phase, cycle_config) * hrv_factor

    # Add baseline wander (low-frequency noise ~0.15 Hz)
    baseline_wander = 0.03 * np.sin(2 * np.pi * 0.15 * t)
    
    # Add high-frequency noise (muscle artifact); same uniform distribution
    # as the frontend generator so analysis paths see comparable signals
    noise = noise_level * (np.random.rand(total_samples) - 0.5)

    ecg = ecg + baseline_wander + noise

    return t, ecg


def pan_tompkins_r_peak_detection(
    ecg_signal: np.ndarray, sampling_rate: int = 500
) -> List[Dict[str, Any]]:
    """
    Simplified Pan-Tompkins algorithm for R-peak detection.

    Polarity-independent pipeline (shared with the frontend implementation
    so both analysis paths count the same heartbeats):
    1. Baseline (DC) removal
    2. Differentiation — highlights sharp QRS slopes
    3. Squaring — derivative energy, same for positive-R (II) and
       negative-R (aVR) leads
    4. Moving window integration (150ms)
    5. Adaptive thresholding; each hit refined to the largest absolute
       deflection of the original signal (true R-peak tip)
    """
    ecg_signal = np.asarray(ecg_signal, dtype=float)
    if ecg_signal.size < 3:
        return []

    # Step 1: Remove DC offset / baseline
    centered = ecg_signal - np.mean(ecg_signal)

    # Step 2: Differentiation
    diff_signal = np.diff(centered)

    # Step 3: Squaring (polarity-independent QRS energy)
    squared = diff_signal ** 2

    # Step 4: Moving window integration (150ms window)
    window_size = max(1, int(0.15 * sampling_rate))
    kernel = np.ones(window_size) / window_size
    integrated = np.convolve(squared, kernel, mode="same")

    # Step 5: Adaptive threshold peak detection
    threshold = np.mean(integrated) + 0.5 * np.std(integrated)
    min_distance = int(0.2 * sampling_rate)  # Minimum 200ms between peaks

    r_peaks = []
    used_indices = set()
    last_peak = -min_distance

    for i in range(1, len(integrated) - 1):
        if integrated[i] <= threshold:
            continue
        if i - last_peak < min_distance:
            continue

        # Refine to the largest absolute deflection in the original signal
        search_start = max(0, i - window_size // 2)
        search_end = min(len(ecg_signal), i + window_size // 2)
        local_peak = search_start + int(
            np.argmax(np.abs(centered[search_start:search_end]))
        )

        if local_peak not in used_indices:
            used_indices.add(local_peak)
            r_peaks.append({
                "index": int(local_peak),
                "time": float(local_peak / sampling_rate),
                "amplitude": float(ecg_signal[local_peak]),
            })
            last_peak = i

    return r_peaks


def calculate_hrv(r_peaks: List[Dict[str, Any]], sampling_rate: int = 500) -> Dict[str, Any]:
    """
    Calculate Heart Rate Variability (HRV) metrics from R-peak positions.
    
    Metrics:
    - Heart Rate (BPM)
    - SDNN: Standard deviation of NN intervals
    - RMSSD: Root mean square of successive differences
    - pNN50: Percentage of successive differences > 50ms
    """
    if len(r_peaks) < 3:
        return {
            "heart_rate": 0.0,
            "sdnn": 0.0,
            "rmssd": 0.0,
            "pnn50": 0.0,
            "nn_intervals": [],
        }

    # Calculate RR intervals in milliseconds
    rr_intervals = []
    for i in range(1, len(r_peaks)):
        rr = (r_peaks[i]["index"] - r_peaks[i - 1]["index"]) / sampling_rate * 1000
        rr_intervals.append(rr)

    rr_array = np.array(rr_intervals)

    # Heart rate from mean RR interval
    mean_rr = np.mean(rr_array)
    heart_rate = 60000.0 / mean_rr if mean_rr > 0 else 0.0

    # SDNN: Standard deviation of all NN intervals
    sdnn = float(np.std(rr_array))

    # RMSSD: Root mean square of successive differences
    successive_diffs = np.diff(rr_array)
    rmssd = float(np.sqrt(np.mean(successive_diffs ** 2))) if len(successive_diffs) > 0 else 0.0

    # pNN50: Percentage of successive differences > 50ms
    if len(successive_diffs) > 0:
        nn50_count = np.sum(np.abs(successive_diffs) > 50)
        pnn50 = float(nn50_count / len(successive_diffs) * 100)
    else:
        pnn50 = 0.0

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
    Detect arrhythmia events based on R-peaks, HRV metrics, and signal morphology.
    
    Detects:
    - Tachycardia: HR > 100 BPM
    - Bradycardia: HR < 60 BPM
    - ST-segment elevation: potential myocardial infarction
    - Irregular rhythm patterns
    """
    events = []
    heart_rate = hrv["heart_rate"]

    # Too few heartbeats (< 3 peaks, i.e. < 2 RR intervals): heart rate
    # cannot be estimated, so no too-fast / too-slow conclusion is given.
    if len(r_peaks) < 3 or heart_rate <= 0:
        events.append({
            "event_type": "insufficient_data",
            "confidence": 1.0,
            "description": (
                f"仅检测到 {len(r_peaks)} 次心跳，数据不足，"
                "无法判断心率是否过快或过慢"
            ),
            "timestamp": r_peaks[0]["time"] if r_peaks else 0.0,
        })
        return events

    # Tachycardia detection
    if heart_rate > 100:
        events.append({
            "event_type": "tachycardia",
            "confidence": min(1.0, (heart_rate - 100) / 50 + 0.6),
            "description": f"心率过快 ({heart_rate:.0f} BPM)，检测到心动过速",
            "timestamp": r_peaks[0]["time"],
        })

    # Bradycardia detection
    if heart_rate < 60:
        events.append({
            "event_type": "bradycardia",
            "confidence": min(1.0, (60 - heart_rate) / 30 + 0.6),
            "description": f"心率过慢 ({heart_rate:.0f} BPM)，检测到心动过缓",
            "timestamp": r_peaks[0]["time"],
        })

    # ST-segment elevation detection. Windows scale with the beat-to-beat
    # cycle length so they land on the ST segment (normalized 0.36-0.40)
    # and PR baseline (normalized 0.06-0.11) at every heart rate.
    if len(r_peaks) > 1:
        mean_period = (
            (r_peaks[-1]["index"] - r_peaks[0]["index"]) / (len(r_peaks) - 1)
        )
    else:
        mean_period = 0.0

    rr_periods = []
    for k, rp in enumerate(r_peaks):
        if k + 1 < len(r_peaks):
            rr_periods.append(r_peaks[k + 1]["index"] - rp["index"])
        else:
            rr_periods.append(mean_period)

    st_elevation_count = 0
    for rp, beat_period in zip(r_peaks, rr_periods):
        if beat_period <= 0:
            continue
        idx = rp["index"]
        # ST segment: +10%-14% of a cycle after the R-peak
        st_start = idx + int(0.10 * beat_period)
        st_end = idx + int(0.14 * beat_period)
        # PR baseline: 20%-15% of a cycle before the R-peak
        bl_start = max(0, idx - int(0.20 * beat_period))
        bl_end = max(0, idx - int(0.15 * beat_period))
        if st_end < len(ecg_signal) and bl_end > bl_start:
            st_level = np.mean(ecg_signal[st_start:st_end])
            baseline = np.mean(ecg_signal[bl_start:bl_end])
            elevation = st_level - baseline
            if elevation > 0.1:  # > 0.1 mV elevation
                st_elevation_count += 1

    if st_elevation_count > len(r_peaks) * 0.5:
        events.append({
            "event_type": "st_elevation",
            "confidence": min(1.0, st_elevation_count / max(1, len(r_peaks))),
            "description": "检测到 ST 段抬高，可能提示心肌梗死",
            "timestamp": r_peaks[0]["time"],
        })

    # Irregular rhythm detection (high SDNN relative to mean)
    if len(hrv.get("nn_intervals", [])) > 3:
        nn_array = np.array(hrv["nn_intervals"])
        cv = np.std(nn_array) / np.mean(nn_array) if np.mean(nn_array) > 0 else 0
        if cv > 0.15:
            events.append({
                "event_type": "atrial_fibrillation",
                "confidence": min(1.0, cv * 2),
                "description": "RR 间期不规则，可能提示房颤",
                "timestamp": r_peaks[0]["time"] if r_peaks else 0.0,
            })

    # Normal rhythm
    if not events:
        events.append({
            "event_type": "normal",
            "confidence": 1.0,
            "description": "正常窦性心律",
            "timestamp": r_peaks[0]["time"] if r_peaks else 0.0,
        })

    return events


def get_rhythm_diagnosis(arrhythmia_events: List[Dict[str, Any]], hrv: Dict[str, Any]) -> str:
    """Generate overall rhythm diagnosis based on detected events and HRV."""
    event_types = [e["event_type"] for e in arrhythmia_events]

    if "insufficient_data" in event_types:
        return "数据不足，无法给出心律诊断"
    if "st_elevation" in event_types:
        return "ST 段抬高 - 建议立即就医检查"
    elif "tachycardia" in event_types and "atrial_fibrillation" in event_types:
        return "快速房颤 - 建议进一步心脏评估"
    elif "tachycardia" in event_types:
        return "窦性心动过速 - 请结合临床症状判断"
    elif "bradycardia" in event_types:
        return "窦性心动过缓 - 建议关注心率变化"
    elif "atrial_fibrillation" in event_types:
        return "心律不规则 - 疑似房颤，建议 Holter 监测"
    else:
        hr = hrv.get("heart_rate", 0)
        sdnn = hrv.get("sdnn", 0)
        return f"正常窦性心律 | HR: {hr:.0f} BPM | SDNN: {sdnn:.1f} ms"
