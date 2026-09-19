import { defineStore } from 'pinia';
import { ref, computed } from 'vue';
import type { ECGLead, HRVData, ArrhythmiaEvent, ECGAnalysisResponse } from '../types';
import {
  generateECGWaveform,
  detectRPeaks,
  calculateHRV,
  detectArrhythmias,
  getRhythmDiagnosis,
} from '../utils/ecgUtils';

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
  const scrollOffset = ref<number>(0);
  // 每次分析自增，用于丢弃过期的异步响应，防止旧结论覆盖新结果
  let analysisSeq = 0;

  // Getters
  const currentSamples = computed(() => ecgData.value?.samples ?? []);
  const currentRPeaks = computed(() => ecgData.value?.rPeaks ?? []);
  const currentHeartRate = computed(() => hrvData.value?.heartRate ?? 0);

  /** 清空上一轮分析遗留的结论与数字（波形也会在新结果到达后整体替换） */
  function clearResults() {
    ecgData.value = null;
    hrvData.value = null;
    arrhythmiaEvents.value = [];
    rhythmDiagnosis.value = '';
    scrollOffset.value = 0;
  }

  /** 用一次完整分析结果整体替换面板状态 */
  function applyAnalysis(lead: ECGLead, hrv: HRVData, events: ArrhythmiaEvent[], diagnosis: string) {
    ecgData.value = lead;
    hrvData.value = hrv;
    arrhythmiaEvents.value = events;
    rhythmDiagnosis.value = diagnosis;
  }

  /**
   * 本地判定：生成与后端同源的确定性波形，并真正逐拍数心跳
   */
  function runFrontendAnalysis() {
    const lead = generateECGWaveform(selectedLead.value, duration.value, samplingRate.value, heartRate.value);
    const peaks = detectRPeaks(lead.samples, lead.samplingRate);
    lead.rPeaks = peaks;

    const hrv = calculateHRV(peaks, lead.samplingRate);
    const events = detectArrhythmias(peaks, hrv, lead.samples, lead.samplingRate);
    const diagnosis = getRhythmDiagnosis(events, hrv, peaks.length);

    applyAnalysis(lead, hrv, events, diagnosis);
  }

  /**
   * Run full ECG analysis
   */
  async function analyzeECG() {
    const seq = ++analysisSeq;
    isLoading.value = true;
    // 先清掉上一轮的旧结论与旧数字，避免残留在面板上
    clearResults();

    if (useBackend.value) {
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
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data: ECGAnalysisResponse = await response.json();
        // 期间已有更新的分析发起，则丢弃这份过期结果
        if (seq !== analysisSeq) return;
        const lead: ECGLead = {
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
        const hrv: HRVData = {
          heartRate: data.hrv.heart_rate,
          sdnn: data.hrv.sdnn,
          rmssd: data.hrv.rmssd,
          pnn50: data.hrv.pnn50,
          nnIntervals: data.hrv.nn_intervals,
        };
        const events: ArrhythmiaEvent[] = data.arrhythmia_events.map((evt) => ({
          eventType: evt.event_type,
          confidence: evt.confidence,
          description: evt.description,
          timestamp: evt.timestamp,
        }));
        applyAnalysis(lead, hrv, events, data.rhythm_diagnosis);
      } catch (error) {
        console.error('Backend API error:', error);
        // 后端不可用时回退到本地判定（仍是本次参数的结果）
        if (seq === analysisSeq) runFrontendAnalysis();
      }
    } else {
      runFrontendAnalysis();
    }

    if (seq === analysisSeq) isLoading.value = false;
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
      if (currentSamples.value.length === 0 || scrollOffset.value >= currentSamples.value.length) {
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
    if (lead === selectedLead.value) return;
    selectedLead.value = lead;
    // 立即按当前参数重新分析，避免面板残留上一条导联的结论与数字
    analyzeECG();
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
    clearResults,
    generateECGWaveform,
    detectRPeaks,
    calculateHRV,
    detectArrhythmias,
  };
});
