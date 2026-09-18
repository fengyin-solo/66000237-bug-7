export interface ECGLead {
  leadName: string;
  samplingRate: number;
  duration: number;
  samples: number[];
  rPeaks: RPeak[];
}

export interface RPeak {
  index: number;
  time: number;
  amplitude: number;
}

export interface HRVData {
  heartRate: number;
  sdnn: number;
  rmssd: number;
  pnn50: number;
  nnIntervals: number[];
}

export interface ArrhythmiaEvent {
  eventType: 'normal' | 'tachycardia' | 'bradycardia' | 'st_elevation' | 'atrial_fibrillation' | 'premature_ventricular_contraction' | 'insufficient_data';
  confidence: number;
  description: string;
  timestamp: number;
}

// Backend wire format (snake_case), as returned by POST /ecg/analyze
export interface RPeakWire {
  index: number;
  time: number;
  amplitude: number;
}

export interface HRVMetricsWire {
  heart_rate: number;
  sdnn: number;
  rmssd: number;
  pnn50: number;
  nn_intervals: number[];
}

export interface ArrhythmiaEventWire {
  event_type: ArrhythmiaEvent['eventType'];
  confidence: number;
  description: string;
  timestamp: number;
}

export interface ECGLeadWire {
  lead_name: string;
  sampling_rate: number;
  duration: number;
  samples: number[];
  r_peaks: RPeakWire[];
}

export interface ECGAnalysisResponse {
  lead: ECGLeadWire;
  hrv: HRVMetricsWire;
  arrhythmia_events: ArrhythmiaEventWire[];
  rhythm_diagnosis: string;
}

export interface ECGAnalysisRequest {
  lead_name: string;
  duration: number;
  sampling_rate: number;
  heart_rate: number;
}

export const LEAD_NAMES: string[] = [
  'I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'
];
