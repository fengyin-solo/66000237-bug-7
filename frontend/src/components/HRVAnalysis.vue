<template>
  <div class="hrv-analysis-container">
    <h3 class="text-lg font-semibold text-cyan-400 mb-4">心率变异性分析 (HRV)</h3>

    <!-- HRV Metrics Cards -->
    <div class="grid grid-cols-2 gap-3 mb-4">
      <div class="metric-card bg-gray-800/60 rounded-lg p-3 border border-gray-700/50">
        <div class="text-xs text-gray-400 mb-1">心率 (HR)</div>
        <div class="text-2xl font-bold" :class="hrColor">
          {{ displayHR }}
          <span class="text-sm font-normal text-gray-400">BPM</span>
        </div>
        <div class="mt-1 h-1 rounded-full bg-gray-700 overflow-hidden">
          <div
            class="h-full rounded-full transition-all duration-500"
            :class="hrBarColor"
            :style="{ width: hrBarWidth + '%' }"
          />
        </div>
      </div>

      <div class="metric-card bg-gray-800/60 rounded-lg p-3 border border-gray-700/50">
        <div class="text-xs text-gray-400 mb-1">SDNN</div>
        <div class="text-2xl font-bold text-blue-400">
          {{ displaySDNN }}
          <span class="text-sm font-normal text-gray-400">ms</span>
        </div>
        <div class="mt-1 text-xs text-gray-500">
          {{ sdnnLevel }}
        </div>
      </div>

      <div class="metric-card bg-gray-800/60 rounded-lg p-3 border border-gray-700/50">
        <div class="text-xs text-gray-400 mb-1">RMSSD</div>
        <div class="text-2xl font-bold text-purple-400">
          {{ displayRMSSD }}
          <span class="text-sm font-normal text-gray-400">ms</span>
        </div>
        <div class="mt-1 text-xs text-gray-500">副交感神经活性指标</div>
      </div>

      <div class="metric-card bg-gray-800/60 rounded-lg p-3 border border-gray-700/50">
        <div class="text-xs text-gray-400 mb-1">pNN50</div>
        <div class="text-2xl font-bold text-amber-400">
          {{ displayPNN50 }}
          <span class="text-sm font-normal text-gray-400">%</span>
        </div>
        <div class="mt-1 text-xs text-gray-500">相邻 RR 差值 > 50ms 占比</div>
      </div>
    </div>

    <!-- Tachogram Chart -->
    <div class="tachogram-section">
      <h4 class="text-sm font-medium text-gray-300 mb-2">RR 间期图 (Tachogram)</h4>
      <v-chart class="tachogram-chart" :option="tachogramOption" autoresize />
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue';
import VChart from 'vue-echarts';
import { use } from 'echarts/core';
import { CanvasRenderer } from 'echarts/renderers';
import { LineChart } from 'echarts/charts';
import { TooltipComponent, GridComponent, VisualMapComponent } from 'echarts/components';
import type { HRVData } from '../types';

use([CanvasRenderer, LineChart, TooltipComponent, GridComponent, VisualMapComponent]);

const props = defineProps<{
  hrvData: HRVData | null;
}>();

const hrColor = computed(() => {
  if (!props.hrvData || props.hrvData.heartRate <= 0) return 'text-gray-500';
  const hr = props.hrvData.heartRate;
  if (hr > 100) return 'text-red-400';
  if (hr < 60) return 'text-yellow-400';
  return 'text-emerald-400';
});

const displayHR = computed(() => {
  const hr = props.hrvData?.heartRate ?? 0;
  return hr > 0 ? hr.toFixed(1) : '--';
});

const insufficient = computed(() => (props.hrvData?.heartRate ?? 0) <= 0);
const displaySDNN = computed(() => (insufficient.value ? '--' : props.hrvData!.sdnn.toFixed(1)));
const displayRMSSD = computed(() => (insufficient.value ? '--' : props.hrvData!.rmssd.toFixed(1)));
const displayPNN50 = computed(() => (insufficient.value ? '--' : props.hrvData!.pnn50.toFixed(1)));

const hrBarColor = computed(() => {
  if (!props.hrvData || props.hrvData.heartRate <= 0) return 'bg-gray-600';
  const hr = props.hrvData.heartRate;
  if (hr > 100) return 'bg-red-500';
  if (hr < 60) return 'bg-yellow-500';
  return 'bg-emerald-500';
});

const hrBarWidth = computed(() => {
  if (!props.hrvData || props.hrvData.heartRate <= 0) return 0;
  return Math.min(100, (props.hrvData.heartRate / 180) * 100);
});

const sdnnLevel = computed(() => {
  if (!props.hrvData || insufficient.value) return '数据不足';
  const sdnn = props.hrvData.sdnn;
  if (sdnn > 50) return 'HRV 正常';
  if (sdnn > 20) return 'HRV 偏低';
  return 'HRV 过低';
});

const tachogramOption = computed(() => {
  const intervals = props.hrvData?.nnIntervals ?? [];
  if (intervals.length === 0) {
    return {
      backgroundColor: 'transparent',
      xAxis: { type: 'category', data: [] },
      yAxis: { type: 'value' },
      series: [],
    };
  }

  const data = intervals.map((v, i) => [i + 1, v]);
  const meanRR = intervals.reduce((a, b) => a + b, 0) / intervals.length;

  return {
    backgroundColor: 'transparent',
    animation: false,
    grid: { left: 50, right: 20, top: 20, bottom: 40 },
    tooltip: {
      trigger: 'axis',
      backgroundColor: 'rgba(0, 0, 0, 0.8)',
      borderColor: '#06b6d4',
      textStyle: { color: '#fff', fontSize: 11 },
      formatter: (params: any) => {
        const p = Array.isArray(params) ? params[0] : params;
        return `第 ${p.data[0]} 拍<br/>RR: ${p.data[1].toFixed(1)} ms`;
      },
    },
    xAxis: {
      type: 'value',
      name: '心跳序号',
      nameTextStyle: { color: '#9ca3af', fontSize: 10 },
      axisLine: { lineStyle: { color: '#374151' } },
      axisLabel: { color: '#9ca3af', fontSize: 10 },
      splitLine: { show: false },
    },
    yAxis: {
      type: 'value',
      name: 'RR (ms)',
      nameTextStyle: { color: '#9ca3af', fontSize: 10 },
      axisLine: { lineStyle: { color: '#374151' } },
      axisLabel: { color: '#9ca3af', fontSize: 10 },
      splitLine: {
        show: true,
        lineStyle: { color: 'rgba(6, 182, 212, 0.1)', type: 'dashed' },
      },
    },
    series: [
      {
        name: 'RR 间期',
        type: 'line',
        data: data,
        showSymbol: true,
        symbolSize: 4,
        lineStyle: { color: '#06b6d4', width: 1.5 },
        itemStyle: { color: '#06b6d4' },
        markLine: {
          silent: true,
          data: [
            {
              yAxis: meanRR,
              lineStyle: { color: '#f59e0b', type: 'dashed', width: 1 },
              label: {
                formatter: `均值: ${meanRR.toFixed(0)} ms`,
                color: '#f59e0b',
                fontSize: 10,
              },
            },
          ],
        },
      },
    ],
  };
});
</script>

<style scoped>
.hrv-analysis-container {
  width: 100%;
  background: rgba(15, 23, 42, 0.6);
  border: 1px solid rgba(6, 182, 212, 0.2);
  border-radius: 8px;
  padding: 16px;
}

.tachogram-chart {
  width: 100%;
  height: 180px;
}

.metric-card {
  transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.metric-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
}
</style>
