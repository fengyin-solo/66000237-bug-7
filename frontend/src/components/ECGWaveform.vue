<template>
  <div class="ecg-waveform-container">
    <div class="flex items-center justify-between mb-2">
      <h3 class="text-lg font-semibold text-emerald-400">
        心电图 - 导联 {{ leadName }}
      </h3>
      <div class="flex items-center gap-2">
        <span class="text-xs text-gray-400">{{ samplingRate }} Hz</span>
        <span class="text-xs text-gray-400">{{ duration }}s</span>
      </div>
    </div>
    <v-chart
      ref="chartRef"
      class="ecg-chart"
      :option="chartOption"
      autoresize
      :update-options="{ notMerge: true }"
    />
  </div>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue';
import VChart from 'vue-echarts';
import { use } from 'echarts/core';
import { CanvasRenderer } from 'echarts/renderers';
import { LineChart, ScatterChart } from 'echarts/charts';
import {
  TitleComponent,
  TooltipComponent,
  GridComponent,
  MarkPointComponent,
  DataZoomComponent,
  LegendComponent,
} from 'echarts/components';
import type { RPeak } from '../types';

use([
  CanvasRenderer,
  LineChart,
  ScatterChart,
  TitleComponent,
  TooltipComponent,
  GridComponent,
  MarkPointComponent,
  DataZoomComponent,
  LegendComponent,
]);

const props = withDefaults(
  defineProps<{
    samples: number[];
    rPeaks: RPeak[];
    leadName: string;
    samplingRate: number;
    duration: number;
    scrollOffset?: number;
  }>(),
  {
    scrollOffset: 0,
  }
);

const chartRef = ref<InstanceType<typeof VChart> | null>(null);

const chartOption = computed(() => {
  const totalSamples = props.samples.length;
  const timeAxis = Array.from({ length: totalSamples }, (_, i) =>
    (i / props.samplingRate).toFixed(3)
  );

  // Downsample for performance: show every 2nd point if too many samples
  const step = totalSamples > 5000 ? 2 : 1;
  const downsampledTime: string[] = [];
  const downsampledValues: number[] = [];
  for (let i = 0; i < totalSamples; i += step) {
    downsampledTime.push(timeAxis[i]);
    downsampledValues.push(props.samples[i]);
  }

  // R-peak markers（负向峰如 aVR 的标签放在下方）
  const rPeakMarkers = props.rPeaks.map((rp) => ({
    name: 'R',
    coord: [(rp.time).toFixed(3), rp.amplitude],
    value: `${rp.amplitude.toFixed(2)} mV`,
    symbol: 'triangle',
    symbolSize: 10,
    symbolRotate: rp.amplitude < 0 ? 180 : 0,
    itemStyle: { color: '#ef4444' },
    label: {
      show: true,
      formatter: 'R',
      color: '#ef4444',
      fontSize: 10,
      position: (rp.amplitude < 0 ? 'bottom' : 'top') as const,
    },
  }));

  // Grid lines for medical ECG appearance
  const gridLines: any[] = [];
  for (let x = 0; x <= props.duration; x += 0.2) {
    gridLines.push([
      { xAxis: x.toFixed(3), yAxis: -2 },
      { xAxis: x.toFixed(3), yAxis: 2 },
    ]);
  }
  for (let y = -2; y <= 2; y += 0.5) {
    gridLines.push([
      { xAxis: '0', yAxis: y },
      { xAxis: props.duration.toFixed(3), yAxis: y },
    ]);
  }

  return {
    backgroundColor: '#0a0a0a',
    animation: false,
    grid: {
      left: 60,
      right: 30,
      top: 30,
      bottom: 50,
    },
    tooltip: {
      trigger: 'axis',
      backgroundColor: 'rgba(0, 0, 0, 0.8)',
      borderColor: '#10b981',
      textStyle: { color: '#fff', fontSize: 12 },
      formatter: (params: any) => {
        const p = Array.isArray(params) ? params[0] : params;
        return `时间: ${p.name}s<br/>振幅: ${p.value?.toFixed(3)} mV`;
      },
    },
    xAxis: {
      type: 'category',
      data: downsampledTime,
      name: '时间 (s)',
      nameTextStyle: { color: '#9ca3af', fontSize: 11 },
      axisLine: { lineStyle: { color: '#374151' } },
      axisLabel: {
        color: '#9ca3af',
        fontSize: 10,
        interval: Math.floor(downsampledTime.length / 10),
      },
      splitLine: {
        show: true,
        lineStyle: { color: 'rgba(16, 185, 129, 0.08)', type: 'dashed' },
      },
    },
    yAxis: {
      type: 'value',
      name: 'mV',
      nameTextStyle: { color: '#9ca3af', fontSize: 11 },
      min: -1.5,
      max: 1.8,
      axisLine: { lineStyle: { color: '#374151' } },
      axisLabel: { color: '#9ca3af', fontSize: 10 },
      splitLine: {
        show: true,
        lineStyle: { color: 'rgba(16, 185, 129, 0.08)', type: 'dashed' },
      },
    },
    dataZoom: [
      {
        type: 'inside',
        xAxisIndex: 0,
        start: 0,
        end: 100,
      },
      {
        type: 'slider',
        xAxisIndex: 0,
        height: 20,
        bottom: 5,
        borderColor: '#374151',
        fillerColor: 'rgba(16, 185, 129, 0.15)',
        handleStyle: { color: '#10b981' },
        textStyle: { color: '#9ca3af' },
      },
    ],
    series: [
      {
        name: 'ECG',
        type: 'line',
        data: downsampledValues,
        showSymbol: false,
        lineStyle: {
          color: '#10b981',
          width: 1.5,
        },
        markPoint: {
          data: rPeakMarkers,
          animation: false,
        },
        z: 10,
      },
    ],
  };
});
</script>

<style scoped>
.ecg-waveform-container {
  width: 100%;
  background: #0a0a0a;
  border: 1px solid rgba(16, 185, 129, 0.2);
  border-radius: 8px;
  padding: 16px;
}

.ecg-chart {
  width: 100%;
  height: 320px;
}
</style>
