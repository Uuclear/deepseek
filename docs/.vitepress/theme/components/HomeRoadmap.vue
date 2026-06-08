<script setup lang="ts">
const stages = [
  { level: 'L0-L3', label: '基础', parts: 'Part 0～5', color: 'var(--vp-c-brand-1)' },
  { level: 'L4-L5', label: '实战', parts: 'Part 6', color: 'var(--vp-c-brand-2)' },
  { level: 'L6', label: '原理', parts: 'Part 7', color: 'var(--vp-c-brand-3)' },
  { level: 'L7', label: '造模型', parts: 'Part 8', color: '#22d3ee' },
  { level: 'L8', label: '做 Agent', parts: 'Part 9', color: '#a78bfa' },
  { level: 'L9', label: '进阶', parts: 'Part 10', color: '#f472b6' },
]
</script>

<template>
  <div class="home-roadmap">
    <h2 class="home-roadmap-title">学习进度路线图</h2>
    <p class="home-roadmap-desc">静态预览 · 无需登录 · 按 Part 循序渐进</p>
    <div class="home-roadmap-track">
      <div
        v-for="(s, i) in stages"
        :key="s.level"
        class="home-roadmap-stage"
        :style="{ '--stage-color': s.color }"
      >
        <div class="home-roadmap-dot">{{ i + 1 }}</div>
        <div class="home-roadmap-level">{{ s.level }}</div>
        <div class="home-roadmap-label">{{ s.label }}</div>
        <div class="home-roadmap-parts">{{ s.parts }}</div>
      </div>
    </div>
  </div>
</template>
