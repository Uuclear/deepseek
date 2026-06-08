<script setup lang="ts">
defineProps<{
  prereq?: string
  time?: string
  output?: string
}>()
</script>

<template>
  <div class="learning-meta">
    <div v-if="prereq" class="learning-meta-item">
      <span class="learning-meta-label">前置知识</span>
      <span class="learning-meta-value" v-html="prereq" />
    </div>
    <div v-if="time" class="learning-meta-item">
      <span class="learning-meta-label">预计时间</span>
      <span class="learning-meta-value">{{ time }}</span>
    </div>
    <div v-if="output" class="learning-meta-item">
      <span class="learning-meta-label">本章产出</span>
      <span class="learning-meta-value">{{ output }}</span>
    </div>
  </div>
</template>
