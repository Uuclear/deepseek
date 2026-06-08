<script setup lang="ts">
defineProps<{
  path: string
  command?: string
  download?: string
  title?: string
}>()
</script>

<template>
  <div class="example-box">
    <div v-if="title" class="example-box-title">{{ title }}</div>
    <div class="example-box-row">
      <span class="example-box-label">路径</span>
      <code class="example-box-path">{{ path }}</code>
    </div>
    <div v-if="command" class="example-box-row">
      <span class="example-box-label">运行</span>
      <pre class="example-box-cmd"><code>{{ command }}</code></pre>
    </div>
    <div v-if="download" class="example-box-row">
      <span class="example-box-label">数据</span>
      <a class="example-box-link" :href="download">{{ download }}</a>
    </div>
  </div>
</template>
