<script setup lang="ts">
import DefaultTheme from 'vitepress/theme'
import { useData } from 'vitepress'

const { Layout: VPDefaultLayout } = DefaultTheme
const { frontmatter, page } = useData()
</script>

<template>
  <VPDefaultLayout>
    <template #layout-bottom>
      <div v-if="frontmatter.layout !== 'home'" class="site-footer-hint">
        <span>© {{ new Date().getFullYear() }} Uuclear/deepseek</span>
        <span v-if="page.lastUpdated" class="last-updated">
          · 本页更新于 {{ new Date(page.lastUpdated).toLocaleDateString('zh-CN') }}
        </span>
      </div>
    </template>
  </VPDefaultLayout>
</template>
