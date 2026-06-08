# 贡献图示与公式

本站支持 **Mermaid 流程图**、**KaTeX 数学公式** 与 **SVG 插图**。本文说明如何在教程文章中添加可视化内容。

## Mermaid 流程图

在 Markdown 中使用 `mermaid` 代码块：

````markdown
```mermaid
flowchart LR
  A[输入] --> B[模型] --> C[输出]
```
````

常用类型：`flowchart`、`sequenceDiagram`、`graph TD`。构建时由 `vitepress-plugin-mermaid` 渲染。

## KaTeX 数学公式

- **行内**：`$a \cdot b = \sum_i a_i b_i$`
- **块级**：

```markdown
$$
L = -\sum_{i} y_i \log(\hat{y}_i)
$$
```

由 `@mdit/plugin-katex` 渲染，样式来自 `katex/dist/katex.min.css`。

## SVG 插图

1. 将原创 SVG 放入 `docs/public/images/`
2. 在文章中引用（VitePress 会自动加上 `base: '/deepseek/'`，部署后为 `/deepseek/images/xxx.svg`）：

```markdown
![向量与矩阵示意图](/images/neural-network-layers.svg)
```

3. 建议在每个章节 intro 后增加 **「本章图示」** 小节，集中放置插图与 Mermaid。

## 配色约定

与站点主题一致：主色 `#3b82f6`（蓝）、辅色 `#8b5cf6`（紫）、背景 `#f8fafc`。SVG 请手写、教育风格，勿使用有版权的外部图片。

## 本地验证

```powershell
npm run dev
npm run build
```

构建失败时检查 Mermaid 语法（缩进、引号）与 KaTeX 转义（反斜杠 `\`）。
