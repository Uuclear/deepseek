import { defineConfig } from 'vitepress'

const editLinkPattern =
  'https://github.com/Uuclear/deepseek/edit/main/docs/:path'

export default defineConfig({
  title: '从零开始学 AI',
  description: 'Python · 数学 · ML · DL · NLP · 构建 LLM · Agent · DeepSeek 实战',
  base: '/deepseek/',
  lang: 'zh-CN',
  cleanUrls: true,
  lastUpdated: true,

  markdown: {
    lineNumbers: true,
  },

  ignoreDeadLinks: [
    /^\/examples\//,
    /^\/data\//,
  ],

  head: [
    ['link', { rel: 'icon', href: '/deepseek/favicon.ico' }],
  ],

  themeConfig: {
    logo: '/logo.svg',
    nav: [
      { text: '首页', link: '/' },
      { text: '学习路线', link: '/roadmap' },
      { text: 'Part 0 环境', link: '/part-00-env/01-python-install' },
      { text: '构建 LLM', link: '/part-08-llm-build/01-tokenizer-bpe' },
      { text: 'Agent 开发', link: '/part-09-agents/01-agent-overview' },
      { text: 'DeepSeek 实战', link: '/part-06-practice/00-deployment' },
      { text: '原理深读', link: '/part-07-theory/training-guide' },
    ],

    sidebar: {
      '/part-00-env/': partSidebar('Part 0 环境与工具', [
        ['01-python-install', 'Python 与开发环境'],
        ['02-venv-git', '虚拟环境与 Git'],
      ]),
      '/part-01-python/': partSidebar('Part 1 Python 基础', [
        ['01-variables-control', '变量与控制流'],
        ['02-functions-modules', '函数与模块'],
        ['03-file-io', '文件读写'],
        ['04-numpy-basics', 'NumPy 入门'],
        ['05-pandas-intro', 'pandas 入门'],
      ]),
      '/part-02-math/': partSidebar('Part 2 数学直觉', [
        ['01-vectors-matrices', '向量与矩阵'],
        ['02-derivatives-gradients', '导数与梯度'],
        ['03-probability', '概率基础'],
        ['04-entropy', '信息论与熵'],
        ['05-loss-functions', '损失函数'],
      ]),
      '/part-03-ml/': partSidebar('Part 3 机器学习', [
        ['01-supervised-unsupervised', '监督与无监督学习'],
        ['02-linear-regression', '线性回归'],
        ['03-logistic-regression', '逻辑回归'],
        ['04-decision-trees', '决策树'],
        ['05-evaluation-metrics', '评估指标'],
        ['06-sklearn-pipeline', 'sklearn 流水线'],
      ]),
      '/part-04-dl/': partSidebar('Part 4 深度学习', [
        ['01-tensors', '张量基础'],
        ['02-autograd', '自动求导'],
        ['03-neural-networks', '全连接网络'],
        ['04-cnn', '卷积神经网络'],
        ['05-training-loop', '训练循环'],
        ['06-overfitting', '过拟合与正则'],
        ['07-save-load', '保存与加载 · MNIST'],
      ]),
      '/part-05-nlp/': partSidebar('Part 5 NLP 与 Transformer', [
        ['01-tokenization-embedding', '分词与 Embedding'],
        ['02-rnn-intuition', 'RNN 直觉'],
        ['03-attention', '注意力机制'],
        ['04-transformer', 'Transformer 结构'],
        ['05-pretraining', '预训练与 LLM 衔接'],
      ]),
      '/part-06-practice/': partSidebar('Part 6 DeepSeek 实战', [
        ['00-deployment', '部署快速入门'],
        ['01-inference', '推理与应用开发'],
        ['02-finetuning', 'SFT / LoRA 微调'],
        ['03-data-evaluation', '数据构造与评测'],
        ['04-rl-roadmap', 'RL 与进阶训练'],
        ['05-roadmap', '八周学习路线图'],
      ]),
      '/part-07-theory/': partSidebar('Part 7 原理深读', [
        ['training-guide', '训练全流程鸟瞰'],
        ['v3-pretrain-posttrain', 'V3 预训练与后训练'],
        ['v3-architecture', 'V3 架构 MoE/MLA'],
        ['grpo-rl', 'GRPO 强化学习'],
        ['r1-pipeline', 'R1 四阶段流水线'],
        ['distill-reproduction', '蒸馏与复现'],
      ]),
      '/part-08-llm-build/': partSidebar('Part 8 从零构建大模型', [
        ['01-tokenizer-bpe', 'BPE 分词器训练'],
        ['02-mini-gpt-arch', '迷你 GPT 架构'],
        ['03-pretrain-ntp', '预训练 NTP'],
        ['04-dataset-jsonl', '数据集 JSONL'],
        ['05-sft-alpaca', 'SFT 指令微调'],
        ['06-lora-qlora', 'LoRA / QLoRA'],
        ['07-eval-perplexity', '评估与困惑度'],
        ['08-scale-to-opensource', '衔接开源生态'],
      ]),
      '/part-09-agents/': partSidebar('Part 9 AI Agent 开发', [
        ['01-agent-overview', 'Agent 概述'],
        ['02-react-tools', 'ReAct 与工具'],
        ['03-function-calling', 'Function Calling'],
        ['04-single-agent', '构建单 Agent'],
        ['05-rag-agent', 'RAG Agent'],
        ['06-multi-agent', '多 Agent 协作'],
        ['07-agent-memory', 'Agent 记忆'],
        ['08-mcp-cursor', 'MCP 与 Cursor'],
      ]),
    },

    socialLinks: [
      { icon: 'github', link: 'https://github.com/Uuclear/deepseek' },
    ],

    editLink: {
      pattern: editLinkPattern,
      text: '在 GitHub 上编辑此页',
    },

    docFooter: {
      prev: '上一篇',
      next: '下一篇',
    },

    outline: {
      label: '本页目录',
    },

    search: {
      provider: 'local',
    },

    footer: {
      message: '从零开始学 AI · DeepSeek 实战与原理',
      copyright: 'MIT License',
    },
  },
})

function partSidebar(text: string, items: [string, string][]) {
  const prefix = getPartPrefix(text)
  return [
    {
      text,
      collapsed: false,
      items: items.map(([slug, title]) => ({
        text: title,
        link: `/${prefix}/${slug}`,
      })),
    },
  ]
}

function getPartPrefix(partTitle: string): string {
  const map: Record<string, string> = {
    'Part 0 环境与工具': 'part-00-env',
    'Part 1 Python 基础': 'part-01-python',
    'Part 2 数学直觉': 'part-02-math',
    'Part 3 机器学习': 'part-03-ml',
    'Part 4 深度学习': 'part-04-dl',
    'Part 5 NLP 与 Transformer': 'part-05-nlp',
    'Part 6 DeepSeek 实战': 'part-06-practice',
    'Part 7 原理深读': 'part-07-theory',
    'Part 8 从零构建大模型': 'part-08-llm-build',
    'Part 9 AI Agent 开发': 'part-09-agents',
  }
  return map[partTitle] ?? ''
}
