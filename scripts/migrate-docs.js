const fs = require('fs');
const path = require('path');

const root = path.resolve(__dirname, '..');

const migrations = [
  ['guide-00-deployment-quickstart-zh.md', 'docs/part-06-practice/00-deployment.md'],
  ['guide-01-inference-and-app-dev-zh.md', 'docs/part-06-practice/01-inference.md'],
  ['guide-02-finetuning-sft-zh.md', 'docs/part-06-practice/02-finetuning.md'],
  ['guide-03-data-and-evaluation-zh.md', 'docs/part-06-practice/03-data-evaluation.md'],
  ['guide-04-rl-and-training-roadmap-zh.md', 'docs/part-06-practice/04-rl-roadmap.md'],
  ['guide-05-ai-dev-roadmap-zh.md', 'docs/part-06-practice/05-roadmap.md'],
  ['deepseek-training-guide-zh.md', 'docs/part-07-theory/training-guide.md'],
  ['deepseek-v3-pretrain-posttrain-deep-dive-zh.md', 'docs/part-07-theory/v3-pretrain-posttrain.md'],
  ['deepseek-v3-architecture-deep-dive-zh.md', 'docs/part-07-theory/v3-architecture.md'],
  ['deepseek-grpo-rl-deep-dive-zh.md', 'docs/part-07-theory/grpo-rl.md'],
  ['deepseek-r1-pipeline-deep-dive-zh.md', 'docs/part-07-theory/r1-pipeline.md'],
  ['deepseek-distill-reproduction-deep-dive-zh.md', 'docs/part-07-theory/distill-reproduction.md'],
];

const linkReplacements = [
  [/\.\/guide-00-deployment-quickstart-zh\.md/g, '/part-06-practice/00-deployment'],
  [/\.\/guide-01-inference-and-app-dev-zh\.md/g, '/part-06-practice/01-inference'],
  [/\.\/guide-02-finetuning-sft-zh\.md/g, '/part-06-practice/02-finetuning'],
  [/\.\/guide-03-data-and-evaluation-zh\.md/g, '/part-06-practice/03-data-evaluation'],
  [/\.\/guide-04-rl-and-training-roadmap-zh\.md/g, '/part-06-practice/04-rl-roadmap'],
  [/\.\/guide-05-ai-dev-roadmap-zh\.md/g, '/part-06-practice/05-roadmap'],
  [/\[guide-00[^\]]*\]\(\.\/guide-00[^)]+\)/g, '[guide-00 部署](/part-06-practice/00-deployment)'],
  [/\[guide-01[^\]]*\]\(\.\/guide-01[^)]+\)/g, '[guide-01 应用](/part-06-practice/01-inference)'],
  [/\[guide-02[^\]]*\]\(\.\/guide-02[^)]+\)/g, '[guide-02 微调](/part-06-practice/02-finetuning)'],
  [/\[guide-03[^\]]*\]\(\.\/guide-03[^)]+\)/g, '[guide-03 数据](/part-06-practice/03-data-evaluation)'],
  [/\[guide-04[^\]]*\]\(\.\/guide-04[^)]+\)/g, '[guide-04 RL](/part-06-practice/04-rl-roadmap)'],
  [/\[guide-05[^\]]*\]\(\.\/guide-05[^)]+\)/g, '[guide-05 路线图](/part-06-practice/05-roadmap)'],
  [/\.\/deepseek-training-guide-zh\.md/g, '/part-07-theory/training-guide'],
  [/\.\/deepseek-v3-pretrain-posttrain-deep-dive-zh\.md/g, '/part-07-theory/v3-pretrain-posttrain'],
  [/\.\/deepseek-v3-architecture-deep-dive-zh\.md/g, '/part-07-theory/v3-architecture'],
  [/\.\/deepseek-grpo-rl-deep-dive-zh\.md/g, '/part-07-theory/grpo-rl'],
  [/\.\/deepseek-r1-pipeline-deep-dive-zh\.md/g, '/part-07-theory/r1-pipeline'],
  [/\.\/deepseek-distill-reproduction-deep-dive-zh\.md/g, '/part-07-theory/distill-reproduction'],
  [/\[training-guide\]\(\.\/deepseek-training-guide-zh\.md\)/g, '[training-guide](/part-07-theory/training-guide)'],
  [/\[GRPO 深挖\]\(\.\/deepseek-grpo-rl-deep-dive-zh\.md\)/g, '[GRPO 深挖](/part-07-theory/grpo-rl)'],
  [/\[R1 pipeline\]\(\.\/deepseek-r1-pipeline-deep-dive-zh\.md\)/g, '[R1 pipeline](/part-07-theory/r1-pipeline)'],
  [/\[distill 复现\]\(\.\/deepseek-distill-reproduction-deep-dive-zh\.md\)/g, '[distill 复现](/part-07-theory/distill-reproduction)'],
  [/\[训练解读\]\(\.\/deepseek-training-guide-zh\.md\)/g, '[训练解读](/part-07-theory/training-guide)'],
  [/\[微调\]\(\.\/guide-02-finetuning-sft-zh\.md\)/g, '[微调](/part-06-practice/02-finetuning)'],
  [/\[训练\]\(\.\/guide-04-rl-and-training-roadmap-zh\.md\)/g, '[训练](/part-06-practice/04-rl-roadmap)'],
  [/\[guide-01\]\(\.\/guide-01-inference-and-app-dev-zh\.md\)/g, '[guide-01](/part-06-practice/01-inference)'],
];

const stubMap = {
  'guide-00-deployment-quickstart-zh.md': '/part-06-practice/00-deployment',
  'guide-01-inference-and-app-dev-zh.md': '/part-06-practice/01-inference',
  'guide-02-finetuning-sft-zh.md': '/part-06-practice/02-finetuning',
  'guide-03-data-and-evaluation-zh.md': '/part-06-practice/03-data-evaluation',
  'guide-04-rl-and-training-roadmap-zh.md': '/part-06-practice/04-rl-roadmap',
  'guide-05-ai-dev-roadmap-zh.md': '/part-06-practice/05-roadmap',
  'deepseek-training-guide-zh.md': '/part-07-theory/training-guide',
  'deepseek-v3-pretrain-posttrain-deep-dive-zh.md': '/part-07-theory/v3-pretrain-posttrain',
  'deepseek-v3-architecture-deep-dive-zh.md': '/part-07-theory/v3-architecture',
  'deepseek-grpo-rl-deep-dive-zh.md': '/part-07-theory/grpo-rl',
  'deepseek-r1-pipeline-deep-dive-zh.md': '/part-07-theory/r1-pipeline',
  'deepseek-distill-reproduction-deep-dive-zh.md': '/part-07-theory/distill-reproduction',
};

function fixLinks(content) {
  let result = content;
  for (const [pattern, replacement] of linkReplacements) {
    result = result.replace(pattern, replacement);
  }
  return result;
}

for (const [src, dest] of migrations) {
  const srcPath = path.join(root, src);
  const destPath = path.join(root, dest);
  fs.mkdirSync(path.dirname(destPath), { recursive: true });
  let content = fs.readFileSync(srcPath, 'utf8');
  content = fixLinks(content);
  fs.writeFileSync(destPath, content, 'utf8');
  console.log(`Migrated: ${src} → ${dest}`);
}

for (const [filename, route] of Object.entries(stubMap)) {
  const stub = `# 文档已迁移

本文已迁移至 VitePress 学习网站：

**[点击阅读新地址](${route})**

本地开发：\`npm run dev\` 后访问 \`http://localhost:5173${route}\`

线上（GitHub Pages）：\`https://YOUR_USERNAME.github.io/deepseek${route}\`
`;
  fs.writeFileSync(path.join(root, filename), stub, 'utf8');
  console.log(`Stub: ${filename}`);
}

console.log('Migration complete.');
