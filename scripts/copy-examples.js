const fs = require('fs');
const path = require('path');

const root = path.resolve(__dirname, '..');
const copies = [
  ['examples', path.join(root, 'docs', 'public', 'examples')],
  ['data', path.join(root, 'docs', 'public', 'data')],
];

function copyDir(srcDir, destDir) {
  if (!fs.existsSync(srcDir)) return;
  fs.mkdirSync(destDir, { recursive: true });
  for (const entry of fs.readdirSync(srcDir, { withFileTypes: true })) {
    const s = path.join(srcDir, entry.name);
    const d = path.join(destDir, entry.name);
    if (entry.isDirectory()) {
      copyDir(s, d);
    } else {
      fs.copyFileSync(s, d);
    }
  }
}

for (const [rel, dest] of copies) {
  copyDir(path.join(root, rel), dest);
  console.log(`Copied ${rel}/ → docs/public/${rel}/`);
}
