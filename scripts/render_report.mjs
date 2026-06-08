// Render a Markdown doc (Markdown + LaTeX math) under docs/ to a sibling PDF.
//
// pandoc/LaTeX are not assumed to be installed; this uses only Node packages:
//   markdown-it (Markdown)  +  markdown-it-texmath/KaTeX (math)  +  puppeteer (PDF).
// KaTeX CSS is inlined and images are resolved relative to the repo root, so the
// comparison chart and confusion matrices embed correctly.
//
// Usage:  npm install  &&  node scripts/render_report.mjs [docs/FILE.md]
//   Defaults to docs/REPORT.md. The output PDF takes the same basename, e.g.
//   `node scripts/render_report.mjs docs/SOURCE_CODE.md` -> docs/SOURCE_CODE.pdf.

import { readFileSync, existsSync } from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { dirname, resolve } from 'node:path';
import MarkdownIt from 'markdown-it';
import texmath from 'markdown-it-texmath';
import katex from 'katex';
import puppeteer from 'puppeteer';

const __dirname = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(__dirname, '..');

// Optional CLI arg: a Markdown file (relative to repo root or absolute).
// Defaults to docs/REPORT.md; the PDF is written next to it with a .pdf extension.
const inArg = process.argv[2] ?? 'docs/REPORT.md';
const inPath = resolve(repoRoot, inArg);
const outPath = inPath.replace(/\.md$/i, '.pdf');

// --- Markdown -> HTML, with KaTeX for $...$ and $$...$$ ----------------------
const md = new MarkdownIt({ html: true, linkify: true, typographer: true }).use(
  texmath,
  {
    engine: katex,
    delimiters: 'dollars',
    katexOptions: { throwOnError: false, strict: false },
  },
);

const markdown = readFileSync(inPath, 'utf8');

// Inline local images as base64 data URIs. A document loaded via page.setContent has
// an about:blank origin, from which Chrome blocks file:// sub-resources — so relative
// <img src> would render as broken icons. Data URIs sidestep that entirely.
const MIME = { png: 'image/png', jpg: 'image/jpeg', jpeg: 'image/jpeg', gif: 'image/gif', svg: 'image/svg+xml' };
function inlineImages(html) {
  return html.replace(/<img\b([^>]*?)\bsrc="([^"]+)"([^>]*)>/g, (m, pre, src, post) => {
    if (/^(data:|https?:)/i.test(src)) return m;
    const abs = resolve(repoRoot, 'docs', src); // src is relative to docs/REPORT.md
    if (!existsSync(abs)) {
      console.warn(`[warn] image not found, left as-is: ${src}`);
      return m;
    }
    const ext = abs.split('.').pop().toLowerCase();
    const b64 = readFileSync(abs).toString('base64');
    return `<img${pre}src="data:${MIME[ext] ?? 'application/octet-stream'};base64,${b64}"${post}>`;
  });
}

const bodyHtml = inlineImages(md.render(markdown));

// Inline KaTeX stylesheet so the PDF needs no network at print time.
const katexCss = readFileSync(
  resolve(repoRoot, 'node_modules', 'katex', 'dist', 'katex.min.css'),
  'utf8',
);

// Base URL = docs/ (where REPORT.md lives) so relative image src like
// ../artifacts/comparison.png resolve to repoRoot/artifacts/comparison.png.
const baseHref = pathToFileURL(resolve(repoRoot, 'docs') + '/').href;

const html = `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<base href="${baseHref}">
<style>${katexCss}</style>
<style>
  body { font-family: "Segoe UI", Helvetica, Arial, sans-serif; font-size: 11pt;
         line-height: 1.5; color: #1a1a1a; max-width: 820px; margin: 0 auto; }
  h1, h2, h3 { line-height: 1.25; }
  h1 { font-size: 20pt; } h2 { font-size: 15pt; border-bottom: 1px solid #ddd;
       padding-bottom: 0.2em; margin-top: 1.4em; } h3 { font-size: 12.5pt; }
  table { border-collapse: collapse; margin: 1em 0; font-size: 10pt; }
  th, td { border: 1px solid #bbb; padding: 4px 9px; text-align: left; }
  th { background: #f2f2f2; }
  code { background: #f4f4f4; padding: 0.1em 0.3em; border-radius: 3px;
         font-family: Consolas, "Courier New", monospace; font-size: 9.5pt; }
  pre { background: #f6f8fa; padding: 0.8em; border-radius: 5px; overflow-x: auto; }
  pre code { background: none; padding: 0; }
  img { max-width: 100%; }
  blockquote { border-left: 3px solid #ccc; margin-left: 0; padding-left: 1em;
               color: #444; }
  hr { border: none; border-top: 1px solid #ddd; margin: 1.6em 0; }
  .katex-display { overflow-x: auto; overflow-y: hidden; padding: 0.2em 0; }
</style>
</head>
<body>
${bodyHtml}
</body>
</html>`;

if (!existsSync(resolve(repoRoot, 'artifacts', 'comparison.png'))) {
  console.warn('[warn] artifacts/comparison.png not found — figures may be blank.');
}

const browser = await puppeteer.launch({ headless: 'new' });
try {
  const page = await browser.newPage();
  await page.setContent(html, { waitUntil: 'networkidle0' });
  await page.pdf({
    path: outPath,
    format: 'A4',
    printBackground: true,
    margin: { top: '18mm', bottom: '18mm', left: '16mm', right: '16mm' },
  });
  console.log(`[ok] wrote ${outPath}`);
} finally {
  await browser.close();
}
