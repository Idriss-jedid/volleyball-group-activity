import { chromium } from 'playwright-core';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';

// Records deck.html to out/deck.webm, and when every slide and every sentence starts to out/marks.json.
//   node record.mjs                  record (each sentence lasts its narration, from timing.js)
//   node record.mjs --stills <dir>   one screenshot per slide, fully revealed, to check the layout
process.chdir(path.dirname(fileURLToPath(import.meta.url)));
const W = 1920, H = 1080;
const stills = process.argv[2] === '--stills' ? path.resolve(process.argv[3]) : null;
const rec = path.join('out', 'rec');
if (!stills) fs.rmSync(rec, { recursive: true, force: true });
const browser = await chromium.launch({ channel: 'msedge', headless: true });
const ctx = await browser.newContext({ viewport: { width: W, height: H }, deviceScaleFactor: 1,
  ...(stills ? {} : { recordVideo: { dir: rec, size: { width: W, height: H } } }) });
const t0 = Date.now();
const now = () => +((Date.now() - t0) / 1000).toFixed(3);
const page = await ctx.newPage();
await page.goto('file:///' + path.resolve('deck.html').replace(/\\/g, '/'), { waitUntil: 'load' });
await page.evaluate(() => Promise.all([...document.images].map(i => i.decode().catch(() => {}))));
await page.evaluate(() => document.fonts.ready);
await page.waitForTimeout(1500);
const lead = now();
const slides = await page.evaluate(() => window.__slides());
const marks = [];
let clock = Date.now();
const until = async ms => { clock += ms; await page.waitForTimeout(Math.max(0, clock - Date.now())); };  // no drift
for (let i = 0; i < slides.length; i++) {
  const m = { id: slides[i].id, start: now(), lines: [], videos: slides[i].videos };
  await page.evaluate(k => window.__go(k), i);
  if (stills) {
    await page.evaluate(k => window.__step(k, 99), i);
    await page.waitForTimeout(2200);
    fs.mkdirSync(stills, { recursive: true });
    await page.screenshot({ path: path.join(stills, `${String(i).padStart(2, '0')}_${slides[i].id}.jpg`), type: 'jpeg', quality: 80 });
    continue;
  }
  for (let k = 0; k < slides[i].lines.length; k++) {
    if (k) await page.evaluate(([a, b]) => window.__step(a, b), [i, k]);
    m.lines.push(now());
    await until(slides[i].lines[k]);
  }
  marks.push(m);
}
const end = now();
await page.waitForTimeout(600);
await ctx.close();
await browser.close();
if (!stills) {
  const f = fs.readdirSync(rec).find(x => x.endsWith('.webm'));
  fs.renameSync(path.join(rec, f), path.join('out', 'deck.webm'));
  fs.writeFileSync(path.join('out', 'marks.json'), JSON.stringify({ lead, end, slides: marks }));
  console.log(`recorded out/deck.webm (lead ${lead.toFixed(2)} s, ${slides.length} slides, ${(end - lead).toFixed(1)} s)`);
} else console.log(`stills -> ${stills}`);
