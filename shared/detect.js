// The machine's eye: segmentation, the box rules, and drawing.
import { POLL, WHO } from "./air.js";

const VEG = new Set(["tree","grass","plant","palm","flower","field"]);

// colour of the human mask. [r,g,b] 0-255, mix 0-1
export const PERSON_TINT = [255, 78, 205];   // magenta
export const PERSON_MIX  = 0.6;

const BASE_BOXES = 130;   // boxes at exactly the WHO guideline
const MIN_BOXES  = 8;     // the machine never reports nothing
const MAX_TOTAL  = 1400;

export const MODELS = [
  { id:"Xenova/segformer-b0-finetuned-ade-512-512", name:"segformer-b0 · fast" },
  { id:"Xenova/segformer-b2-finetuned-ade-512-512", name:"segformer-b2 · slow" }
];

/* ---------- model ---------- */
let segmenter = null, segName = null;
export async function loadSegmenter(model, say = () => {}){
  if (segmenter && segName === model) return segmenter;
  say("loading segmentation model. first time only, 15-40 MB.");
  let mod;
  try { mod = await import("https://cdn.jsdelivr.net/npm/@huggingface/transformers@3"); }
  catch (e) { mod = await import("https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2"); }
  try { segmenter = await mod.pipeline("image-segmentation", model, navigator.gpu ? { device:"webgpu" } : {}); }
  catch (e) { segmenter = await mod.pipeline("image-segmentation", model); }
  segName = model;
  return segmenter;
}

/* ---------- segmentation -> masks ---------- */
export function emptyMasks(w, h){
  return { w, h, veg:new Uint8Array(w*h), person:new Uint8Array(w*h),
           vegPct:0, personPct:0, people:0, headY:-1 };
}

export async function segment(src, model, say){
  const pipe = await loadSegmenter(model, say);
  const out = await pipe(src.toDataURL("image/jpeg", 0.85));
  const w = src.width, h = src.height, m = emptyMasks(w, h);
  let vegPx = 0, personPx = 0;

  for (const seg of out){
    const isVeg = VEG.has(seg.label), isPerson = seg.label === "person";
    if (!isVeg && !isPerson) continue;
    const mk = seg.mask, mw = mk.width, mh = mk.height, md = mk.data, ch = mk.channels || 1;
    for (let y = 0; y < h; y++){
      const sy = Math.min(mh-1, (y*mh/h) | 0);
      for (let x = 0; x < w; x++){
        const sx = Math.min(mw-1, (x*mw/w) | 0);
        if (md[(sy*mw + sx) * ch] > 127){
          const i = y*w + x;
          if (isVeg && !m.veg[i]) { m.veg[i] = 1; vegPx++; }
          if (isPerson && !m.person[i]) { m.person[i] = 1; personPx++; }
        }
      }
    }
  }
  m.vegPct = 100 * vegPx / (w*h);
  m.personPct = 100 * personPx / (w*h);
  m.people = countBlobs(m.person, w, h);
  m.headY = headLine(m.person, w, h);
  return m;
}

function countBlobs(mask, w, h){
  const sw = Math.max(1, w>>2), sh = Math.max(1, h>>2);
  const small = new Uint8Array(sw*sh);
  for (let y=0;y<sh;y++) for (let x=0;x<sw;x++) small[y*sw+x] = mask[(y*4)*w + x*4];
  const seen = new Uint8Array(sw*sh), stack = [], minArea = 0.004*sw*sh;
  let count = 0;
  for (let i=0;i<small.length;i++){
    if (!small[i] || seen[i]) continue;
    let area = 0; stack.length = 0; stack.push(i); seen[i] = 1;
    while (stack.length){
      const p = stack.pop(); area++;
      const x = p % sw, y = (p/sw) | 0;
      if (x>0    && small[p-1]  && !seen[p-1])  { seen[p-1]=1;  stack.push(p-1); }
      if (x<sw-1 && small[p+1]  && !seen[p+1])  { seen[p+1]=1;  stack.push(p+1); }
      if (y>0    && small[p-sw] && !seen[p-sw]) { seen[p-sw]=1; stack.push(p-sw); }
      if (y<sh-1 && small[p+sw] && !seen[p+sw]) { seen[p+sw]=1; stack.push(p+sw); }
    }
    if (area > minArea) count++;
  }
  return count;
}

function headLine(mask, w, h){
  for (let y=0;y<h;y++) for (let x=0;x<w;x++) if (mask[y*w+x]) return y;
  return -1;
}

/* ---------- the box rules ---------- */
export function boxCounts(air){
  const counts = {}; let total = 0;
  for (const k of Object.keys(WHO)){
    counts[k] = Math.max(MIN_BOXES, Math.round(MIN_BOXES + BASE_BOXES * (air[k]||0) / WHO[k]));
    total += counts[k];
  }
  if (total > MAX_TOTAL){
    const f = MAX_TOTAL / total; total = 0;
    for (const k of Object.keys(counts)){ counts[k] = Math.max(MIN_BOXES, Math.round(counts[k]*f)); total += counts[k]; }
  }
  return { counts, total };
}

// boxes are spread evenly. trees and bodies decide, frame by frame, which ones are drawn.
export function makeBoxes(w, h, air){
  const { counts } = boxCounts(air), boxes = [];
  for (const key of Object.keys(counts)){
    const r = (air[key]||0) / WHO[key];
    const baseConf = 0.99 - 0.35 * Math.min(1, r);     // the less there is, the surer it gets
    for (let i = 0; i < counts[key]; i++){
      boxes.push({
        key,
        x: Math.random()*w, y: Math.random()*h,
        base: POLL[key].size * (0.7 + Math.random()*0.7),
        conf: Math.min(0.99, Math.max(0.40, baseConf + (Math.random()-0.5)*0.08)),
        r: Math.random(),                // fixed per box, so absorption doesn't flicker
        label: Math.random() < 0.25,
        dx: (Math.random()-0.5)*0.35, dy: -0.15 - Math.random()*0.35
      });
    }
  }
  return boxes;
}

export function stepBoxes(boxes, w, h, speed = 1){
  for (const b of boxes){
    b.x += b.dx*speed; b.y += b.dy*speed;
    if (b.y < -20){ b.y = h + 10; b.x = Math.random()*w; }
    if (b.x < -20) b.x = w + 10;
    if (b.x > w+20) b.x = -10;
  }
}

// a body takes particles out of the air it stands in: strongest at the head, weaker around it
const RING = [[1,0],[-1,0],[0,1],[0,-1],[.7,.7],[-.7,.7],[.7,-.7],[-.7,-.7]];
function absorbProb(m, x, y){
  if (!m.personPct) return 0;
  const { w, h, person, headY } = m, band = h*0.18;
  if (x<0 || y<0 || x>=w || y>=h) return 0;
  if (person[y*w+x]) return (headY >= 0 && y < headY + band) ? 0.97 : 0.85;
  const r = Math.max(2, Math.round(w*0.03));
  for (const [ox, oy] of RING){
    const sx = (x + ox*r) | 0, sy = (y + oy*r) | 0;
    if (sx>=0 && sy>=0 && sx<w && sy<h && person[sy*w+sx])
      return (headY >= 0 && sy < headY + band) ? 0.6 : 0.4;
  }
  return 0;
}

/* ---------- drawing ---------- */
export function drawFrame(canvas, frame, m, boxes, air, { stamp = true } = {}){
  const { w, h } = m;
  if (canvas.width !== w) canvas.width = w;
  if (canvas.height !== h) canvas.height = h;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(frame, 0, 0, w, h);

  if (m.vegPct > 0 || m.personPct > 0){
    const img = ctx.getImageData(0,0,w,h), d = img.data, t = PERSON_TINT, mx = PERSON_MIX;
    for (let i = 0; i < m.veg.length; i++){
      const p = i*4;
      if (m.veg[i]){
        d[p] = d[p]*0.35; d[p+1] = d[p+1]*0.35 + 150; d[p+2] = d[p+2]*0.35 + 40;
      } else if (m.person[i]){
        d[p]   = d[p]*(1-mx)   + t[0]*mx;
        d[p+1] = d[p+1]*(1-mx) + t[1]*mx;
        d[p+2] = d[p+2]*(1-mx) + t[2]*mx;
      }
    }
    ctx.putImageData(img, 0, 0);
  }

  const fs = Math.max(8, Math.round(w/110));
  ctx.font = `${fs}px "JetBrains Mono", monospace`;
  ctx.lineWidth = Math.max(1, w/1400);

  let visible = 0;
  for (const b of boxes){
    const xi = b.x|0, yi = b.y|0;
    if (xi>=0 && yi>=0 && xi<w && yi<h && m.veg[yi*w+xi]) continue;   // trees: clean
    if (b.r < absorbProb(m, xi, yi)) continue;                           // bodies: breathed in
    visible++;
    const depth = 0.45 + 1.25 * Math.max(0, Math.min(1, b.y/h));
    const s = b.base * depth, c = POLL[b.key].color;
    ctx.strokeStyle = c;
    ctx.globalAlpha = 0.55 + 0.4*b.conf;
    wireBox(ctx, b.x, b.y, s, s*0.8, s*0.4);
    if (b.label && s > 16){
      ctx.globalAlpha = 0.9; ctx.fillStyle = c;
      ctx.fillText(`${POLL[b.key].label} ${b.conf.toFixed(2)}`, b.x, b.y - 3);
    }
  }
  ctx.globalAlpha = 1;
  if (stamp) drawStamp(ctx, m, air, boxes.length, visible, fs);
  return visible;
}

function wireBox(ctx, x, y, w, h, d){
  ctx.beginPath();
  ctx.rect(x, y, w, h);
  ctx.rect(x+d, y-d, w, h);
  ctx.moveTo(x, y);     ctx.lineTo(x+d, y-d);
  ctx.moveTo(x+w, y);   ctx.lineTo(x+w+d, y-d);
  ctx.moveTo(x, y+h);   ctx.lineTo(x+d, y+h-d);
  ctx.moveTo(x+w, y+h); ctx.lineTo(x+w+d, y+h-d);
  ctx.stroke();
}

function drawStamp(ctx, m, air, claimed, drawn, fs){
  const lines = [
    `vegetation ${m.vegPct.toFixed(1)}%`,
    `humans ${m.personPct.toFixed(1)}% (${m.people})`,
    `claimed ${claimed}  drawn ${drawn}`,
    `${air.city} · ${air.source} · ${air.time}`
  ];
  const pad = fs, lh = fs*1.5;
  const wmax = Math.max(...lines.map(l => ctx.measureText(l).width));
  ctx.fillStyle = "rgba(0,0,0,.62)";
  ctx.fillRect(pad*0.6, pad*0.6, wmax + pad*0.9, lines.length*lh + fs*0.6);
  ctx.fillStyle = "#3bff8f";
  lines.forEach((l,i) => ctx.fillText(l, pad, pad + lh*(i+1) - fs*0.4));
}

/* ---------- readout rows ---------- */
export function readoutRows(m, air, claimed, drawn){
  const row = (k,v) => `<div class="row"><span class="k">${k}</span><span class="v">${v}</span></div>`;
  const out = [
    row("vegetation", m.vegPct.toFixed(1) + " % of frame"),
    row("humans", m.personPct.toFixed(1) + " % (" + m.people + ")")
  ];
  for (const k of Object.keys(WHO))
    out.push(row(`<span class="swatch" style="background:${POLL[k].color}"></span>${POLL[k].name}`,
                 (air[k]||0).toFixed(0) + " µg/m³"));
  out.push(row("detections claimed", claimed.toLocaleString()));
  out.push(row("detections drawn", drawn.toLocaleString()));
  return out.join("");
}
