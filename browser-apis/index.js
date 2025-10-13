// index.js (ES module)

const els = {
  // Hero
  hero: document.getElementById('hero'),
  heroStartCam: document.getElementById('heroStartCam'),
  heroFileInput: document.getElementById('heroFileInput'),
  // Stage & media
  stage: document.getElementById('stage'),
  dropOverlay: document.getElementById('dropOverlay'),
  video: document.getElementById('video'),
  imageEl: document.getElementById('imageEl'),
  canvas: document.getElementById('canvas'),
  // Primary actions
  detectBtn: document.getElementById('detectBtn'),
  freezeBtn: document.getElementById('freezeBtn'),
  stopCam: document.getElementById('stopCam'),
  // Secondary actions
  clearBtn: document.getElementById('clearBtn'),
  saveBtn: document.getElementById('saveBtn'),
  copyJsonBtn: document.getElementById('copyJsonBtn'),
  // URL load
  urlInput: document.getElementById('urlInput'),
  loadUrlBtn: document.getElementById('loadUrlBtn'),
  // Detectors
  optBarcode: document.getElementById('optBarcode'),
  optFace: document.getElementById('optFace'),
  optText: document.getElementById('optText'),
  barcodeFormats: document.getElementById('barcodeFormats'),
  barcodeOpts: document.getElementById('barcodeOpts'),
  faceOpts: document.getElementById('faceOpts'),
  maxFaces: document.getElementById('maxFaces'),
  fastMode: document.getElementById('fastMode'),
  // Status & results
  statusList: document.getElementById('statusList'),
  resultsTable: document.querySelector('#resultsTable tbody'),
};

let stream = null;
let mode = null; // 'video' | 'image' | null
let lastDetections = [];

const ctx = els.canvas.getContext('2d', { alpha: true });

// ---------- Utils ----------
const sleep = (ms) => new Promise(r => setTimeout(r, ms));
const clamp = (n, min, max) => Math.max(min, Math.min(max, n));
const toBox = (r) => [Math.round(r.x), Math.round(r.y), Math.round(r.width), Math.round(r.height)];

// Force visible to override CSS display:none
function setHidden(el, hidden) {
  el.style.display = hidden ? 'none' : 'block';
}

function updateActionStates() {
  const hasSource = (mode === 'video' || mode === 'image');
  els.detectBtn.disabled = !hasSource;   // ALWAYS available with any source (live or still)
  els.clearBtn.disabled = !hasSource;
  els.saveBtn.disabled = !hasSource;
  els.copyJsonBtn.disabled = (lastDetections.length === 0);
  els.freezeBtn.disabled = (mode !== 'video');
  els.stopCam.disabled = (mode !== 'video');
}

function logStatus(html) {
  const li = document.createElement('li'); li.innerHTML = html;
  els.statusList.appendChild(li);
}
function clearStatus() { els.statusList.innerHTML = ''; }

function setStageSizeFrom(el) {
  const w = el.naturalWidth || el.videoWidth || el.width || 0;
  const h = el.naturalHeight || el.videoHeight || el.height || 0;
  if (!w || !h) return;
  els.canvas.width = w;
  els.canvas.height = h;
}
function clearOverlays() { ctx.clearRect(0, 0, els.canvas.width, els.canvas.height); }
function drawBox(rect, { label = '', color = '#7aa2ff' } = {}) {
  const [x, y, w, h] = toBox(rect);
  ctx.lineWidth = Math.max(2, Math.min(6, Math.floor(Math.max(w, h) / 100)));
  ctx.strokeStyle = color;
  ctx.strokeRect(x, y, w, h);
  if (label) {
    ctx.font = `${Math.max(10, Math.round(ctx.lineWidth * 5))}px ui-sans-serif`;
    ctx.fillStyle = color;
    const pad = 4, textW = ctx.measureText(label).width, labelH = ctx.lineWidth * 6 + 6;
    const yTop = Math.max(0, y - labelH - 4);
    ctx.fillRect(x, yTop, textW + pad * 2, labelH);
    ctx.fillStyle = '#0b0d12';
    ctx.fillText(label, x + pad, Math.max(12, yTop + labelH - 6));
  }
}
function row({ type, summary, box }) {
  const tr = document.createElement('tr');
  const a = document.createElement('td'); a.textContent = type;
  const b = document.createElement('td'); b.textContent = summary;
  const c = document.createElement('td'); c.textContent = box.join(', ');
  tr.append(a, b, c); return tr;
}

// ---------- Feature detection + startup alert + pre-check ----------
async function checkSupportAndAlert() {
  clearStatus();
  const issues = [];
  const support = { barcode:false, face:false, text:false };

  // Camera API
  const hasCam = !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
  logStatus(`Camera: ${hasCam ? '<span class="good">available</span>' : '<span class="bad">unavailable</span>'} (<code>getUserMedia</code>)`);
  if (!hasCam) issues.push('Camera (getUserMedia) unavailable');

  // Barcode
  if ('BarcodeDetector' in window) {
    try {
      const formats = await window.BarcodeDetector.getSupportedFormats();
      support.barcode = true;
      logStatus(`BarcodeDetector: <span class="good">present</span> — formats: <code>${formats.join(', ') || 'none'}</code>`);
      populateBarcodeFormats(formats || []);
      if (!formats || formats.length === 0) issues.push('Barcode formats not reported by platform');
    } catch (e) {
      // If getSupportedFormats fails, still assume detector exists
      support.barcode = true;
      logStatus(`BarcodeDetector: <span class="warn">present</span> but <code>getSupportedFormats()</code> failed (${e.name})`);
    }
  } else {
    logStatus('BarcodeDetector: <span class="bad">not present</span>');
    issues.push('BarcodeDetector not present');
  }

  // Face (defensive check)
  if ('FaceDetector' in window) {
    const tmp = document.createElement('canvas'); tmp.width = 2; tmp.height = 2;
    try {
      await new window.FaceDetector().detect(tmp);
      support.face = true;
      logStatus('FaceDetector: <span class="good">present & likely supported</span>');
    } catch (e) {
      if (e.name === 'NotSupportedError') {
        logStatus('FaceDetector: <span class="warn">present</span> but <span class="bad">not supported</span> (enable experimental features).');
        issues.push('FaceDetector present but not supported');
      } else {
        logStatus(`FaceDetector: <span class="warn">present</span> (test threw ${e.name})`);
      }
    }
  } else {
    logStatus('FaceDetector: <span class="bad">not present</span>');
    issues.push('FaceDetector not present');
  }

  // Text (defensive check)
  if ('TextDetector' in window) {
    const tmp = document.createElement('canvas'); tmp.width = 2; tmp.height = 2;
    try {
      await new window.TextDetector().detect(tmp);
      support.text = true;
      logStatus('TextDetector: <span class="good">present & likely supported</span>');
    } catch (e) {
      if (e.name === 'NotSupportedError') {
        logStatus('TextDetector: <span class="warn">present</span> but <span class="bad">not supported</span> (experimental).');
        issues.push('TextDetector present but not supported');
      } else {
        logStatus(`TextDetector: <span class="warn">present</span> (test threw ${e.name})`);
      }
    }
  } else {
    logStatus('TextDetector: <span class="bad">not present</span>');
    issues.push('TextDetector not present');
  }

  // Pre-check what’s supported; disable what’s not
  els.optBarcode.checked = support.barcode;
  els.optBarcode.disabled = !support.barcode;
  els.optFace.checked = support.face;
  els.optFace.disabled = !support.face;
  els.optText.checked = support.text;
  els.optText.disabled = !support.text;

  setFaceOptsVisibility();
  setBarcodeOptsVisibility();

  if (issues.length) {
    alert(
`Heads up: some Shape Detection features may not be supported on this browser/OS.

Missing or limited:
- ${issues.join('\n- ')}

Tips:
• Try enabling chrome://flags/#enable-experimental-web-platform-features then restart the browser.
• Barcode is usually available by default; Face/Text vary by platform.`
    );
  }
}

function populateBarcodeFormats(formats) {
  els.barcodeFormats.innerHTML = '';
  const desired = ['qr_code','aztec','pdf417','data_matrix','ean_13','ean_8','upc_a','upc_e','code_128','code_39','code_93','itf','codabar'];
  const ordered = formats.slice().sort((a,b)=> desired.indexOf(a)-desired.indexOf(b));
  (ordered.length ? ordered : desired).forEach(fmt => {
    const opt = document.createElement('option');
    opt.value = fmt; opt.textContent = fmt; opt.selected = true;
    els.barcodeFormats.appendChild(opt);
  });
}

function setFaceOptsVisibility(){ els.faceOpts.style.display = els.optFace.checked ? 'block' : 'none'; }
function setBarcodeOptsVisibility(){ els.barcodeOpts.style.display = els.optBarcode.checked ? 'block' : 'none'; }

// ---------- Source switching ----------
function showVideo() {
  setHidden(els.imageEl, true);
  setHidden(els.video, false);    // force block
  mode = 'video';
  updateActionStates();
}
function showImage() {
  setHidden(els.video, true);
  setHidden(els.imageEl, false);  // force block
  mode = 'image';
  updateActionStates();
}
function hideHero() { setHidden(els.hero, true); }

// ---------- Loading inputs ----------
async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    alert('Camera API not available in this browser.');
    return;
  }
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: { ideal: 'environment' } }, audio: false });
    els.video.srcObject = stream;

    // Wait for metadata to get videoWidth/Height, then play.
    await new Promise(res => {
      if (els.video.readyState >= 1) res();
      else els.video.addEventListener('loadedmetadata', res, { once:true });
    });
    await els.video.play();

    setStageSizeFrom(els.video);
    clearOverlays();
    showVideo();
    hideHero();
  } catch (e) {
    alert(`Camera error: ${e.name}. ${e.message || ''}`);
  }
}
function stopCamera() {
  if (stream) { stream.getTracks().forEach(t=>t.stop()); stream = null; }
  mode = null;
  updateActionStates();
}

async function loadFile(file) {
  if (!file) return;
  const url = URL.createObjectURL(file);
  try {
    await loadImageURL(url, true);
    hideHero();
  } catch (e) {
    alert(e.message);
  }
}
function loadImageURL(url, isObjectURL=false) {
  return new Promise((resolve, reject) => {
    const img = els.imageEl;
    img.onload = () => {
      setStageSizeFrom(img);
      clearOverlays();
      showImage();
      if (isObjectURL) setTimeout(()=>URL.revokeObjectURL(url), 0);
      resolve();
    };
    img.onerror = () => reject(new Error('Failed to load image (CORS or network).'));
    img.crossOrigin = 'anonymous';
    img.src = url;
  });
}

// Drag & drop to stage
['dragenter','dragover'].forEach(evt => {
  els.stage.addEventListener(evt, e => { e.preventDefault(); els.dropOverlay.style.display='grid'; });
});
['dragleave','drop'].forEach(evt => {
  els.stage.addEventListener(evt, e => { e.preventDefault(); els.dropOverlay.style.display='none'; });
});
els.stage.addEventListener('drop', async (e) => {
  const file = [...e.dataTransfer.files].find(f => f.type.startsWith('image/'));
  if (file) await loadFile(file);
});

// Paste image
window.addEventListener('paste', async (e) => {
  const items = e.clipboardData?.items || [];
  for (const it of items) {
    if (it.type.startsWith('image/')) {
      await loadFile(it.getAsFile());
      return;
    }
  }
});

// Freeze current frame into an <img> snapshot
function freezeCurrentFrame() {
  if (mode !== 'video') return;
  const off = document.createElement('canvas');
  off.width = els.canvas.width;
  off.height = els.canvas.height;
  const octx = off.getContext('2d');
  try {
    octx.drawImage(els.video, 0, 0, off.width, off.height);
    els.imageEl.src = off.toDataURL('image/png');
    showImage();
    logStatus('Snapshot captured from video and switched to still image.');
  } catch (e) {
    alert('Could not capture frame from camera.');
  }
}

// Download composed PNG (base + overlays)
function downloadPNG() {
  const w = els.canvas.width || 640;
  const h = els.canvas.height || 400;
  const out = document.createElement('canvas');
  out.width = w; out.height = h;
  const octx = out.getContext('2d');

  try {
    if (mode === 'video') octx.drawImage(els.video, 0, 0, w, h);
    if (mode === 'image') octx.drawImage(els.imageEl, 0, 0, w, h);
  } catch {}
  octx.drawImage(els.canvas, 0, 0);

  const url = out.toDataURL('image/png');
  const a = document.createElement('a');
  a.href = url; a.download = 'shape-detection.png'; a.click();
}

// ---------- Detection ----------
async function detectAll() {
  if (!mode) { alert('Choose an input source first (camera or image).'); return; }
  clearOverlays();
  els.resultsTable.innerHTML = '';
  lastDetections = [];

  // Build detectors as requested
  const detectors = {};
  if (els.optBarcode.checked && 'BarcodeDetector' in window) {
    const selected = [...els.barcodeFormats.options].filter(o=>o.selected).map(o=>o.value);
    detectors.barcode = new window.BarcodeDetector(selected.length ? { formats: selected } : undefined);
  }
  if (els.optFace.checked && 'FaceDetector' in window) {
    const maxDetectedFaces = clamp(parseInt(els.maxFaces.value || '5', 10), 1, 50);
    const fastMode = !!els.fastMode.checked;
    detectors.face = new window.FaceDetector({ maxDetectedFaces, fastMode });
  }
  if (els.optText.checked && 'TextDetector' in window) {
    detectors.text = new window.TextDetector();
  }

  // Use ImageBitmap to standardize input across detectors
  let inputBitmap = null;
  try {
    inputBitmap = (mode === 'video') ? await createImageBitmap(els.video) : await createImageBitmap(els.imageEl);
  } catch {
    inputBitmap = (mode === 'video') ? els.video : els.imageEl; // fallback
  }

  await Promise.all([
    runBarcode(detectors.barcode, inputBitmap),
    runFace(detectors.face, inputBitmap),
    runText(detectors.text, inputBitmap),
  ]);

  console.log('Detections:', lastDetections);
  els.copyJsonBtn.disabled = lastDetections.length === 0;
}

async function runBarcode(detector, src) {
  if (!detector) return;
  try {
    const results = await detector.detect(src);
    results.forEach(bc => {
      const label = `barcode: ${bc.format}${bc.rawValue ? ` → ${bc.rawValue}` : ''}`;
      drawBox(bc.boundingBox, { label, color: '#7aa2ff' });
      els.resultsTable.appendChild(row({ type:'barcode', summary:`${bc.format}${bc.rawValue ? `: ${bc.rawValue}` : ''}`, box: toBox(bc.boundingBox) }));
      lastDetections.push({ type:'barcode', raw: bc });
    });
  } catch (e) { handleDetectError('BarcodeDetector', e); }
}
async function runFace(detector, src) {
  if (!detector) return;
  try {
    const results = await detector.detect(src);
    results.forEach(face => {
      drawBox(face.boundingBox, { label:'face', color:'#32d296' });
      if (Array.isArray(face.landmarks)) {
        ctx.fillStyle = '#32d296';
        face.landmarks.forEach(lm => {
          if (lm.locations) lm.locations.forEach(pt => { ctx.beginPath(); ctx.arc(pt.x, pt.y, 3, 0, Math.PI*2); ctx.fill(); });
          else if (lm.x !== undefined && lm.y !== undefined) { ctx.beginPath(); ctx.arc(lm.x, lm.y, 3, 0, Math.PI*2); ctx.fill(); }
        });
      }
      els.resultsTable.appendChild(row({ type:'face', summary: face.landmarks ? `landmarks: ${face.landmarks.length}` : 'landmarks: n/a', box: toBox(face.boundingBox) }));
      lastDetections.push({ type:'face', raw: face });
    });
  } catch (e) { handleDetectError('FaceDetector', e); }
}
async function runText(detector, src) {
  if (!detector) return;
  try {
    const results = await detector.detect(src);
    results.forEach(t => {
      const text = t.rawValue || t.data || '(text)';
      drawBox(t.boundingBox, { label:`text: ${truncate(text, 24)}`, color:'#ffb86b' });
      els.resultsTable.appendChild(row({ type:'text', summary:text, box: toBox(t.boundingBox) }));
      lastDetections.push({ type:'text', raw: t });
    });
  } catch (e) { handleDetectError('TextDetector', e); }
}

function truncate(s, n){ if(!s) return ''; return s.length>n ? s.slice(0,n-1)+'…' : s; }
function handleDetectError(kind, e){
  const map = {
    NotAllowedError: 'Permission denied.',
    NotReadableError: 'Device is busy or not readable.',
    NotFoundError: 'No device or source found.',
    SecurityError: 'Cross-origin or secure-context requirement failed.',
    NotSupportedError: 'Detector not supported on this platform/build.',
  };
  alert(`${kind} failed: ${e.name}. ${map[e.name] || e.message || String(e)}`);
}

// ---------- Events ----------
els.heroStartCam.addEventListener('click', startCamera);
els.heroFileInput.addEventListener('change', async (e)=>{ const f=e.currentTarget.files?.[0]; if(f) await loadFile(f); });

els.loadUrlBtn.addEventListener('click', async ()=>{
  const url = els.urlInput.value.trim();
  if (!url) return;
  try { await loadImageURL(url); hideHero(); } catch(e){ alert(e.message); }
});

els.stopCam.addEventListener('click', stopCamera);
els.freezeBtn.addEventListener('click', freezeCurrentFrame);
els.detectBtn.addEventListener('click', detectAll);
els.clearBtn.addEventListener('click', ()=>{ clearOverlays(); els.resultsTable.innerHTML=''; lastDetections=[]; updateActionStates(); });
els.saveBtn.addEventListener('click', downloadPNG);

els.optFace.addEventListener('change', setFaceOptsVisibility);
els.optBarcode.addEventListener('change', setBarcodeOptsVisibility);

// ---------- Init ----------
window.addEventListener('load', async () => {
  if (!els.canvas.width || !els.canvas.height) { els.canvas.width = 640; els.canvas.height = 360; }
  await checkSupportAndAlert();
  setFaceOptsVisibility();
  setBarcodeOptsVisibility();

  // Pre-run state: disable actions until there’s a source
  mode = null;
  updateActionStates();

  // ?img= support
  const params = new URLSearchParams(location.search);
  const img = params.get('img');
  if (img) { try { await loadImageURL(img); hideHero(); } catch(e){ console.warn('Query image failed:', e); } }
});
