/* ===================================================================
   Clothic AI - frontend for the Clothic AI explainable compliance API
   Vanilla JS. No build step. Talks to the local backend via fetch().
   =================================================================== */

// ---- Configuration -------------------------------------------------
// Absolute backend origin so the frontend works wherever it is hosted
// (clothic.site on Hostinger is frontend-only and must call the HF Space
// backend cross-origin; the API allows it via CLOTHIC_CORS_ORIGINS=*). On the
// HF Space itself this still resolves to the same host, so one file fits both.
// Set to "" for pure same-origin local dev (uvicorn serves SPA + API together).
const API_BASE = "https://cipaimian-clothic.hf.space";
const REALTIME_INTERVAL_MS = 3000;            // live scan cadence
const CAPTURE_QUALITY = 0.92;                 // JPEG quality for webcam frames

// ---- Tiny DOM helpers ----------------------------------------------
const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const clamp01 = (n) => Math.max(0, Math.min(1, Number(n) || 0));
const pct = (n) => `${Math.round(clamp01(n) * 100)}%`;
const esc = (s) =>
  String(s ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const titleCase = (s) =>
  String(s ?? "").replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());

// The backend explanation embeds a "Coverage - …" clause, but the UI renders
// coverage in its own panel below. Strip the duplicate so the prose reads clean.
// (Lazy match stops at the first period followed by whitespace/end - decimals
// like "0.48" are safe because their period is followed by a digit.)
const cleanExplanation = (text) =>
  String(text ?? "").replace(/\s*Coverage -[\s\S]*?\.(?=\s|$)/, "").trim() ||
  "No explanation provided.";

// ===================================================================
//  Toast notifications
// ===================================================================
const Toast = (() => {
  const stack = $("#toastStack");
  const icons = { success: "✓", error: "×", info: "i", warn: "!" };

  function show({ type = "info", title = "", msg = "", duration = 4000, progress = false, id = null }) {
    if (id) dismiss(id); // replace an existing toast with the same id
    const el = document.createElement("div");
    el.className = `toast ${type}`;
    if (id) el.dataset.id = id;
    el.innerHTML = `
      <span class="t-ic">${icons[type] || "i"}</span>
      <div class="t-body">
        ${title ? `<div class="t-title">${esc(title)}</div>` : ""}
        ${msg ? `<div class="t-msg">${esc(msg)}</div>` : ""}
        ${progress ? `<div class="toast-progress"><span></span></div>` : ""}
      </div>`;
    stack.appendChild(el);
    if (duration > 0) setTimeout(() => remove(el), duration);
    return el;
  }
  function remove(el) {
    if (!el || !el.parentNode) return;
    el.classList.add("out");
    setTimeout(() => el.remove(), 300);
  }
  function dismiss(id) {
    const el = stack.querySelector(`.toast[data-id="${id}"]`);
    if (el) remove(el);
  }
  return { show, remove, dismiss };
})();

// ===================================================================
//  Backend connectivity
// ===================================================================
const apiDot = $("#apiDot");
const apiStatusText = $("#apiStatusText");

function setApiState(state, text) {
  apiDot.dataset.state = state;
  apiStatusText.textContent = text;
}

async function checkHealth() {
  // Backend status lives only in the top-right navbar pill - no popups.
  setApiState("connecting", "Loading model…");
  try {
    const res = await fetch(`${API_BASE}/v1/health`, { method: "GET" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    setApiState("online", "Model connected");
    return true;
  } catch (err) {
    setApiState("offline", "Model unavailable");
    return false;
  }
}

async function loadProfiles() {
  const select = $("#profileSelect");
  try {
    const res = await fetch(`${API_BASE}/v1/profiles`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    const profiles = data.profiles || [];
    select.innerHTML = "";
    profiles.forEach((id) => {
      const opt = document.createElement("option");
      opt.value = id;
      opt.textContent = titleCase(id);
      select.appendChild(opt);
    });
    if (!profiles.length) {
      select.innerHTML = `<option value="default">default</option>`;
    }
  } catch (err) {
    select.innerHTML = `<option value="default">default</option>`;
  }
}

// ===================================================================
//  Core inference call - POST /v1/infer_image
// ===================================================================
async function inferImage(blob, { silent = false } = {}) {
  const profileId = $("#profileSelect").value || "default";
  const form = new FormData();
  form.append("file", blob, "frame.jpg");

  const url = `${API_BASE}/v1/infer_image?profile_id=${encodeURIComponent(profileId)}&camera_id=web0`;
  const res = await fetch(url, { method: "POST", body: form });
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try { detail = (await res.json()).detail || detail; } catch { /* ignore */ }
    const err = new Error(detail);
    err.handled = silent;
    throw err;
  }
  return res.json();
}

// ===================================================================
//  Results rendering
// ===================================================================
const resultsEmpty = $("#resultsEmpty");
const resultsLoading = $("#resultsLoading");
const resultsContent = $("#resultsContent");

function showResultsState(state) {
  resultsEmpty.hidden = state !== "empty";
  resultsLoading.hidden = state !== "loading";
  resultsContent.hidden = state !== "content";
}

const DECISION_LABEL = {
  compliant: "Compliant",
  minor_violation: "Minor Violation",
  major_violation: "Major Violation",
  insufficient_evidence: "Insufficient Evidence",
};

function renderFrame(frame) {
  const persons = frame.persons || [];
  $("#resultsMeta").innerHTML = `
    <span>Frame <code>${esc(frame.frame_id)}</code></span>
    <span>Profile <code>${esc(frame.profile_id)}</code> · v${esc(frame.policy_version)}</span>`;

  const container = $("#personsContainer");
  if (!persons.length) {
    container.innerHTML = `
      <div class="insufficient-note">
        <div><strong>No person detected.</strong> Make sure a person is clearly visible in the frame and try again.</div>
      </div>`;
  } else {
    container.innerHTML = persons.map(renderPerson).join("");
  }

  const L = frame.latency_ms || {};
  $("#latencyBar").innerHTML = `
    <span>Perception <b>${(L.perception ?? 0).toFixed(1)} ms</b></span>
    <span>Reasoning <b>${(L.reasoning ?? 0).toFixed(1)} ms</b></span>
    <span>Total <b>${(L.total ?? 0).toFixed(1)} ms</b></span>`;

  showResultsState("content");
  // Trigger the animated progress bars after paint.
  requestAnimationFrame(() => {
    $$(".score-fill, .exp-fill").forEach((el) => { el.style.width = el.dataset.width; });
  });
}

function renderPerson(p, idx) {
  const d = p.decision;
  const s = p.scores || {};
  const obs = p.observation || {};
  const isInsufficient = d === "insufficient_evidence";

  return `
  <div class="person-card" style="animation-delay:${idx * 60}ms">
    <div class="person-head">
      <span class="person-id">Person · track #${esc(p.track_id)}</span>
      <span class="decision-badge d-${esc(d)}"><span class="dot"></span>${esc(DECISION_LABEL[d] || titleCase(d))}</span>
    </div>

    ${isInsufficient ? `
      <div class="insufficient-note">
        <div><strong>The AI needs a clearer view.</strong> Evidence quality was too low to judge confidently (occlusion, blur, or low resolution), so it abstained rather than risk a false call.</div>
      </div>` : ""}

    <p class="block-title">Explanation</p>
    <div class="explanation">${esc(cleanExplanation(p.explanation))}</div>

    <p class="block-title">Four Scores</p>
    ${renderScores(s)}

    <p class="block-title">Detected Garments</p>
    <div class="garments">
      ${renderGarment("Upper", obs.upper)}
      ${renderGarment("Lower", obs.lower)}
      ${renderGarment("Footwear", obs.footwear)}
    </div>

    <p class="block-title">Body Exposure</p>
    ${renderExposure(obs.exposure)}

    <p class="block-title">Matched Rules</p>
    ${renderRules(p.matched_rules)}

    ${(p.coverage && p.coverage.length) ? `
      <p class="block-title">Coverage</p>
      ${p.coverage.map(renderCoverage).join("")}` : ""}

    ${renderRemediation(p.remediation)}
  </div>`;
}

function scoreRow(name, value, fill, isOverall = false) {
  if (value === null || value === undefined) {
    return `
    <div class="score ${isOverall ? "overall-row" : ""}">
      <div class="score-top"><span class="score-name">${esc(name)}</span><span class="score-val" style="color:var(--gray)">N/A</span></div>
      <div class="score-track"></div>
    </div>`;
  }
  const v = clamp01(value);
  return `
    <div class="score ${isOverall ? "overall-row" : ""}">
      <div class="score-top"><span class="score-name">${esc(name)}</span><span class="score-val">${pct(v)}</span></div>
      <div class="score-track"><span class="score-fill ${fill}" data-width="${pct(v)}"></span></div>
    </div>`;
}

function renderScores(s) {
  return `
  <div class="scores">
    ${scoreRow("Exposure", s.exposure_score, "fill-amber")}
    ${scoreRow("Formality", s.formality_score, "fill-cyan")}
    ${scoreRow("Compliance", s.compliance_score, "fill-green")}
    ${scoreRow("Uncertainty", s.uncertainty_score, "fill-purple")}
    ${scoreRow("Overall Violation", s.overall_violation, "fill-purple", true)}
  </div>`;
}

function renderGarment(slot, g) {
  if (!g || !g.type) {
    return `<div class="garment"><span class="garment-slot">${esc(slot)}</span><span class="garment-empty">not detected</span></div>`;
  }
  const attrs = Object.entries(g.attributes || {})
    .filter(([, v]) => v >= 0.5)
    .sort((a, b) => b[1] - a[1])
    .map(([k, v]) => `<span class="attr-tag">${esc(titleCase(k))} ${pct(v)}</span>`)
    .join("");
  return `
    <div class="garment">
      <div style="flex:1">
        <div style="display:flex;align-items:center;gap:.8rem">
          <span class="garment-slot">${esc(slot)}</span>
          <span class="garment-type">${esc(titleCase(g.type))}</span>
          <span class="garment-conf">${pct(g.conf)}</span>
        </div>
        ${attrs ? `<div class="garment-attrs">${attrs}</div>` : ""}
      </div>
    </div>`;
}

function renderExposure(exposure) {
  const entries = Object.entries(exposure || {});
  if (!entries.length) return `<div class="garment-empty">No exposure data.</div>`;
  return `<div class="exposure-grid">${entries.map(([region, val]) => {
    const v = clamp01(val);
    const over = v > 0.4; // visual emphasis only; real limits live in the profile
    const fill = over ? "fill-amber" : "fill-cyan";
    return `
      <div class="exp-row">
        <span class="exp-name">${esc(region)}</span>
        <span class="exp-track"><span class="exp-fill ${fill}" data-width="${pct(v)}"></span></span>
        <span class="exp-val ${over ? "exp-over" : ""}">${pct(v)}</span>
      </div>`;
  }).join("")}</div>`;
}

function renderRules(rules) {
  if (!rules || !rules.length) {
    return `<p class="rules-none">No policy rules fired. Outfit is within the dress code.</p>`;
  }
  return rules.map((r) => `
    <div class="rule ${r.advisory_only ? "advisory" : ""}">
      <div class="rule-head">
        <span class="rule-desc">${esc(r.description)}</span>
        <span class="rule-sev">sev ${(r.severity ?? 0).toFixed(2)}</span>
      </div>
      <div class="rule-meta">
        <span class="rule-cat">${esc(titleCase(r.category))}</span>
        ${r.advisory_only ? `<span>advisory only</span>` : `<span>weight ${(r.weight ?? 0).toFixed(2)}</span>`}
        ${r.citation ? `<span class="rule-cite">${esc(r.citation)}</span>` : ""}
      </div>
    </div>`).join("");
}

function renderCoverage(c) {
  const regions = Object.entries(c.regions || {})
    .map(([k, v]) => `<span class="cov-pill">${esc(k)} ${pct(v)}</span>`)
    .join("");
  return `
    <div class="coverage-item">
      <div class="coverage-head">
        <span class="coverage-slot">${esc(c.slot)}</span>
        <span class="coverage-type">${esc(titleCase(c.garment_type))}</span>
      </div>
      <div class="coverage-regions">${regions || '<span class="garment-empty">-</span>'}</div>
    </div>`;
}

function renderRemediation(rem) {
  if (!rem || !(rem.steps && rem.steps.length)) return "";
  const badge = rem.verified
    ? `<span class="rem-verified">Verified fix</span>`
    : `<span class="rem-unverified">Suggested</span>`;
  const resulting = rem.resulting_decision
    ? ` → ${esc(DECISION_LABEL[rem.resulting_decision] || titleCase(rem.resulting_decision))}`
    : "";
  return `
    <div class="remediation">
      <div class="rem-head">How to become compliant ${badge}<span style="color:var(--text-faint);font-weight:400;font-size:.8rem">${resulting}</span></div>
      <ul class="rem-steps">${rem.steps.map((st) => `<li>${esc(st)}</li>`).join("")}</ul>
    </div>`;
}

// Compact summary for the live camera overlay tag.
function overlayFromFrame(frame) {
  const persons = frame.persons || [];
  if (!persons.length) return { label: "No person", cls: "d-insufficient_evidence", color: "var(--gray)" };
  // Pick the most severe person in view.
  const order = { major_violation: 3, minor_violation: 2, insufficient_evidence: 1, compliant: 0 };
  const worst = persons.reduce((a, b) => (order[b.decision] > order[a.decision] ? b : a));
  const colorMap = {
    compliant: "var(--green)", minor_violation: "var(--amber)",
    major_violation: "var(--red)", insufficient_evidence: "var(--gray)",
  };
  const ov = worst.scores?.overall_violation;
  const conf = ov === null || ov === undefined ? "" : ` · ${pct(ov)}`;
  return { label: `${DECISION_LABEL[worst.decision]}${conf}`, cls: `d-${worst.decision}`, color: colorMap[worst.decision] };
}

// ===================================================================
//  Upload mode
// ===================================================================
let selectedFile = null;

const dropzone = $("#dropzone");
const fileInput = $("#fileInput");
const dzEmpty = $("#dropzoneEmpty");
const dzPreview = $("#dropzonePreview");
const previewImg = $("#previewImg");
const analyzeBtn = $("#analyzeBtn");
const uploadScanOverlay = $("#uploadScanOverlay");

const ACCEPTED = ["image/jpeg", "image/png", "image/webp"];

function handleFile(file) {
  if (!file) return;
  if (!ACCEPTED.includes(file.type)) {
    Toast.show({ type: "warn", title: "Unsupported file", msg: "Please use a JPG, PNG, or WEBP image." });
    return;
  }
  selectedFile = file;
  const reader = new FileReader();
  reader.onload = (e) => {
    previewImg.src = e.target.result;
    dzEmpty.hidden = true;
    dzPreview.hidden = false;
    analyzeBtn.disabled = false;
  };
  reader.readAsDataURL(file);
}

function clearFile() {
  selectedFile = null;
  fileInput.value = "";
  previewImg.src = "";
  dzEmpty.hidden = false;
  dzPreview.hidden = true;
  analyzeBtn.disabled = true;
}

dropzone.addEventListener("click", (e) => {
  if (e.target.closest(".remove-btn")) return;
  if (!selectedFile) fileInput.click();
});
dropzone.addEventListener("keydown", (e) => {
  if ((e.key === "Enter" || e.key === " ") && !selectedFile) { e.preventDefault(); fileInput.click(); }
});
fileInput.addEventListener("change", (e) => handleFile(e.target.files[0]));
$("#removeBtn").addEventListener("click", (e) => { e.stopPropagation(); clearFile(); });

["dragenter", "dragover"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.add("dragover"); }));
["dragleave", "drop"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.remove("dragover"); }));
dropzone.addEventListener("drop", (e) => {
  const file = e.dataTransfer.files?.[0];
  if (file) handleFile(file);
});

analyzeBtn.addEventListener("click", async () => {
  if (!selectedFile) return;
  analyzeBtn.disabled = true;
  analyzeBtn.querySelector(".btn-label").textContent = "Analyzing…";
  uploadScanOverlay.hidden = false;
  showResultsState("loading");
  try {
    const frame = await inferImage(selectedFile);
    renderFrame(frame);
  } catch (err) {
    showResultsState("empty");
    Toast.show({ type: "error", title: "Analysis failed", msg: err.message });
  } finally {
    analyzeBtn.disabled = false;
    analyzeBtn.querySelector(".btn-label").textContent = "Analyze";
    uploadScanOverlay.hidden = true;
  }
});

// ===================================================================
//  Camera mode
// ===================================================================
const video = $("#video");
const canvas = $("#canvas");
const camOverlay = $("#camOverlay");
const liveBadge = $("#liveBadge");
const cameraPlaceholder = $("#cameraPlaceholder");
const cameraToggleBtn = $("#cameraToggleBtn");
const captureBtn = $("#captureBtn");
const realtimeToggle = $("#realtimeToggle");

let stream = null;
let realtimeTimer = null;
let realtimeBusy = false;

async function startCamera() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" }, audio: false });
    video.srcObject = stream;
    cameraPlaceholder.hidden = true;
    liveBadge.hidden = false;
    captureBtn.disabled = false;
    realtimeToggle.disabled = false;
    cameraToggleBtn.textContent = "Stop Camera";
    Toast.show({ type: "success", title: "Camera started", msg: "Live feed is mirrored for a natural view." });
  } catch (err) {
    Toast.show({ type: "error", title: "Camera access denied", msg: err.message || "Could not access the webcam." });
  }
}

function stopCamera() {
  stopRealtime();
  if (stream) { stream.getTracks().forEach((t) => t.stop()); stream = null; }
  video.srcObject = null;
  cameraPlaceholder.hidden = false;
  liveBadge.hidden = true;
  captureBtn.disabled = true;
  realtimeToggle.disabled = true;
  realtimeToggle.checked = false;
  camOverlay.innerHTML = "";
  cameraToggleBtn.textContent = "Start Camera";
}

cameraToggleBtn.addEventListener("click", () => (stream ? stopCamera() : startCamera()));

// Grab the current video frame as a JPEG blob (un-mirrored - the mirror is only visual).
function grabFrameBlob() {
  return new Promise((resolve, reject) => {
    if (!video.videoWidth) return reject(new Error("Camera not ready"));
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    canvas.toBlob((blob) => (blob ? resolve(blob) : reject(new Error("Capture failed"))), "image/jpeg", CAPTURE_QUALITY);
  });
}

captureBtn.addEventListener("click", async () => {
  captureBtn.disabled = true;
  showResultsState("loading");
  try {
    const blob = await grabFrameBlob();
    const frame = await inferImage(blob);
    renderFrame(frame);
    setOverlay(frame);
  } catch (err) {
    showResultsState("empty");
    Toast.show({ type: "error", title: "Capture failed", msg: err.message });
  } finally {
    captureBtn.disabled = !stream;
  }
});

function setOverlay(frame) {
  const o = overlayFromFrame(frame);
  camOverlay.innerHTML = `<div class="cam-tag" style="color:${o.color};background:rgba(10,10,15,.7)">${esc(o.label)}</div>`;
}

// ---- Real-time continuous scan ----
function startRealtime() {
  if (realtimeTimer || !stream) return;
  Toast.show({ type: "info", title: "Real-time scan on", msg: `Auto-analyzing every ${REALTIME_INTERVAL_MS / 1000}s.` });
  const tick = async () => {
    if (realtimeBusy || !stream) return;
    realtimeBusy = true;
    try {
      const blob = await grabFrameBlob();
      const frame = await inferImage(blob, { silent: true });
      renderFrame(frame);
      setOverlay(frame);
    } catch (err) {
      // Stay quiet during live scanning, but surface a backend drop once.
      if (apiDot.dataset.state === "online") setApiState("offline", "Model unavailable");
    } finally {
      realtimeBusy = false;
    }
  };
  tick();
  realtimeTimer = setInterval(tick, REALTIME_INTERVAL_MS);
}

function stopRealtime() {
  if (realtimeTimer) { clearInterval(realtimeTimer); realtimeTimer = null; }
}

realtimeToggle.addEventListener("change", (e) => {
  if (e.target.checked) startRealtime();
  else { stopRealtime(); Toast.show({ type: "info", title: "Real-time scan off", duration: 2500 }); }
});

// ===================================================================
//  Mode switching
// ===================================================================
const uploadPanel = $("#uploadPanel");
const cameraPanel = $("#cameraPanel");

function setMode(mode) {
  const upload = mode === "upload";
  $("#modeUploadBtn").classList.toggle("active", upload);
  $("#modeCameraBtn").classList.toggle("active", !upload);
  $("#modeUploadBtn").setAttribute("aria-selected", String(upload));
  $("#modeCameraBtn").setAttribute("aria-selected", String(!upload));
  uploadPanel.hidden = !upload;
  cameraPanel.hidden = upload;
  if (upload) stopCamera(); // free the webcam when leaving camera mode
}
$("#modeUploadBtn").addEventListener("click", () => setMode("upload"));
$("#modeCameraBtn").addEventListener("click", () => setMode("camera"));

// ===================================================================
//  Hero stats counters + navbar scroll + reveal on scroll
// ===================================================================
function animateCounters() {
  $$(".stat-num").forEach((el) => {
    const target = Number(el.dataset.target) || 0;
    const prefix = (el.dataset.prefix || "").replace("&lt;", "<");
    const suffix = el.dataset.suffix || "";
    const dur = 1400;
    const start = performance.now();
    const step = (now) => {
      const t = Math.min(1, (now - start) / dur);
      const eased = 1 - Math.pow(1 - t, 3);
      el.textContent = `${prefix}${Math.round(target * eased)}${suffix}`;
      if (t < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  });
}

const navbar = $("#navbar");
window.addEventListener("scroll", () => navbar.classList.toggle("scrolled", window.scrollY > 20), { passive: true });

// ===================================================================
//  Boot
// ===================================================================
window.addEventListener("DOMContentLoaded", async () => {
  showResultsState("empty");
  animateCounters();
  await Promise.all([checkHealth(), loadProfiles()]);
});

window.addEventListener("beforeunload", () => stopCamera());

// ===================================================================
//  Pointer-reactive glass cards (the spotlight follows the cursor)
// ===================================================================
$$(".feature, .step").forEach((card) => {
  card.addEventListener("pointermove", (e) => {
    const r = card.getBoundingClientRect();
    card.style.setProperty("--mx", `${e.clientX - r.left}px`);
    card.style.setProperty("--my", `${e.clientY - r.top}px`);
  });
});
