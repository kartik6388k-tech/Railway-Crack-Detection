(() => {
  "use strict";

  const ALLOWED_TYPES = ["image/jpeg", "image/png", "image/webp"];
  const ALLOWED_EXT = [".jpg", ".jpeg", ".png", ".webp"];

  const dropzone = document.getElementById("dropzone");
  const dropzoneEmpty = document.getElementById("dropzoneEmpty");
  const fileInput = document.getElementById("fileInput");
  const previewWrap = document.getElementById("previewWrap");
  const previewImg = document.getElementById("previewImg");

  const scanBtn = document.getElementById("scanBtn");
  const resetBtn = document.getElementById("resetBtn");

  const errorBanner = document.getElementById("errorBanner");
  const errorMessage = document.getElementById("errorMessage");

  const statusBand = document.getElementById("statusBand");
  const statusIcon = document.getElementById("statusIcon");
  const statusTitle = document.getElementById("statusTitle");
  const statusSubtitle = document.getElementById("statusSubtitle");

  const detectionsList = document.getElementById("detectionsList");
  const resultsPlaceholder = document.getElementById("resultsPlaceholder");
  const previewFilename = document.getElementById("previewFilename");

  const settingsToggle = document.getElementById("settingsToggle");
  const settingsPanel = document.getElementById("settingsPanel");
  const confInput = document.getElementById("confInput");
  const confVal = document.getElementById("confVal");
  const iouInput = document.getElementById("iouInput");
  const iouVal = document.getElementById("iouVal");
  const imgszInput = document.getElementById("imgszInput");
  const maxDetInput = document.getElementById("maxDetInput");
  const deviceInput = document.getElementById("deviceInput");

  let selectedFile = null;
  let objectUrl = null;

  const ICONS = {
    alert: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" width="18" height="18"><path d="M12 9v4M12 17h.01M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0Z"></path></svg>',
    check: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" width="18" height="18"><path d="M20 6 9 17l-5-5"></path></svg>',
    scan: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" width="18" height="18"><path d="M3 7V4a1 1 0 0 1 1-1h3M17 3h3a1 1 0 0 1 1 1v3M21 17v3a1 1 0 0 1-1 1h-3M7 21H4a1 1 0 0 1-1-1v-3"></path></svg>',
  };

  // ------------------------------------------------------------------
  // Settings panel
  // ------------------------------------------------------------------

  settingsToggle.addEventListener("click", () => {
    const isOpen = !settingsPanel.hidden;
    settingsPanel.hidden = isOpen;
    settingsToggle.setAttribute("aria-expanded", String(!isOpen));
  });

  confInput.addEventListener("input", () => {
    confVal.textContent = Number(confInput.value).toFixed(2);
  });
  iouInput.addEventListener("input", () => {
    iouVal.textContent = Number(iouInput.value).toFixed(2);
  });

  // ------------------------------------------------------------------
  // File selection (click, keyboard, drag & drop)
  // ------------------------------------------------------------------

  dropzone.addEventListener("click", () => fileInput.click());
  dropzone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      fileInput.click();
    }
  });

  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("is-dragover");
  });
  dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("is-dragover");
  });
  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("is-dragover");
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener("change", () => {
    if (fileInput.files && fileInput.files[0]) {
      handleFile(fileInput.files[0]);
    }
  });

  function isAllowedFile(file) {
    if (ALLOWED_TYPES.includes(file.type)) return true;
    const name = file.name.toLowerCase();
    return ALLOWED_EXT.some((ext) => name.endsWith(ext));
  }

  function handleFile(file) {
    hideError();
    clearResults();

    if (!isAllowedFile(file)) {
      showError(`Unsupported file type. Use ${ALLOWED_EXT.join(", ")}.`);
      return;
    }

    selectedFile = file;

    if (objectUrl) URL.revokeObjectURL(objectUrl);
    objectUrl = URL.createObjectURL(file);
    previewImg.src = objectUrl;
    previewFilename.textContent = file.name;

    dropzoneEmpty.hidden = true;
    previewWrap.hidden = false;

    scanBtn.disabled = false;
    resetBtn.disabled = false;
  }

  // ------------------------------------------------------------------
  // Scan
  // ------------------------------------------------------------------

  scanBtn.addEventListener("click", runScan);

  async function runScan() {
    if (!selectedFile) return;

    hideError();
    clearResults();
    setScanning(true);

    const formData = new FormData();
    formData.append("file", selectedFile);
    formData.append("conf", confInput.value);
    formData.append("iou", iouInput.value);
    formData.append("imgsz", imgszInput.value);
    formData.append("max_det", maxDetInput.value);
    if (deviceInput.value) formData.append("device", deviceInput.value);

    try {
      const res = await fetch("/api/predict", { method: "POST", body: formData });
      const data = await parseResponse(res);

      if (!res.ok || data.error) {
        const status = `HTTP ${res.status} ${res.statusText || ""}`.trim();
        const detail = data.error || (data.raw ? `${status} — ${data.raw}` : status);
        throw new Error(`Scan failed: ${detail}`);
      }

      renderResult(data);
    } catch (err) {
      clearResults();
      showError(err.message || "Scan failed. Check the connection and try again.");
    } finally {
      setScanning(false);
    }
  }

  // Reads a fetch Response as text first, then attempts JSON parsing.
  // This means a 500 error page, a crashed backend returning HTML/plain
  // text, or an empty body all fail with a readable message instead of
  // res.json() throwing a raw "Unexpected token" parse error.
  async function parseResponse(res) {
    let text = "";
    try {
      text = await res.text();
    } catch {
      return { raw: `HTTP ${res.status} ${res.statusText || ""}`.trim() };
    }
    if (!text) return {};
    try {
      return JSON.parse(text);
    } catch {
      return { raw: text.slice(0, 200).trim() || `HTTP ${res.status}` };
    }
  }

  function setScanning(isScanning) {
    previewWrap.classList.toggle("is-scanning", isScanning);
    scanBtn.disabled = isScanning || !selectedFile;
    resetBtn.disabled = isScanning;
    scanBtn.textContent = isScanning ? "Scanning…" : "Scan for defects";

    if (isScanning) {
      resultsPlaceholder.hidden = true;
      statusBand.hidden = false;
      statusBand.className = "status-band status-band--scanning";
      statusIcon.innerHTML = ICONS.scan;
      statusTitle.textContent = "Scanning for defects";
      statusSubtitle.textContent = "Running inference on the uploaded frame";
    }
  }

  // ------------------------------------------------------------------
  // Render result
  // ------------------------------------------------------------------

  function renderResult(data) {
    const detections = Array.isArray(data.detections) ? data.detections : [];
    // Trust the detections array itself over total_detections: if the
    // backend ever omits or miscounts that field, the array length is
    // still the ground truth and prevents a false "no defects" result.
    const count = data.detections !== undefined ? detections.length : (data.total_detections || 0);
    const hasDefect = count > 0;

    resultsPlaceholder.hidden = true;
    statusBand.hidden = false;
    statusBand.className = `status-band ${hasDefect ? "status-band--defect" : "status-band--clear"}`;
    statusIcon.innerHTML = hasDefect ? ICONS.alert : ICONS.check;
    statusTitle.textContent = hasDefect
      ? `${count} defect${count === 1 ? "" : "s"} detected`
      : "No gap detected";
    statusSubtitle.textContent = hasDefect
      ? "Review the flagged regions below"
      : "This segment scanned clear at the current settings";

    if (data.result_image_url) {
      previewImg.onerror = () => {
        previewImg.onerror = null;
        if (objectUrl) previewImg.src = objectUrl; // fall back to the original upload
      };
      previewImg.src = data.result_image_url;
    }

    detectionsList.innerHTML = "";
    if (hasDefect && detections.length) {
      detections.forEach((d, i) => {
        detectionsList.appendChild(buildDetectionItem(d, i));
      });
      detectionsList.hidden = false;
    } else {
      detectionsList.hidden = true;
    }
  }

  function buildDetectionItem(detection, index) {
    const li = document.createElement("li");
    li.className = "detection-item";
    li.style.animationDelay = `${index * 40}ms`;

    const confPct = Math.round((detection.confidence || 0) * 100);
    const bbox = detection.bbox || {};

    li.innerHTML = `
      <div class="detection-item__top">
        <span class="detection-item__class">${escapeHtml(detection.class_name ?? "unknown")}</span>
        <span class="detection-item__conf">${confPct}%</span>
      </div>
      <div class="confidence-track">
        <div class="confidence-fill" style="width:${confPct}%"></div>
      </div>
      <div class="detection-item__bbox">x1 ${bbox.x1 ?? "—"} · y1 ${bbox.y1 ?? "—"} · x2 ${bbox.x2 ?? "—"} · y2 ${bbox.y2 ?? "—"}</div>
    `;
    return li;
  }

  function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = String(str);
    return div.innerHTML;
  }

  // ------------------------------------------------------------------
  // Reset / error helpers
  // ------------------------------------------------------------------

  resetBtn.addEventListener("click", resetAll);

  function resetAll() {
    selectedFile = null;
    fileInput.value = "";
    if (objectUrl) {
      URL.revokeObjectURL(objectUrl);
      objectUrl = null;
    }
    previewImg.removeAttribute("src");
    previewFilename.textContent = "";
    previewWrap.hidden = true;
    previewWrap.classList.remove("is-scanning");
    dropzoneEmpty.hidden = false;

    scanBtn.disabled = true;
    scanBtn.textContent = "Scan for defects";
    resetBtn.disabled = true;

    hideError();
    clearResults();
  }

  function clearResults() {
    statusBand.hidden = true;
    detectionsList.hidden = true;
    detectionsList.innerHTML = "";
    resultsPlaceholder.hidden = false;
  }

  function showError(message) {
    errorMessage.textContent = message;
    errorBanner.hidden = false;
  }

  function hideError() {
    errorBanner.hidden = true;
  }

  // ------------------------------------------------------------------
  // Initial state
  // ------------------------------------------------------------------
  // Guarantees a clean starting UI (no stray error banner, no stray
  // results) the moment this script runs, regardless of what markup
  // state the page happened to load with.
  hideError();
  clearResults();
})();