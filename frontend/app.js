const API = "/api";

const el = (id) => document.getElementById(id);

const dropzone = el("dropzone");
const fileInput = el("file-input");
const fileInfoRow = el("file-info-row");
const uploadError = el("upload-error");
const paramsPanel = el("params-panel");
const resultPanel = el("result-panel");
const progressRow = el("progress-row");
const progressFill = el("progress-fill");
const progressLabel = el("progress-label");
const convertError = el("convert-error");

let currentJobId = null;
let pollTimer = null;

function humanSize(bytes) {
  const units = ["o", "Ko", "Mo", "Go"];
  let size = bytes;
  let i = 0;
  while (size >= 1024 && i < units.length - 1) {
    size /= 1024;
    i += 1;
  }
  return `${size.toFixed(1)} ${units[i]}`;
}

function showBanner(node, message, cls) {
  node.textContent = message;
  node.className = `banner ${cls}`;
  node.hidden = !message;
}

function resetUiForNewFile() {
  paramsPanel.hidden = true;
  resultPanel.hidden = true;
  progressRow.hidden = true;
  showBanner(convertError, "", "error");
  showBanner(uploadError, "", "error");
  if (pollTimer) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
}

["dragenter", "dragover"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  })
);
["dragleave", "drop"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
  })
);
dropzone.addEventListener("drop", (e) => {
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});
dropzone.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) handleFile(fileInput.files[0]);
});

el("btn-clear").addEventListener("click", () => {
  currentJobId = null;
  fileInfoRow.hidden = true;
  fileInput.value = "";
  resetUiForNewFile();
});

async function handleFile(file) {
  resetUiForNewFile();
  fileInfoRow.hidden = false;
  el("file-name").textContent = file.name;
  el("file-meta").textContent = `(${humanSize(file.size)})`;

  const form = new FormData();
  form.append("file", file);

  try {
    const res = await fetch(`${API}/upload`, { method: "POST", body: form });
    const body = await res.json();
    if (!res.ok) {
      showBanner(uploadError, body.detail || "Échec de l'upload.", "error");
      return;
    }
    currentJobId = body.job_id;
    paramsPanel.hidden = false;
  } catch (err) {
    showBanner(uploadError, `Erreur réseau : ${err}`, "error");
  }
}

el("btn-convert").addEventListener("click", startConversion);

async function startConversion() {
  if (!currentJobId) return;
  showBanner(convertError, "", "error");
  resultPanel.hidden = true;

  const params = {
    k_neighbors: parseInt(el("p-k").value, 10),
    scale_factor: parseFloat(el("p-scale").value),
    anisotropy_ratio: parseFloat(el("p-aniso").value),
    opacity_dense: parseFloat(el("p-op-dense").value),
    opacity_sparse: parseFloat(el("p-op-sparse").value),
    center_mode: el("p-center").value,
    voxel_size: parseFloat(el("p-voxel").value) || null,
    include_f_rest: el("p-frest").checked,
    convert_z_up_to_y_up: el("p-zup").checked,
  };

  el("btn-convert").disabled = true;
  progressRow.hidden = false;
  progressFill.style.width = "0%";
  progressLabel.textContent = "Démarrage…";

  try {
    const res = await fetch(`${API}/convert/${currentJobId}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(params),
    });
    const body = await res.json();
    if (!res.ok) {
      showBanner(convertError, body.detail || "Échec du démarrage de la conversion.", "error");
      el("btn-convert").disabled = false;
      progressRow.hidden = true;
      return;
    }
    pollTimer = setInterval(pollJob, 800);
  } catch (err) {
    showBanner(convertError, `Erreur réseau : ${err}`, "error");
    el("btn-convert").disabled = false;
  }
}

async function pollJob() {
  try {
    const res = await fetch(`${API}/jobs/${currentJobId}`);
    const job = await res.json();
    if (!res.ok) return;

    const pct = Math.round((job.progress || 0) * 100);
    progressFill.style.width = `${pct}%`;
    progressLabel.textContent = `${pct}% — ${job.label || ""}`;

    if (job.status === "done") {
      clearInterval(pollTimer);
      pollTimer = null;
      el("btn-convert").disabled = false;
      showResult(job.result);
    } else if (job.status === "error") {
      clearInterval(pollTimer);
      pollTimer = null;
      el("btn-convert").disabled = false;
      progressRow.hidden = true;
      showBanner(convertError, job.error || "Erreur inconnue pendant la conversion.", "error");
    }
  } catch (err) {
    // transient network hiccup while polling: keep trying silently
  }
}

function showResult(result) {
  resultPanel.hidden = false;
  el("stat-source").textContent = result.n_points_source.toLocaleString("fr-CH");
  el("stat-splats").textContent = result.n_splats.toLocaleString("fr-CH");
  el("stat-size").textContent = humanSize(result.file_size_bytes);
  el("stat-time").textContent = `${result.elapsed_seconds} s`;

  const warningsDiv = el("result-warnings");
  warningsDiv.innerHTML = "";
  (result.warnings || []).forEach((w) => {
    const banner = document.createElement("div");
    banner.className = "banner warn";
    banner.textContent = w;
    warningsDiv.appendChild(banner);
  });
}

el("btn-download").addEventListener("click", () => {
  if (currentJobId) window.location.href = `${API}/download/${currentJobId}`;
});
el("btn-download-offset").addEventListener("click", () => {
  if (currentJobId) window.location.href = `${API}/download/${currentJobId}/offset`;
});
