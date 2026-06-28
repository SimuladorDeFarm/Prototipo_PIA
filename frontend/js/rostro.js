const API_ROSTRO = "http://localhost:8000";

const EMOJIS_ROSTRO = {
  neutral: "😐",
  happy: "😄",
  sad: "😢",
  anger: "😠",
  fear: "😨",
  disgust: "🤢",
  surprise: "😲",
};

const fileInputR = document.getElementById("fileInputR");
const fileLabelR = document.getElementById("fileLabelR");
const dropZoneR = document.getElementById("dropZoneR");
const cameraBtn = document.getElementById("cameraBtn");
const captureBtn = document.getElementById("captureBtn");
const video = document.getElementById("video");
const preview = document.getElementById("preview");
const canvas = document.getElementById("canvas");
const predictBtnR = document.getElementById("predictBtnR");

const resultCardR = document.getElementById("resultCardR");
const statusR = document.getElementById("statusR");
const resultBodyR = document.getElementById("resultBodyR");
const emojiBigR = document.getElementById("emojiBigR");
const emotionNameR = document.getElementById("emotionNameR");
const confidenceR = document.getElementById("confidenceR");
const rankingR = document.getElementById("rankingR");

// Imagen activa a predecir: un Blob (sea archivo subido o captura de cámara).
let imagenBlob = null;
let camStream = null;

// --- Selección de archivo -------------------------------------------------
fileInputR.addEventListener("change", () => {
  const file = fileInputR.files[0];
  if (!file) return;
  imagenBlob = file;
  fileLabelR.textContent = file.name;
  dropZoneR.classList.add("has-file");
  mostrarPreview(URL.createObjectURL(file));
  detenerCamara();
  predictBtnR.disabled = false;
});

// --- Cámara ---------------------------------------------------------------
cameraBtn.addEventListener("click", async () => {
  if (camStream) {
    detenerCamara();
    return;
  }
  try {
    camStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" } });
    video.srcObject = camStream;
    video.hidden = false;
    preview.hidden = true;
    await video.play();
    cameraBtn.textContent = "✖️ Cerrar cámara";
    captureBtn.hidden = false;
  } catch (err) {
    mostrarEstadoR(`⚠️ No se pudo acceder a la cámara: ${err.message}`, true);
  }
});

captureBtn.addEventListener("click", () => {
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext("2d").drawImage(video, 0, 0);
  canvas.toBlob((blob) => {
    imagenBlob = blob;
    fileLabelR.textContent = "Captura de cámara";
    dropZoneR.classList.add("has-file");
    mostrarPreview(URL.createObjectURL(blob));
    detenerCamara();
    predictBtnR.disabled = false;
  }, "image/jpeg", 0.95);
});

function detenerCamara() {
  if (camStream) {
    camStream.getTracks().forEach((t) => t.stop());
    camStream = null;
  }
  video.hidden = true;
  captureBtn.hidden = true;
  cameraBtn.textContent = "📷 Usar cámara";
}

function mostrarPreview(src) {
  preview.src = src;
  preview.hidden = false;
}

// --- Predicción -----------------------------------------------------------
predictBtnR.addEventListener("click", () => {
  if (!imagenBlob) return;
  const form = new FormData();
  const nombre = imagenBlob.name || "captura.jpg";
  form.append("imagen", imagenBlob, nombre);
  enviarR(`${API_ROSTRO}/predecir_rostro`, { method: "POST", body: form });
});

async function enviarR(url, opciones) {
  mostrarEstadoR("Analizando imagen…");
  predictBtnR.disabled = true;
  try {
    const res = await fetch(url, opciones);
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Error en el servidor");
    mostrarResultadoR(data);
  } catch (err) {
    mostrarEstadoR(`⚠️ ${err.message}`, true);
  } finally {
    predictBtnR.disabled = !imagenBlob;
  }
}

function mostrarEstadoR(texto, esError = false) {
  resultCardR.hidden = false;
  resultBodyR.hidden = true;
  statusR.hidden = false;
  statusR.textContent = texto;
  statusR.classList.toggle("error", esError);
}

function mostrarResultadoR(data) {
  resultCardR.hidden = false;
  statusR.hidden = true;
  resultBodyR.hidden = false;

  emojiBigR.textContent = EMOJIS_ROSTRO[data.emocion] || "🙂";
  emotionNameR.textContent = data.emocion_es || data.emocion;
  confidenceR.textContent = `Confianza: ${(data.confianza * 100).toFixed(1)}%`;

  rankingR.innerHTML = "";
  for (const [emocion, prob] of data.ranking) {
    const pct = (prob * 100).toFixed(1);
    const row = document.createElement("div");
    row.className = "rank-row";
    row.innerHTML = `
      <span class="rank-label">${EMOJIS_ROSTRO[emocion] || ""} ${emocion}</span>
      <span class="rank-track"><span class="rank-fill" style="width:${pct}%"></span></span>
      <span class="rank-pct">${pct}%</span>`;
    rankingR.appendChild(row);
  }
}
