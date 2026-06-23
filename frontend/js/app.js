const API = "http://localhost:8000";

const EMOJIS = {
  neutral: "😐",
  joy: "😄",
  sadness: "😢",
  anger: "😠",
  fear: "😨",
  disgust: "🤢",
  surprise: "😲",
};

const fileInput = document.getElementById("fileInput");
const fileLabel = document.getElementById("fileLabel");
const dropZone = document.getElementById("dropZone");
const recordBtn = document.getElementById("recordBtn");
const recordTime = document.getElementById("recordTime");
const player = document.getElementById("player");
const predictBtn = document.getElementById("predictBtn");
const testBtn = document.getElementById("testBtn");

const resultCard = document.getElementById("resultCard");
const statusEl = document.getElementById("status");
const resultBody = document.getElementById("resultBody");
const emojiBig = document.getElementById("emojiBig");
const emotionName = document.getElementById("emotionName");
const confidence = document.getElementById("confidence");
const ranking = document.getElementById("ranking");

// Audio activo a predecir: un Blob WAV (sea archivo subido o grabación).
let audioWav = null;

let mediaRecorder = null;
let chunks = [];
let timerId = null;
let segundos = 0;

// --- Selección de archivo -------------------------------------------------
fileInput.addEventListener("change", async () => {
  const file = fileInput.files[0];
  if (!file) return;
  audioWav = file;
  fileLabel.textContent = file.name;
  dropZone.classList.add("has-file");
  cargarReproductor(file);
  predictBtn.disabled = false;
});

// --- Grabación de voz -----------------------------------------------------
recordBtn.addEventListener("click", async () => {
  if (mediaRecorder && mediaRecorder.state === "recording") {
    detenerGrabacion();
    return;
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    chunks = [];
    mediaRecorder = new MediaRecorder(stream);
    mediaRecorder.ondataavailable = (e) => chunks.push(e.data);
    mediaRecorder.onstop = () => procesarGrabacion(stream);
    mediaRecorder.start();
    iniciarUI();
  } catch (err) {
    mostrarEstado(`⚠️ No se pudo acceder al micrófono: ${err.message}`, true);
  }
});

function iniciarUI() {
  recordBtn.textContent = "⏹️ Detener";
  recordBtn.classList.add("recording");
  recordTime.hidden = false;
  segundos = 0;
  recordTime.textContent = "0:00";
  timerId = setInterval(() => {
    segundos++;
    const m = Math.floor(segundos / 60);
    const s = String(segundos % 60).padStart(2, "0");
    recordTime.textContent = `${m}:${s}`;
  }, 1000);
}

function detenerGrabacion() {
  if (mediaRecorder && mediaRecorder.state === "recording") mediaRecorder.stop();
  clearInterval(timerId);
  recordBtn.textContent = "🎙️ Grabar voz";
  recordBtn.classList.remove("recording");
}

async function procesarGrabacion(stream) {
  stream.getTracks().forEach((t) => t.stop());
  const blob = new Blob(chunks, { type: mediaRecorder.mimeType });
  mostrarEstado("Convirtiendo grabación…");
  try {
    audioWav = await blobAWav(blob);
    fileLabel.textContent = "Grabación de voz";
    dropZone.classList.add("has-file");
    cargarReproductor(audioWav);
    predictBtn.disabled = false;
    statusEl.hidden = true;
    resultCard.hidden = true;
  } catch (err) {
    mostrarEstado(`⚠️ Error al convertir el audio: ${err.message}`, true);
  }
}

// Decodifica el audio grabado (webm/opus) y lo reencoda como WAV PCM 16-bit.
async function blobAWav(blob) {
  const arrayBuffer = await blob.arrayBuffer();
  const ctx = new (window.AudioContext || window.webkitAudioContext)();
  const audioBuffer = await ctx.decodeAudioData(arrayBuffer);
  ctx.close();
  return codificarWav(audioBuffer);
}

function codificarWav(audioBuffer) {
  const numCanales = audioBuffer.numberOfChannels;
  const sampleRate = audioBuffer.sampleRate;
  const numMuestras = audioBuffer.length;

  // Intercalar canales en un solo array.
  const canales = [];
  for (let c = 0; c < numCanales; c++) canales.push(audioBuffer.getChannelData(c));

  const buffer = new ArrayBuffer(44 + numMuestras * numCanales * 2);
  const view = new DataView(buffer);

  const escribirTexto = (offset, texto) => {
    for (let i = 0; i < texto.length; i++) view.setUint8(offset + i, texto.charCodeAt(i));
  };

  const bytesData = numMuestras * numCanales * 2;
  escribirTexto(0, "RIFF");
  view.setUint32(4, 36 + bytesData, true);
  escribirTexto(8, "WAVE");
  escribirTexto(12, "fmt ");
  view.setUint32(16, 16, true);          // tamaño subchunk fmt
  view.setUint16(20, 1, true);           // PCM
  view.setUint16(22, numCanales, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * numCanales * 2, true);  // byte rate
  view.setUint16(32, numCanales * 2, true);               // block align
  view.setUint16(34, 16, true);          // bits por muestra
  escribirTexto(36, "data");
  view.setUint32(40, bytesData, true);

  let offset = 44;
  for (let i = 0; i < numMuestras; i++) {
    for (let c = 0; c < numCanales; c++) {
      let muestra = Math.max(-1, Math.min(1, canales[c][i]));
      view.setInt16(offset, muestra < 0 ? muestra * 0x8000 : muestra * 0x7fff, true);
      offset += 2;
    }
  }

  return new Blob([view], { type: "audio/wav" });
}

function cargarReproductor(blobOFile) {
  player.src = URL.createObjectURL(blobOFile);
  player.hidden = false;
}

// --- Predicción -----------------------------------------------------------
predictBtn.addEventListener("click", () => {
  if (!audioWav) return;
  const form = new FormData();
  form.append("audio", audioWav, "audio.wav");
  enviar(`${API}/predecir`, { method: "POST", body: form });
});

testBtn.addEventListener("click", () => {
  enviar(`${API}/test`, { method: "GET" });
});

async function enviar(url, opciones) {
  mostrarEstado("Analizando audio…");
  predictBtn.disabled = true;
  testBtn.disabled = true;
  try {
    const res = await fetch(url, opciones);
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Error en el servidor");
    mostrarResultado(data);
  } catch (err) {
    mostrarEstado(`⚠️ ${err.message}`, true);
  } finally {
    predictBtn.disabled = !audioWav;
    testBtn.disabled = false;
  }
}

function mostrarEstado(texto, esError = false) {
  resultCard.hidden = false;
  resultBody.hidden = true;
  statusEl.hidden = false;
  statusEl.textContent = texto;
  statusEl.classList.toggle("error", esError);
}

function mostrarResultado(data) {
  resultCard.hidden = false;
  statusEl.hidden = true;
  resultBody.hidden = false;

  emojiBig.textContent = EMOJIS[data.emocion] || "🙂";
  emotionName.textContent = data.emocion_es || data.emocion;
  confidence.textContent = `Confianza: ${(data.confianza * 100).toFixed(1)}%`;

  ranking.innerHTML = "";
  for (const [emocion, prob] of data.ranking) {
    const pct = (prob * 100).toFixed(1);
    const row = document.createElement("div");
    row.className = "rank-row";
    row.innerHTML = `
      <span class="rank-label">${EMOJIS[emocion] || ""} ${emocion}</span>
      <span class="rank-track"><span class="rank-fill" style="width:${pct}%"></span></span>
      <span class="rank-pct">${pct}%</span>`;
    ranking.appendChild(row);
  }
}
