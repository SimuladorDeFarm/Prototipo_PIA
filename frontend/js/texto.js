const API_TEXTO = "http://localhost:8000";

const EMOJIS_TEXTO = {
  others: "😐",
  joy: "😄",
  sadness: "😢",
  anger: "😠",
  fear: "😨",
  disgust: "🤢",
  surprise: "😲",
};

const textoInput = document.getElementById("textoInput");
const charCount = document.getElementById("charCount");
const predictBtnT = document.getElementById("predictBtnT");
const clearBtnT = document.getElementById("clearBtnT");

const resultCardT = document.getElementById("resultCardT");
const statusT = document.getElementById("statusT");
const resultBodyT = document.getElementById("resultBodyT");
const emojiBigT = document.getElementById("emojiBigT");
const emotionNameT = document.getElementById("emotionNameT");
const confidenceT = document.getElementById("confidenceT");
const rankingT = document.getElementById("rankingT");

textoInput.addEventListener("input", () => {
  charCount.textContent = textoInput.value.length;
  predictBtnT.disabled = textoInput.value.trim().length === 0;
});

clearBtnT.addEventListener("click", () => {
  textoInput.value = "";
  charCount.textContent = "0";
  predictBtnT.disabled = true;
  resultCardT.hidden = true;
});

predictBtnT.addEventListener("click", () => {
  const texto = textoInput.value.trim();
  if (texto.length > 0) enviarT(texto);
});

textoInput.addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
    e.preventDefault();
    const texto = textoInput.value.trim();
    if (texto.length > 0) enviarT(texto);
  }
});

// --- Predicción -----------------------------------------------------------
async function enviarT(texto) {
  mostrarEstadoT("Analizando texto…");
  predictBtnT.disabled = true;
  clearBtnT.disabled = true;
  try {
    const res = await fetch(`${API_TEXTO}/predecir_texto`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ texto }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Error en el servidor");
    mostrarResultadoT(data);
  } catch (err) {
    mostrarEstadoT(`⚠️ ${err.message}`, true);
  } finally {
    predictBtnT.disabled = textoInput.value.trim().length === 0;
    clearBtnT.disabled = false;
  }
}

function mostrarEstadoT(texto, esError = false) {
  resultCardT.hidden = false;
  resultBodyT.hidden = true;
  statusT.hidden = false;
  statusT.textContent = texto;
  statusT.classList.toggle("error", esError);
}

function mostrarResultadoT(data) {
  resultCardT.hidden = false;
  statusT.hidden = true;
  resultBodyT.hidden = false;

  emojiBigT.textContent = EMOJIS_TEXTO[data.emocion] || "🙂";
  emotionNameT.textContent = data.emocion_es || data.emocion;
  confidenceT.textContent = `Confianza: ${(data.confianza * 100).toFixed(1)}%`;

  rankingT.innerHTML = "";
  for (const [emocion, prob] of data.ranking) {
    const pct = (prob * 100).toFixed(1);
    const row = document.createElement("div");
    row.className = "rank-row";
    row.innerHTML = `
      <span class="rank-label">${EMOJIS_TEXTO[emocion] || ""} ${emocion}</span>
      <span class="rank-track"><span class="rank-fill" style="width:${pct}%"></span></span>
      <span class="rank-pct">${pct}%</span>`;
    rankingT.appendChild(row);
  }
}
