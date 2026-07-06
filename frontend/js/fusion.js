// ============================================================================
// Fusión multimodal (voto suave ponderado).
//
// Lanza los tres modelos a la vez y combina sus predicciones en una sola.
// Reutiliza los endpoints individuales que ya existen (/predecir,
// /predecir_rostro, /predecir_texto) y las entradas ya cargadas en los otros
// módulos (audioWav, imagenBlob, textoInput — variables globales de app.js,
// rostro.js y texto.js).
//
// La fusión replica el diseño acordado:
//   1. Alinear las etiquetas de cada modelo a las 7 clases canónicas
//      (rostro usa happy/sad; texto usa "others" = neutral).
//   2. Promediar las probabilidades ponderadas por el F1 de validación de cada
//      modelo, renormalizando sobre las modalidades presentes.
// ============================================================================

const API_FUSION = "http://localhost:8000";

// Orden canónico de las 7 emociones (el mismo del módulo de voz).
const CLASES_FUSION = ["neutral", "joy", "sadness", "anger", "fear", "disgust", "surprise"];

// Nombres en español para la emoción fusionada.
const NOMBRES_ES_FUSION = {
  neutral: "neutral", joy: "felicidad", sadness: "tristeza", anger: "enojo",
  fear: "miedo", disgust: "disgusto", surprise: "sorpresa",
};

// Cada modelo usa nombres de clase distintos → se mapean a los canónicos.
// (rostro: happy/sad ; texto: "others" es la clase neutral con otro nombre.)
const MAPA_ETIQUETAS = { happy: "joy", sad: "sadness", others: "neutral" };
const aCanonica = (etiqueta) => MAPA_ETIQUETAS[etiqueta] || etiqueta;

// Pesos = F1 macro de validación de cada modelo: a mayor F1, más confianza.
// Voz 0.675 · Rostro 0.473 · Texto 0.070 (texto está a nivel de azar, por eso
// pesa poco). Ajusta estos valores cuando reentrenes un modelo.
const PESOS = { voz: 0.675, rostro: 0.473, texto: 0.070 };

const fusionBtn = document.getElementById("fusionBtn");
const resultCardF = document.getElementById("resultCardF");
const statusF = document.getElementById("statusF");
const resultBodyF = document.getElementById("resultBodyF");
const emojiBigF = document.getElementById("emojiBigF");
const emotionNameF = document.getElementById("emotionNameF");
const confidenceF = document.getElementById("confidenceF");
const modalidadesF = document.getElementById("modalidadesF");
const rankingF = document.getElementById("rankingF");

fusionBtn.addEventListener("click", lanzarFusion);

// --- Orquestación ---------------------------------------------------------
async function lanzarFusion() {
  // Entradas ya cargadas en los otros módulos (globales; guardas por si acaso).
  const texto = typeof textoInput !== "undefined" ? textoInput.value.trim() : "";
  const tieneVoz = typeof audioWav !== "undefined" && !!audioWav;
  const tieneRostro = typeof imagenBlob !== "undefined" && !!imagenBlob;
  const tieneTexto = texto.length > 0;

  if (!tieneVoz && !tieneRostro && !tieneTexto) {
    mostrarEstadoF(
      "⚠️ Carga al menos una entrada (audio, imagen o texto) en los módulos de arriba.",
      true,
    );
    return;
  }

  mostrarEstadoF("Lanzando los modelos disponibles…");
  fusionBtn.disabled = true;

  // Se ejecutan en paralelo; ninguna promesa lanza (los errores se capturan).
  const estados = await Promise.all([
    tieneVoz ? ejecutar("voz", () => predecirVoz()) : ausente("voz", "sin audio"),
    tieneRostro ? ejecutar("rostro", () => predecirRostro()) : ausente("rostro", "sin imagen"),
    tieneTexto ? ejecutar("texto", () => predecirTexto(texto)) : ausente("texto", "sin texto"),
  ]);

  fusionBtn.disabled = false;

  const exitosos = estados.filter((e) => e.ok);
  if (exitosos.length === 0) {
    mostrarEstadoF("⚠️ Ningún modelo pudo predecir con las entradas actuales.", true);
    return;
  }

  mostrarResultadoF(fusionar(exitosos), estados);
}

const ausente = (modalidad, motivo) => Promise.resolve({ modalidad, ok: false, motivo });

async function ejecutar(modalidad, fn) {
  try {
    return { modalidad, ok: true, data: await fn() };
  } catch (err) {
    return { modalidad, ok: false, motivo: err.message };
  }
}

// --- Llamadas a los endpoints individuales --------------------------------
function predecirVoz() {
  const form = new FormData();
  form.append("audio", audioWav, "audio.wav");
  return pedir(`${API_FUSION}/predecir`, { method: "POST", body: form });
}

function predecirRostro() {
  const form = new FormData();
  form.append("imagen", imagenBlob, imagenBlob.name || "captura.jpg");
  return pedir(`${API_FUSION}/predecir_rostro`, { method: "POST", body: form });
}

function predecirTexto(texto) {
  return pedir(`${API_FUSION}/predecir_texto`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ texto }),
  });
}

async function pedir(url, opciones) {
  const res = await fetch(url, opciones);
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || "error del servidor");
  return data;
}

// --- Fusión (voto suave ponderado) ----------------------------------------
function fusionar(exitosos) {
  // Peso de cada modalidad presente; si todas quedaran en 0, reparte por igual
  // para no dividir por cero.
  let pesos = exitosos.map(({ modalidad }) => PESOS[modalidad] ?? 0);
  if (pesos.reduce((a, b) => a + b, 0) === 0) pesos = exitosos.map(() => 1);
  const total = pesos.reduce((a, b) => a + b, 0);

  const acumulado = Object.fromEntries(CLASES_FUSION.map((c) => [c, 0]));
  exitosos.forEach(({ data }, i) => {
    for (const [etiqueta, prob] of data.ranking) {
      const clase = aCanonica(etiqueta);
      if (clase in acumulado) acumulado[clase] += pesos[i] * prob;
    }
  });

  const ranking = CLASES_FUSION
    .map((clase) => [clase, acumulado[clase] / total])
    .sort((a, b) => b[1] - a[1]);

  return { emocion: ranking[0][0], confianza: ranking[0][1], ranking };
}

// --- Render ---------------------------------------------------------------
function mostrarEstadoF(texto, esError = false) {
  resultCardF.hidden = false;
  resultBodyF.hidden = true;
  statusF.hidden = false;
  statusF.textContent = texto;
  statusF.classList.toggle("error", esError);
}

function mostrarResultadoF(fusion, estados) {
  resultCardF.hidden = false;
  statusF.hidden = true;
  resultBodyF.hidden = false;

  emojiBigF.textContent = EMOJIS[fusion.emocion] || "🙂";
  emotionNameF.textContent = NOMBRES_ES_FUSION[fusion.emocion] || fusion.emocion;
  confidenceF.textContent = `Confianza: ${(fusion.confianza * 100).toFixed(1)}%`;

  // Chips: qué modalidades participaron (y con qué predijeron) y cuáles no.
  const iconos = { voz: "🎙️ Voz", rostro: "📷 Rostro", texto: "✍️ Texto" };
  modalidadesF.innerHTML = "";
  for (const estado of estados) {
    const chip = document.createElement("span");
    chip.className = "modalidad-chip" + (estado.ok ? "" : " off");
    const detalle = estado.ok
      ? `: ${estado.data.emocion_es || estado.data.emocion}`
      : ` (${estado.motivo})`;
    chip.textContent = (iconos[estado.modalidad] || estado.modalidad) + detalle;
    modalidadesF.appendChild(chip);
  }

  rankingF.innerHTML = "";
  for (const [emocion, prob] of fusion.ranking) {
    const pct = (prob * 100).toFixed(1);
    const row = document.createElement("div");
    row.className = "rank-row";
    row.innerHTML = `
      <span class="rank-label">${EMOJIS[emocion] || ""} ${emocion}</span>
      <span class="rank-track"><span class="rank-fill" style="width:${pct}%"></span></span>
      <span class="rank-pct">${pct}%</span>`;
    rankingF.appendChild(row);
  }
}
