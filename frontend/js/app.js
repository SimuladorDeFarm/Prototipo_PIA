var API = "http://localhost:8000";

var EMOJIS = {
  others:   "😐",
  joy:      "😄",
  sadness:  "😢",
  anger:    "😠",
  fear:     "😨",
  disgust:  "🤢",
  surprise: "😲",
};

window.onload = function () {
  var textoInput   = document.getElementById("textoInput");
  var charCount    = document.getElementById("charCount");
  var predictBtn   = document.getElementById("predictBtn");
  var clearBtn     = document.getElementById("clearBtn");
  var resultCard   = document.getElementById("resultCard");
  var statusEl     = document.getElementById("status");
  var resultBody   = document.getElementById("resultBody");
  var emojiBig     = document.getElementById("emojiBig");
  var emotionName  = document.getElementById("emotionName");
  var confidenceEl = document.getElementById("confidenceEl");
  var rankingEl    = document.getElementById("ranking");

  textoInput.oninput = function () {
    var len = textoInput.value.length;
    charCount.textContent = len;
    predictBtn.disabled = (len === 0);
  };

  clearBtn.onclick = function () {
    textoInput.value = "";
    charCount.textContent = "0";
    predictBtn.disabled = true;
    resultCard.style.display = "none";
  };

  predictBtn.onclick = function () {
    var texto = textoInput.value.trim();
    if (texto.length > 0) enviar(texto);
  };

  textoInput.onkeydown = function (e) {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
      e.preventDefault();
      var texto = textoInput.value.trim();
      if (texto.length > 0) enviar(texto);
    }
  };

  function enviar(texto) {
    mostrarEstado("Analizando texto…", false);
    predictBtn.disabled = true;
    clearBtn.disabled = true;

    fetch(API + "/predecir", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ texto: texto }),
    })
    .then(function (res) {
      return res.json().then(function (data) {
        if (!res.ok) throw new Error(data.detail || "Error en el servidor");
        return data;
      });
    })
    .then(function (data) {
      mostrarResultado(data);
    })
    .catch(function (err) {
      mostrarEstado("⚠️ " + err.message, true);
    })
    .finally(function () {
      predictBtn.disabled = (textoInput.value.trim().length === 0);
      clearBtn.disabled = false;
    });
  }

  function mostrarEstado(texto, esError) {
    resultCard.style.display = "block";
    resultBody.style.display = "none";
    statusEl.style.display = "block";
    statusEl.textContent = texto;
    statusEl.className = "status" + (esError ? " error" : "");
  }

  function mostrarResultado(data) {
    resultCard.style.display = "block";
    statusEl.style.display = "none";
    resultBody.style.display = "block";

    emojiBig.textContent = EMOJIS[data.emocion] || "🙂";
    emotionName.textContent = data.emocion_es || data.emocion;
    confidenceEl.textContent = "Confianza: " + (data.confianza * 100).toFixed(1) + "%";

    rankingEl.innerHTML = "";
    for (var i = 0; i < data.ranking.length; i++) {
      var emocion = data.ranking[i][0];
      var prob    = data.ranking[i][1];
      var pct     = (prob * 100).toFixed(1);
      var row     = document.createElement("div");
      row.className = "rank-row";
      row.innerHTML =
        '<span class="rank-label">' + (EMOJIS[emocion] || "") + " " + emocion + "</span>" +
        '<span class="rank-track"><span class="rank-fill" style="width:' + pct + '%"></span></span>' +
        '<span class="rank-pct">' + pct + "%</span>";
      rankingEl.appendChild(row);
    }
  }
};