#!/usr/bin/env bash
# Levanta el backend (FastAPI/uvicorn) y el frontend (http.server) juntos,
# en una sola terminal. Muestra el estado de cada uno y el link del
# frontend. Con Ctrl+C detiene ambos de forma segura y libera los puertos.

set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$PROJECT_ROOT/backend"
FRONTEND_DIR="$PROJECT_ROOT/frontend"
PYTHON_BACKEND="$BACKEND_DIR/.venv/bin/python"

BACKEND_HOST="0.0.0.0"
BACKEND_PORT=8000
FRONTEND_PORT=5501

if [ ! -x "$PYTHON_BACKEND" ]; then
    echo "No se encontró el entorno virtual del backend en: $PYTHON_BACKEND"
    echo "Crea el venv e instala dependencias primero (ver README -> Instalacion)."
    exit 1
fi

BACKEND_PID=""
FRONTEND_PID=""

liberar_puerto() {
    local puerto="$1"
    command -v lsof >/dev/null 2>&1 || return 0
    local pids
    pids=$(lsof -ti tcp:"$puerto" 2>/dev/null || true)
    if [ -n "$pids" ]; then
        kill -9 $pids 2>/dev/null || true
    fi
}

detener() {
    echo ""
    echo "Deteniendo backend y frontend..."
    [ -n "$BACKEND_PID" ] && kill -TERM "$BACKEND_PID" 2>/dev/null
    [ -n "$FRONTEND_PID" ] && kill -TERM "$FRONTEND_PID" 2>/dev/null
    sleep 1
    # Por si uvicorn --reload dejó algún proceso hijo vivo en el puerto.
    liberar_puerto "$BACKEND_PORT"
    liberar_puerto "$FRONTEND_PORT"
    echo "Backend y frontend detenidos. Puertos $BACKEND_PORT y $FRONTEND_PORT liberados."
    exit 0
}

trap detener INT TERM

echo "== PIA - Prediccion de Emociones =="
echo ""

# Por si quedó algo colgado de una corrida anterior sin cerrar bien.
liberar_puerto "$BACKEND_PORT"
liberar_puerto "$FRONTEND_PORT"

echo "Iniciando backend en http://localhost:${BACKEND_PORT} ..."
(cd "$BACKEND_DIR" && exec "$PYTHON_BACKEND" -m uvicorn main:app --reload --host "$BACKEND_HOST" --port "$BACKEND_PORT") &
BACKEND_PID=$!

echo "Iniciando frontend en http://localhost:${FRONTEND_PORT} ..."
(cd "$FRONTEND_DIR" && exec python3 -m http.server "$FRONTEND_PORT") &
FRONTEND_PID=$!

sleep 2

if kill -0 "$BACKEND_PID" 2>/dev/null; then
    echo "[OK]    Backend activo   (PID $BACKEND_PID) -> http://localhost:${BACKEND_PORT}"
else
    echo "[ERROR] El backend no pudo iniciar (revisa el error arriba)."
fi

if kill -0 "$FRONTEND_PID" 2>/dev/null; then
    echo "[OK]    Frontend activo  (PID $FRONTEND_PID) -> http://localhost:${FRONTEND_PORT}"
else
    echo "[ERROR] El frontend no pudo iniciar (revisa el error arriba)."
fi

echo ""
echo ">>> Abre http://localhost:${FRONTEND_PORT} en tu navegador <<<"
echo "Presiona Ctrl+C para detener todo y liberar los puertos."
echo ""

# Si uno de los dos procesos muere solo (crash), se detiene el otro también.
wait -n "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null
detener
