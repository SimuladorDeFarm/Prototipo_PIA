#!/usr/bin/env bash
# Instala todo lo necesario para correr PIA desde cero tras un `git clone`:
# crea el entorno virtual del backend, instala las dependencias y descarga
# los checkpoints de los 3 modelos (voz, rostro, texto) desde Hugging Face.
#
# Uso:
#   ./install.sh
#
# Despues, para levantar el proyecto: ./iniciar.sh

set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$PROJECT_ROOT/backend"
VENV_DIR="$BACKEND_DIR/.venv"
PYTHON_VENV="$VENV_DIR/bin/python"

echo "== PIA - Instalacion =="
echo ""

if ! command -v python3 >/dev/null 2>&1; then
    echo "[ERROR] No se encontró python3 en el sistema. Instala Python 3.11+ y vuelve a correr este script."
    exit 1
fi
echo "Python detectado: $(python3 --version)"

if [ -x "$PYTHON_VENV" ]; then
    echo "[OK] El entorno virtual ya existe en $VENV_DIR (no se vuelve a crear)."
else
    echo "Creando entorno virtual en $VENV_DIR ..."
    if ! python3 -m venv "$VENV_DIR"; then
        echo "[ERROR] No se pudo crear el entorno virtual."
        exit 1
    fi
fi

if [ ! -x "$PYTHON_VENV" ]; then
    echo "[ERROR] El entorno virtual no quedó bien formado en $VENV_DIR."
    exit 1
fi

echo ""
echo "Instalando dependencias (backend/requirements.txt) ..."
"$PYTHON_VENV" -m pip install --upgrade pip
if ! "$PYTHON_VENV" -m pip install -r "$BACKEND_DIR/requirements.txt"; then
    echo "[ERROR] Falló la instalación de dependencias. Revisa el error de arriba y vuelve a correr ./install.sh."
    exit 1
fi
echo "[OK] Dependencias instaladas."

echo ""
echo "Descargando modelos (voz, rostro, texto) desde Hugging Face Hub ..."
(cd "$BACKEND_DIR" && "$PYTHON_VENV" -m models.descargar_modelos)

echo ""
echo "== Instalacion terminada =="
echo "Si algún modelo no se pudo descargar, revisa el aviso de arriba: te da el link"
echo "para bajarlo a mano (ver tambien README -> Modelos)."
echo ""
echo "Para levantar el backend y el frontend juntos: ./iniciar.sh"
