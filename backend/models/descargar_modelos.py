"""Descarga los checkpoints de voz, rostro y texto desde Hugging Face Hub.

Usa las rutas locales y los repos definidos en `verificar_modelos.py`.
Si un checkpoint ya está en disco, no lo vuelve a descargar.

Uso:
    python -m models.descargar_modelos
"""

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import enable_progress_bars, logging as hf_logging

from models.verificar_modelos import REPOS_HF, RUTAS_MODELOS, modelo_disponible

# Fuerza la barra de progreso y el log de huggingface_hub a pantalla, sin
# depender de variables de entorno (HF_HUB_DISABLE_PROGRESS_BARS, etc.):
# así se ve el avance (% y velocidad) mientras baja cada checkpoint pesado.
enable_progress_bars()
hf_logging.set_verbosity_info()


def descargar_modelo(nombre: str) -> bool:
    """Descarga el checkpoint de `nombre` si falta.

    Devuelve True si al terminar el archivo está disponible en disco (ya
    estaba o se bajó bien). Si la descarga falla (sin red, repo no
    encontrado, etc.) no lanza la excepción hacia arriba: avisa y
    devuelve False para que el programa pueda seguir sin ese módulo.
    """
    ruta = RUTAS_MODELOS[nombre]

    if modelo_disponible(nombre):
        print(f"[OK] {nombre}: ya está en {ruta}")
        return True

    print(f"[descargando] {nombre} desde {REPOS_HF[nombre]} -> {ruta}")
    try:
        hf_hub_download(
            repo_id=REPOS_HF[nombre],
            filename=ruta.name,
            local_dir=ruta.parent,
        )
    except Exception as e:
        url_manual = f"https://huggingface.co/{REPOS_HF[nombre]}/resolve/main/{ruta.name}"
        print(f"[ERROR] No se pudo descargar el modelo de {nombre}: {e}")
        print(f"        Descárgalo manualmente desde: {url_manual}")
        print(f"        y colócalo en: {ruta}")
        print("        (ver README -> Modelos para la guía completa).")
        return False

    print(f"[listo] {nombre}: {ruta}")
    return True


def descargar_todos() -> dict:
    """Intenta descargar los tres modelos sin detenerse si alguno falla."""
    resultados = {nombre: descargar_modelo(nombre) for nombre in RUTAS_MODELOS}

    fallidos = [nombre for nombre, ok in resultados.items() if not ok]
    if fallidos:
        print("")
        print(f"[aviso] No se pudieron descargar automáticamente: {', '.join(fallidos)}.")
        print("        El programa sigue funcionando; esos módulos quedarán deshabilitados")
        print("        hasta que coloques el archivo manualmente.")

    return resultados


if __name__ == "__main__":
    descargar_todos()
