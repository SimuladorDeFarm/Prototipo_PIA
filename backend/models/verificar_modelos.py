"""Comprueba si el checkpoint de cada módulo ya existe en disco.

Por ahora solo verifica; la descarga automática (Hugging Face Hub) se
añade en un paso posterior dentro de `modelo_disponible`. El nombre de
archivo de cada .pt es el mismo en disco que en su repo de Hugging Face.
"""

from pathlib import Path

MODELS_DIR = Path(__file__).parent

RUTAS_MODELOS = {
    "voz": MODELS_DIR / "voz" / "clasificador_voz_v4.pt",
    "rostro": MODELS_DIR / "rostro" / "mejor_modelo_v2.pt",
    "texto": MODELS_DIR / "texto" / "clasificador_texto_v4.pt",
}

REPOS_HF = {
    "voz": "SimuladorDeFarm/NeuroEmoInnovat_modelo_voz",
    "rostro": "SimuladorDeFarm/NeuroEmoInnovat_modelo_rostro",
    "texto": "SimuladorDeFarm/NeuroEmoInnovat_modelo_texto",
}


def modelo_disponible(nombre: str) -> bool:
    """True si el checkpoint del módulo `nombre` ya está en disco.

    Si falta, por ahora no hace nada más: el llamador debe tratar el
    módulo como no disponible.
    """
    return RUTAS_MODELOS[nombre].exists()


def verificar_todos() -> dict:
    """Comprueba los tres checkpoints y muestra el resultado en pantalla."""
    estado = {}
    for nombre in RUTAS_MODELOS:
        disponible = modelo_disponible(nombre)
        estado[nombre] = disponible
        marca = "OK" if disponible else "FALTA"
        print(f"[{marca}] {nombre}: {RUTAS_MODELOS[nombre]}")
    return estado


if __name__ == "__main__":
    verificar_todos()
