"""Prepara las figuras del informe final: quita el título interno ("Figura N. …") y recorta la captura de la app.

En el informe el título lo pone LaTeX, con su propia numeración; las figuras originales de
reports/figuras/ no se modifican. Uso: python reports/informe_final/preparar_figuras.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

AQUI = Path(__file__).resolve().parent
ORIGEN, CAPTURAS, DESTINO = AQUI.parent / "figuras", AQUI.parent / "capturas_app", AQUI / "figuras"
ARCHIVOS = {"fig_articulo_vs_trm": "fig11_articulo_vs_trm", "fig_interpretacion": "fig12_interpretacion_global",
            "fig_rango_adaptativo": "fig15_rango_adaptativo"}


def sin_titulo(imagen: Image.Image) -> Image.Image:
    """Recorta el primer bloque de texto de una figura (su título interno) y el espacio que lo sigue.

    Parámetros
    ----------
    imagen : PIL.Image.Image
        Figura con un título en la parte superior.

    Devuelve
    --------
    PIL.Image.Image
        La figura desde la primera fila con contenido después del título.
    """
    gris = np.asarray(imagen.convert("L"))
    con_tinta = (gris < 200).any(axis=1)
    fila = int(np.argmax(con_tinta))                       # inicio del título
    while fila < len(con_tinta) and con_tinta[fila]:       # fin del título
        fila += 1
    while fila < len(con_tinta) and not con_tinta[fila]:   # espacio en blanco siguiente
        fila += 1
    return imagen.crop((0, fila, imagen.width, imagen.height))


def main() -> None:
    """Escribe las figuras del informe en reports/informe_final/figuras/."""
    DESTINO.mkdir(exist_ok=True)
    for nombre, archivo in ARCHIVOS.items():
        sin_titulo(Image.open(ORIGEN / f"{archivo}.png").convert("RGB")).save(DESTINO / f"{nombre}.png")
    app = Image.open(CAPTURAS / "1_manana.png").convert("RGB")
    app.crop((int(app.width * 0.12), int(app.height * 0.04), int(app.width * 0.88), int(app.height * 0.74))).save(DESTINO / "fig_app.png")
    print("Figuras listas en", DESTINO)


if __name__ == "__main__":
    main()
