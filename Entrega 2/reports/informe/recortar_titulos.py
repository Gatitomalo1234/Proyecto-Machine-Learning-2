"""Quita el título interno ("Figura N. ...") de las figuras: en el informe lo pone LaTeX.

Detecta el primer bloque de texto de la imagen (el título) y corta en el espacio en blanco que lo sigue.
Uso: python reports/informe/recortar_titulos.py
"""
from pathlib import Path

import numpy as np
from PIL import Image

AQUI = Path(__file__).resolve().parent
ORIGEN, DESTINO = AQUI.parent / "figuras", AQUI / "figuras"
DESTINO.mkdir(exist_ok=True)

for nombre in ["fig03_techo_extrapolacion", "fig05_shap"]:
    im = Image.open(ORIGEN / f"{nombre}.png").convert("RGB")
    gris = np.asarray(im.convert("L"))
    con_tinta = (gris < 200).any(axis=1)
    fila = int(np.argmax(con_tinta))                 # inicio del título
    while fila < len(con_tinta) and con_tinta[fila]:  # fin del título
        fila += 1
    while fila < len(con_tinta) and not con_tinta[fila]:  # espacio en blanco siguiente
        fila += 1
    im.crop((0, fila, im.width, im.height)).save(DESTINO / f"{nombre}.png")
    print(nombre, "recortada desde la fila", fila)
