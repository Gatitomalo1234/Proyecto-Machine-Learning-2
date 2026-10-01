"""Figura 1 del informe: flujo del proyecto con texto legible al imprimir.

Versión para el informe de la fig02_pipeline del notebook 05: mismas 8 etapas, en palabras sencillas
y con letra grande. Uso: python reports/informe/figura_flujo.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

SALIDA = Path(__file__).resolve().parent / "figuras" / "fig_flujo.png"

ETAPAS = [
    ("1. Descarga", "Fuentes oficiales;\ncopia fija con huella MD5"),
    ("2. Alineación", "Último dato anterior\nal día; Fed, un día más"),
    ("3. Variables", "Solo datos pasados;\nel objetivo es el futuro"),
    ("4. Partición 80/20", "Cronológica; un día\npurgado; test cerrado"),
    ("5. Validación", "5 cortes; siempre valida\ndespués de entrenar"),
    ("6. Búsqueda", "Hiperparámetros solo con\nel entrenamiento del corte"),
    ("7. Pipeline", "Escalado ajustado solo\ncon el entrenamiento"),
    ("8. Predicción", "Retorno → precio;\nerror en pesos"),
]
COLOR = {0: ("#e8eef6", "#4a6fa5"), 1: ("#f6ebe6", "#b5654a")}   # fila 1: datos; fila 2: modelado

fig, ax = plt.subplots(figsize=(12, 4.2))
ax.set_xlim(0, 4)
ax.set_ylim(0, 2)
ax.axis("off")
ancho, alto = 0.86, 0.78
for i, (titulo, texto) in enumerate(ETAPAS):
    fila, col = divmod(i, 4)
    x, y = col + (1 - ancho) / 2, 1.12 - fila * 1.04
    fondo, borde = COLOR[fila]
    ax.add_patch(FancyBboxPatch((x, y), ancho, alto, boxstyle="round,pad=0.01,rounding_size=0.04",
                                fc=fondo, ec=borde, lw=1.6))
    ax.text(x + ancho / 2, y + alto - 0.17, titulo, ha="center", va="center", fontsize=12, weight="bold")
    ax.text(x + ancho / 2, y + 0.29, texto, ha="center", va="center", fontsize=10.5, linespacing=1.3)
    if col < 3:
        ax.annotate("", xy=(col + 1 + (1 - ancho) / 2, y + alto / 2), xytext=(x + ancho, y + alto / 2),
                    arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.4))
# de la etapa 4 a la 5
ax.annotate("", xy=(0.5, 0.08 + alto), xytext=(3.5, 1.12),
            arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.4, connectionstyle="arc3,rad=-0.08"))
ax.text(-0.02, 1.51, "Datos", rotation=90, va="center", ha="center", fontsize=12, color=COLOR[0][1], weight="bold")
ax.text(-0.02, 0.47, "Modelado", rotation=90, va="center", ha="center", fontsize=12, color=COLOR[1][1], weight="bold")
fig.savefig(SALIDA, dpi=200, bbox_inches="tight")
print("Guardada", SALIDA)
