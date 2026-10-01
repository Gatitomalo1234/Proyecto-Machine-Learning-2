"""Estilo común y guardado de figuras y tablas numeradas (mismo número que en el informe)."""
import matplotlib.pyplot as plt

from .config import FIGURAS, TABLAS

COLORES = ["#1f4e79", "#c0504d", "#4f8a3a", "#8064a2", "#d98c1f", "#2c8c99", "#7f7f7f"]


def estilo():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 150, "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "axes.titleweight": "bold",
    })


def guardar_figura(fig, nombre):
    """Guarda en reports/figuras/<nombre>.png (p. ej. 'fig01_series')."""
    fig.savefig(FIGURAS / f"{nombre}.png", bbox_inches="tight")
    return FIGURAS / f"{nombre}.png"


def guardar_tabla(df, nombre, index=True):
    """Guarda en reports/tablas/<nombre>.csv (p. ej. 'tab01_calidad')."""
    df.to_csv(TABLAS / f"{nombre}.csv", index=index)
    return TABLAS / f"{nombre}.csv"
