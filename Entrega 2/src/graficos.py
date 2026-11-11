"""Estilo común y guardado de figuras y tablas numeradas (mismo número que en el informe).

Todos los notebooks llaman `estilo` al empezar y guardan sus salidas con `guardar_figura` y
`guardar_tabla`, así cada figura y cada tabla tiene un nombre estable (figNN_…, tabNN_…) que el
informe y el README pueden citar.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

from .config import FIGURAS, TABLAS

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

COLORES = ["#1f4e79", "#c0504d", "#4f8a3a", "#8064a2", "#d98c1f", "#2c8c99", "#7f7f7f"]   # paleta de todas las figuras

# Letra para las figuras de tres paneles que el informe final imprime a todo el ancho (≈ 7 pt impresas).
LETRA_GRANDE = {"font.size": 15, "axes.titlesize": 15, "axes.labelsize": 14, "xtick.labelsize": 13,
                "ytick.labelsize": 13, "legend.fontsize": 12}


def coma_decimal(eje: Axes, ejes: str = "xy") -> None:
    """Escribe los números de los ejes con coma decimal y signo menos tipográfico (formato del informe).

    Parámetros
    ----------
    eje : matplotlib.axes.Axes
        Eje que se formatea.
    ejes : str, por defecto "xy"
        Qué ejes numéricos cambiar: "x", "y" o "xy" (no usar en un eje de categorías).
    """
    from matplotlib.ticker import FuncFormatter
    formato = FuncFormatter(lambda v, _: f"{v:g}".replace("-", "−").replace(".", ","))
    if "x" in ejes:
        eje.xaxis.set_major_formatter(formato)
    if "y" in ejes:
        eje.yaxis.set_major_formatter(formato)


def estilo() -> None:
    """Fija el estilo de matplotlib para que todas las figuras del proyecto se vean iguales.

    Define la resolución (110 ppp en pantalla, 150 al guardar), el tamaño de letra, la cuadrícula
    tenue, los títulos en negrita y quita los bordes superior y derecho.
    """
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 150, "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "axes.titleweight": "bold",
    })


def guardar_figura(fig: Figure, nombre: str) -> Path:
    """Guarda una figura en reports/figuras/<nombre>.png.

    Parámetros
    ----------
    fig : matplotlib.figure.Figure
        Figura que se guarda (con ``bbox_inches="tight"`` para recortar márgenes).
    nombre : str
        Nombre sin extensión, con la convención ``figNN_descripcion`` (p. ej. "fig01_series").

    Devuelve
    --------
    Path
        Ruta del archivo escrito.
    """
    fig.savefig(FIGURAS / f"{nombre}.png", bbox_inches="tight")
    return FIGURAS / f"{nombre}.png"


def guardar_tabla(df: pd.DataFrame, nombre: str, index: bool = True) -> Path:
    """Guarda una tabla en reports/tablas/<nombre>.csv.

    Parámetros
    ----------
    df : pd.DataFrame
        Tabla que se guarda.
    nombre : str
        Nombre sin extensión, con la convención ``tabNN_descripcion`` (p. ej. "tab01_calidad").
    index : bool, por defecto True
        Si se escribe el índice como primera columna.

    Devuelve
    --------
    Path
        Ruta del archivo escrito.
    """
    df.to_csv(TABLAS / f"{nombre}.csv", index=index)
    return TABLAS / f"{nombre}.csv"
