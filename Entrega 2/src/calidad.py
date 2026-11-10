"""Verificación de calidad de las series crudas (rúbrica 7.1: datos consolidados y verificados).

Se usa en el notebook 01, antes de alinear: recorta cada serie al periodo de estudio, detecta
cambios diarios extremos y arma la tabla de calidad (Tabla 1 del informe, tab01_calidad.csv).
Ninguna función rellena, interpola ni corrige datos: solo los describe.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def recortar_periodo(df: pd.DataFrame, inicio: str | pd.Timestamp, fin: str | pd.Timestamp) -> pd.DataFrame:
    """Deja solo las fechas del periodo de estudio [inicio, fin].

    Parámetros
    ----------
    df : pd.DataFrame
        Serie con una columna ``fecha`` de tipo fecha.
    inicio, fin : str o pd.Timestamp
        Extremos del periodo; ambos se incluyen.

    Devuelve
    --------
    pd.DataFrame
        Las filas de `df` dentro del periodo, con el índice reiniciado (0, 1, …).
    """
    return df[(df["fecha"] >= pd.Timestamp(inicio)) & (df["fecha"] <= pd.Timestamp(fin))].reset_index(drop=True)


def cambios_extremos(df: pd.DataFrame, en_niveles: bool = False, umbral_sd: float = 5.0,
                     umbral_abs: float | None = None) -> pd.DataFrame:
    """Encuentra las filas cuyo cambio diario es extremo.

    Hay dos modos, porque los precios y las tasas cambian de forma distinta:
    precios (``en_niveles=False``): retorno logarítmico con |r| > `umbral_sd` desviaciones estándar;
    tasas (``en_niveles=True``): cambio en puntos porcentuales con |Δ| > `umbral_abs`.
    Los extremos no se eliminan: se revisan uno por uno para confirmar que son episodios reales.

    Parámetros
    ----------
    df : pd.DataFrame
        Serie con columnas ``fecha`` y ``valor``; las filas sin valor se descartan antes de calcular.
    en_niveles : bool, por defecto False
        False para precios (TRM, Brent, DXY); True para tasas (TPM, DFF).
    umbral_sd : float, por defecto 5.0
        Número de desviaciones estándar del retorno, calculada sobre toda la serie recibida.
        Solo se usa con ``en_niveles=False``.
    umbral_abs : float o None, por defecto None
        Cambio mínimo en puntos porcentuales. Obligatorio con ``en_niveles=True`` (en el proyecto,
        1.0 para la TPM y 0.5 para la DFF).

    Devuelve
    --------
    pd.DataFrame
        Columnas ``fecha``, ``valor`` y ``cambio`` (retorno o cambio en pp) de las filas extremas.

    Notas
    -----
    Con ``en_niveles=True`` y ``umbral_abs=None`` la comparación falla con TypeError; el proyecto
    nunca la llama así.
    """
    d = df.dropna(subset=["valor"]).copy()
    if en_niveles:
        d["cambio"] = d["valor"].diff()
        mask = d["cambio"].abs() > umbral_abs
    else:
        d["cambio"] = np.log(d["valor"]).diff()
        mask = d["cambio"].abs() > umbral_sd * d["cambio"].std()
    return d.loc[mask, ["fecha", "valor", "cambio"]]


def tabla_calidad(series: dict[str, pd.DataFrame], rangos: dict[str, tuple[float, float]],
                  extremos: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Resume la calidad de cada serie en una fila (Tabla 1 del informe).

    Parámetros
    ----------
    series : dict[str, pd.DataFrame]
        Nombre → serie con columnas ``fecha`` y ``valor``, ya recortada al periodo.
    rangos : dict[str, tuple[float, float]]
        Nombre → (mínimo, máximo) plausibles; se cuenta cuántos valores caen fuera.
    extremos : dict[str, pd.DataFrame]
        Nombre → resultado de `cambios_extremos` para esa serie.

    Devuelve
    --------
    pd.DataFrame
        Índice ``serie`` y columnas ``desde``, ``hasta``, ``observaciones`` (valores no vacíos),
        ``faltantes``, ``fechas_duplicadas``, ``minimo``, ``maximo``, ``fuera_de_rango``,
        ``mayor_hueco_dias`` (días calendario entre dos datos seguidos) y ``cambios_extremos``.
    """
    filas = []
    for nombre, df in series.items():
        v = df.dropna(subset=["valor"])
        huecos = v["fecha"].diff().dt.days
        lo, hi = rangos[nombre]
        filas.append({
            "serie": nombre,
            "desde": v["fecha"].min().date(),
            "hasta": v["fecha"].max().date(),
            "observaciones": len(v),
            "faltantes": int(df["valor"].isna().sum()),
            "fechas_duplicadas": int(df["fecha"].duplicated().sum()),
            "minimo": v["valor"].min(),
            "maximo": v["valor"].max(),
            "fuera_de_rango": int((~v["valor"].between(lo, hi)).sum()),
            "mayor_hueco_dias": int(huecos.max()),
            "cambios_extremos": len(extremos[nombre]),
        })
    return pd.DataFrame(filas).set_index("serie")
