"""Verificación de calidad de las series crudas (rúbrica 7.1: datos consolidados y verificados)."""
import numpy as np
import pandas as pd


def recortar_periodo(df, inicio, fin):
    """Deja solo las fechas del periodo de estudio [inicio, fin]."""
    return df[(df["fecha"] >= pd.Timestamp(inicio)) & (df["fecha"] <= pd.Timestamp(fin))].reset_index(drop=True)


def cambios_extremos(df, en_niveles=False, umbral_sd=5.0, umbral_abs=None):
    """Filas cuyo cambio diario es extremo.

    Precios (en_niveles=False): retorno logarítmico con |r| > umbral_sd desviaciones estándar.
    Tasas (en_niveles=True): cambio en puntos porcentuales con |Δ| > umbral_abs.
    """
    d = df.dropna(subset=["valor"]).copy()
    if en_niveles:
        d["cambio"] = d["valor"].diff()
        mask = d["cambio"].abs() > umbral_abs
    else:
        d["cambio"] = np.log(d["valor"]).diff()
        mask = d["cambio"].abs() > umbral_sd * d["cambio"].std()
    return d.loc[mask, ["fecha", "valor", "cambio"]]


def tabla_calidad(series, rangos, extremos):
    """Una fila por serie: periodo, observaciones, faltantes, duplicados, rango, mayor hueco y extremos.

    series:   {nombre: DataFrame(fecha, valor)} ya recortado al periodo.
    rangos:   {nombre: (mín, máx)} plausibles; se verifica que todo valor caiga dentro.
    extremos: {nombre: DataFrame de cambios extremos}.
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
