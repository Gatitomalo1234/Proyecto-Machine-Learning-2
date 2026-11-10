"""Alineación temporal sin fuga de información (rúbrica 7.2).

La TRM es la columna vertebral: una fila por TRM publicada, con fecha t = inicio de su vigencia.
Regla anti-fuga: en la fila t solo entra información disponible ANTES del inicio del día t.
La TRM_{t+1} (el objetivo) se forma con las operaciones del día t, así que cualquier dato
conocido antes de que empiece t es información legítima y ninguno posterior lo es.
Se usa en el notebook 02 (Tabla 2 de disponibilidad y tab02b_relleno.csv).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Iterable

# Días que tarda en publicarse el dato del día d (se suma a su fecha para saber cuándo se conoce)
REZAGO_PUBLICACION = {
    "brent": 0,  # cierre del día d, conocido al final de d
    "dxy": 0,    # cierre del día d, conocido al final de d
    "tpm": 0,    # el Banrep anuncia la tasa antes de que entre en vigencia
    "dff": 1,    # la Fed de Nueva York publica la DFF del día d la mañana de d+1
}


def alinear(trm: pd.DataFrame, macros: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Une a cada TRM publicada el último valor de cada variable macro conocido antes de su fecha.

    Usa ``pd.merge_asof(direction="backward", allow_exact_matches=False)`` sobre la fecha en que el
    dato queda disponible (fecha + rezago de publicación). Nunca interpola: si un día no hay dato
    (festivo), se usa el último valor conocido, que es lo que vería un analista ese día.

    Parámetros
    ----------
    trm : pd.DataFrame
        Columnas ``fecha`` y ``valor``, con una fila por TRM publicada (fecha = inicio de la vigencia).
        Las demás columnas que traiga se conservan.
    macros : dict[str, pd.DataFrame]
        Nombre → serie con columnas ``fecha`` y ``valor``. Cada nombre debe estar en
        `REZAGO_PUBLICACION` ("brent", "dxy", "tpm", "dff"). Las filas sin valor se descartan.

    Devuelve
    --------
    pd.DataFrame
        Una fila por TRM, ordenada por fecha y con índice 0, 1, …: ``fecha``, ``trm`` (la columna
        ``valor`` renombrada) y, por cada variable macro, tres columnas:
        ``<nombre>`` (valor usado), ``fecha_<nombre>`` (fecha real de ese dato, para auditar) y
        ``antig_<nombre>`` (días entre ese dato y la fecha de la TRM).

    Notas
    -----
    `verificar_sin_fuga` comprueba después que ningún dato usado estuviera disponible en t o más tarde.
    """
    base = trm.rename(columns={"valor": "trm"}).sort_values("fecha").reset_index(drop=True)
    for nombre, df in macros.items():
        d = df.dropna(subset=["valor"]).sort_values("fecha").copy()
        d["disponible"] = d["fecha"] + pd.Timedelta(days=REZAGO_PUBLICACION[nombre])
        d = d.rename(columns={"valor": nombre, "fecha": f"fecha_{nombre}"})
        base = pd.merge_asof(base, d[["disponible", nombre, f"fecha_{nombre}"]],
                             left_on="fecha", right_on="disponible",
                             direction="backward", allow_exact_matches=False).drop(columns="disponible")
        base[f"antig_{nombre}"] = (base["fecha"] - base[f"fecha_{nombre}"]).dt.days
    return base


def verificar_sin_fuga(tabla: pd.DataFrame, macros: Iterable[str]) -> bool:
    """Comprueba que ninguna variable de la tabla alineada use un dato no disponible antes de t.

    Son pruebas automáticas: si alguna falla, la ejecución se detiene. Comprueba que las fechas
    estén en orden creciente y sin duplicados, y que, para cada variable macro, la fecha de su dato
    más el rezago de publicación sea estrictamente anterior a la fecha de la TRM.

    Parámetros
    ----------
    tabla : pd.DataFrame
        Resultado de `alinear`.
    macros : Iterable[str]
        Nombres de las variables macro que se verifican (sirve el mismo dict que se pasó a `alinear`).

    Devuelve
    --------
    bool
        True si todas las pruebas pasan.

    Lanza
    -----
    AssertionError
        Si las fechas no están en orden o están repetidas, o si una variable usa un dato que se
        conoció en t o después; el mensaje dice cuál.
    """
    assert tabla["fecha"].is_monotonic_increasing and not tabla["fecha"].duplicated().any()
    for nombre in macros:
        ok = tabla[f"fecha_{nombre}"].notna()
        disponible = tabla.loc[ok, f"fecha_{nombre}"] + pd.Timedelta(days=REZAGO_PUBLICACION[nombre])
        assert (disponible < tabla.loc[ok, "fecha"]).all(), f"{nombre}: usa un dato no disponible antes de t"
    return True


def tabla_relleno(tabla: pd.DataFrame, macros: Iterable[str], umbral_dias: int = 5) -> pd.DataFrame:
    """Mide cuánto se apoya cada variable en el último valor conocido (festivos, cierres de mercado).

    Parámetros
    ----------
    tabla : pd.DataFrame
        Resultado de `alinear`.
    macros : Iterable[str]
        Nombres de las variables macro que se resumen.
    umbral_dias : int, por defecto 5
        Antigüedad a partir de la cual un dato se considera viejo (una semana hábil).

    Devuelve
    --------
    pd.DataFrame
        Índice ``variable`` y columnas ``rezago_publicacion_dias``, ``filas_con_dato_repetido``
        (filas que reusan el mismo dato que la fila anterior), ``pct_repetido``,
        ``antiguedad_mediana_dias``, ``antiguedad_p95_dias``, ``antiguedad_max_dias`` y
        ``filas_con_dato_de_mas_de_<umbral_dias>_dias``.
    """
    filas = []
    for nombre in macros:
        a = tabla[f"antig_{nombre}"].dropna()
        repetido = tabla[f"fecha_{nombre}"].eq(tabla[f"fecha_{nombre}"].shift())
        filas.append({
            "variable": nombre,
            "rezago_publicacion_dias": REZAGO_PUBLICACION[nombre],
            "filas_con_dato_repetido": int(repetido.sum()),
            "pct_repetido": 100 * repetido.mean(),
            "antiguedad_mediana_dias": a.median(),
            "antiguedad_p95_dias": a.quantile(0.95),
            "antiguedad_max_dias": a.max(),
            f"filas_con_dato_de_mas_de_{umbral_dias}_dias": int((a > umbral_dias).sum()),
        })
    return pd.DataFrame(filas).set_index("variable")
