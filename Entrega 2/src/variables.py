"""Objetivo y variables predictoras (las de la Entrega 1, expresadas en retornos).

Fila t = una TRM publicada. Todas las variables miran solo hacia atrás (shift(k) con k >= 0);
el único shift(-1) del proyecto construye el objetivo.
Se usa en el notebook 02 (dataset.csv, Tabla 3 de transformaciones y prueba de perturbación).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .alineacion import alinear
from .config import VARS_TODAS

# Rezago k del retorno: r_trm_lk = ln(TRM_{t-k+1} / TRM_{t-k}); k = 1 es el movimiento más reciente
REZAGOS_TRM = {"r_trm_l1": 1, "r_trm_l5": 5, "r_trm_l10": 10}


def agregar_variables(tabla: pd.DataFrame) -> pd.DataFrame:
    """Agrega el objetivo y las 6 variables predictoras a la tabla alineada.

    Parámetros
    ----------
    tabla : pd.DataFrame
        Resultado de `alinear`, con las columnas ``trm``, ``brent``, ``dxy``, ``tpm`` y ``dff``.

    Devuelve
    --------
    pd.DataFrame
        Copia de `tabla` con estas columnas nuevas (las filas incompletas no se quitan aquí):

        - ``r_trm_l1``, ``r_trm_l5``, ``r_trm_l10``: ln(TRM_{t-k+1} / TRM_{t-k}) con k = 1, 5, 10.
        - ``r_brent``, ``r_dxy``: log-diferencia del último valor conocido entre dos TRM seguidas
          (es 0 si no llegó dato nuevo).
        - ``dif_tasas``: TPM − DFF en puntos porcentuales, en nivel (es una tasa acotada).
        - ``trm_sig``: TRM_{t+1}; solo sirve para el objetivo y para medir el error en pesos.
        - ``y``: objetivo, ln(TRM_{t+1} / TRM_t).

    Notas
    -----
    ``trm_sig`` y ``y`` son las únicas columnas que miran al futuro (el único ``shift(-1)``); nunca
    entran como variables predictoras.
    """
    t = tabla.copy()
    r_trm = np.log(t["trm"]).diff()                 # ln(TRM_t / TRM_{t-1})
    for nombre, k in REZAGOS_TRM.items():
        t[nombre] = r_trm.shift(k - 1)
    t["r_brent"] = np.log(t["brent"]).diff()        # retorno del último Brent conocido
    t["r_dxy"] = np.log(t["dxy"]).diff()            # retorno del último DXY conocido
    t["dif_tasas"] = t["tpm"] - t["dff"]            # diferencial en puntos porcentuales (nivel)
    t["trm_sig"] = t["trm"].shift(-1)               # TRM_{t+1}: solo para el objetivo y el nivel en pesos
    t["y"] = np.log(t["trm_sig"] / t["trm"])        # objetivo: ln(TRM_{t+1} / TRM_t)
    return t


def agregar_ventanas(datos: pd.DataFrame) -> pd.DataFrame:
    """Agrega las variables del experimento de ventanas de rezago (notebook 07).

    Amplía la información del pasado de la TRM de dos formas: cada uno de los movimientos de los
    últimos 20 días, y resúmenes de la ventana (tendencia acumulada, distancia a la media del último
    mes y volatilidad reciente). Todas miran solo hacia atrás: se calculan con ``shift(k ≥ 0)`` y
    ventanas móviles que terminan en t.

    Parámetros
    ----------
    datos : pd.DataFrame
        Filas consecutivas del conjunto de modelado (una por TRM publicada, sin huecos), ordenadas
        por fecha, con las columnas ``trm`` y ``r_trm_l1`` = ln(TRM_t / TRM_{t-1}).

    Devuelve
    --------
    pd.DataFrame
        Copia de `datos` con estas columnas nuevas (vacías al inicio, mientras no hay historia):

        - ``r_trm_l2`` … ``r_trm_l20``: movimiento de hace k días, con la misma fórmula que
          ``r_trm_l1``; ``r_trm_l5`` y ``r_trm_l10`` ya existen y se conservan sin cambios.
        - ``r_trm_acum5``, ``r_trm_acum10``, ``r_trm_acum20``, ``r_trm_acum60``: movimiento
          acumulado de los últimos k días (tendencia), ln(TRM_t / TRM_{t-k}).
        - ``dist_media20``: ln(TRM_t / media de las últimas 20 TRM); positiva si la TRM está por
          encima de su promedio del último mes.
        - ``vol_trm5``, ``vol_trm20``, ``vol_trm60``: desviación estándar de los últimos k movimientos
          (volatilidad reciente, el régimen del mercado).

    Notas
    -----
    Se necesitan 60 filas de historia para la ventana más larga, así que las primeras 59 filas quedan
    con vacíos. Si `datos` es solo el desarrollo, el test nunca interviene en el cálculo.
    """
    t = datos.copy()
    r = t["r_trm_l1"]                                # ln(TRM_t / TRM_{t-1}), conocido en t
    for k in range(2, 21):
        if f"r_trm_l{k}" not in t:                   # r_trm_l5 y r_trm_l10 ya vienen del notebook 02
            t[f"r_trm_l{k}"] = r.shift(k - 1)
    for k in (5, 10, 20, 60):
        t[f"r_trm_acum{k}"] = r.rolling(k).sum()     # tendencia de las últimas k TRM
    t["dist_media20"] = np.log(t["trm"] / t["trm"].rolling(20).mean())
    for k in (5, 20, 60):
        t[f"vol_trm{k}"] = r.rolling(k).std()        # volatilidad reciente
    return t


def construir_dataset(trm: pd.DataFrame, macros: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Construye el conjunto de modelado a partir de las series limpias.

    Encadena `alinear` → `agregar_variables` → quitar las filas incompletas. En el proyecto se
    pierden 11 filas: las 10 primeras (los rezagos y los retornos necesitan días anteriores) y la
    última (no tiene TRM del día siguiente).

    Parámetros
    ----------
    trm : pd.DataFrame
        TRM con columnas ``fecha`` y ``valor``, una fila por TRM publicada.
    macros : dict[str, pd.DataFrame]
        Nombre → serie macro con columnas ``fecha`` y ``valor`` (ver `alinear`).

    Devuelve
    --------
    pd.DataFrame
        Tabla alineada con las 6 variables y el objetivo, sin vacíos en ``VARS_TODAS`` ni en ``y``,
        con índice 0, 1, ….
    """
    t = agregar_variables(alinear(trm, macros))
    return t.dropna(subset=VARS_TODAS + ["y"]).reset_index(drop=True)


def prueba_perturbacion(trm: pd.DataFrame, macros: dict[str, pd.DataFrame], fecha_corte: str | pd.Timestamp,
                        semilla: int = 42) -> dict[str, str | int | float]:
    """Altera al azar TODOS los datos posteriores a fecha_corte y reconstruye el conjunto.

    Si no hay fuga, ninguna variable en filas con fecha <= fecha_corte puede cambiar.
    Control: el objetivo de la última fila antes del corte SÍ debe cambiar, porque usa la TRM siguiente.

    Parámetros
    ----------
    trm : pd.DataFrame
        TRM con columnas ``fecha`` y ``valor``.
    macros : dict[str, pd.DataFrame]
        Nombre → serie macro con columnas ``fecha`` y ``valor``.
    fecha_corte : str o pd.Timestamp
        Último día que no se altera; todo lo posterior se multiplica por un factor al azar.
    semilla : int, por defecto 42
        Semilla del generador, para que la prueba sea repetible.

    Devuelve
    --------
    dict[str, str | int | float]
        ``fecha_corte``, ``filas_antes_del_corte``, ``valores_de_variables_que_cambiaron`` (debe ser 0),
        ``objetivo_ultima_fila_original`` y ``objetivo_ultima_fila_alterado`` (deben ser distintos).
    """
    rng = np.random.default_rng(semilla)
    corte = pd.Timestamp(fecha_corte)

    def alterar(df: pd.DataFrame) -> pd.DataFrame:
        """Multiplica por un factor U(0,5; 1,5) cada valor posterior al corte.

        Parámetros
        ----------
        df : pd.DataFrame
            Serie con columnas ``fecha`` y ``valor``; no se modifica.

        Devuelve
        --------
        pd.DataFrame
            Copia con los valores posteriores a `fecha_corte` alterados y los anteriores intactos.
        """
        d = df.copy()
        futuro = d["fecha"] > corte
        d.loc[futuro, "valor"] = d.loc[futuro, "valor"] * rng.uniform(0.5, 1.5, futuro.sum())
        return d

    original = construir_dataset(trm, macros)
    alterado = construir_dataset(alterar(trm), {n: alterar(m) for n, m in macros.items()})
    antes_o = original[original["fecha"] <= corte].set_index("fecha")
    antes_a = alterado[alterado["fecha"] <= corte].set_index("fecha")
    cambios_vars = int((~np.isclose(antes_o[VARS_TODAS], antes_a[VARS_TODAS])).sum())
    ultima = antes_o.index.max()
    return {
        "fecha_corte": str(corte.date()),
        "filas_antes_del_corte": len(antes_o),
        "valores_de_variables_que_cambiaron": cambios_vars,
        "objetivo_ultima_fila_original": float(antes_o.loc[ultima, "y"]),
        "objetivo_ultima_fila_alterado": float(antes_a.loc[ultima, "y"]),
    }
