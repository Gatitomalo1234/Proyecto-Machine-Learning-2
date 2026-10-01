"""Objetivo y variables predictoras (las de la Entrega 1, expresadas en retornos).

Fila t = una TRM publicada. Todas las variables miran solo hacia atrás (shift(k) con k >= 0);
el único shift(-1) del proyecto construye el objetivo.
"""
import numpy as np
import pandas as pd

from .alineacion import alinear
from .config import VARS_TODAS

# Rezago k del retorno: r_trm_lk = ln(TRM_{t-k+1} / TRM_{t-k}); k = 1 es el movimiento más reciente
REZAGOS_TRM = {"r_trm_l1": 1, "r_trm_l5": 5, "r_trm_l10": 10}


def agregar_variables(tabla):
    """Agrega el objetivo y las 6 variables a la tabla alineada."""
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


def construir_dataset(trm, macros):
    """Series limpias → tabla alineada con variables y objetivo, sin filas incompletas."""
    t = agregar_variables(alinear(trm, macros))
    return t.dropna(subset=VARS_TODAS + ["y"]).reset_index(drop=True)


def prueba_perturbacion(trm, macros, fecha_corte, semilla=42):
    """Altera al azar TODOS los datos posteriores a fecha_corte y reconstruye el conjunto.

    Si no hay fuga, ninguna variable en filas con fecha <= fecha_corte puede cambiar.
    Control: el objetivo de la última fila antes del corte SÍ debe cambiar, porque usa la TRM siguiente.
    """
    rng = np.random.default_rng(semilla)
    corte = pd.Timestamp(fecha_corte)

    def alterar(df):
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
