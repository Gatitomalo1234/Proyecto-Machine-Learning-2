"""Partición cronológica, cortes walk-forward y métricas (rúbricas 7.2 y 7.4).

La partición (notebook 02) separa desarrollo y test sin mezclar el orden del tiempo; los cortes
walk-forward (notebook 05) son el bucle externo de la validación anidada; las métricas son las del
contrato con la Entrega 1 y se usan igual en validación (notebook 05) y en test (notebook 06).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from .config import FRACCION_TEST, GAP, N_CORTES_EXTERNOS

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


def cortes_walk_forward(n: int, n_cortes: int = N_CORTES_EXTERNOS, gap: int = GAP) -> list[tuple[np.ndarray, np.ndarray]]:
    """Define los cortes del bucle externo: ventana expansiva, validación siempre posterior al entrenamiento.

    Parámetros
    ----------
    n : int
        Número de filas del conjunto de desarrollo, ordenado por fecha.
    n_cortes : int, por defecto `N_CORTES_EXTERNOS` (5)
        Número de cortes. Cada validación tiene n // (n_cortes + 1) filas (720 en el desarrollo).
    gap : int, por defecto `GAP` (5)
        Filas que se descartan entre el entrenamiento y la validación. Basta 1 (el objetivo de la
        última fila de entrenamiento usa la TRM del día siguiente); 5, una semana hábil, deja margen.

    Devuelve
    --------
    list[tuple[np.ndarray, np.ndarray]]
        Un par (índices de entrenamiento, índices de validación) por corte. Son posiciones (para
        ``.iloc``), no etiquetas del índice.
    """
    return list(TimeSeriesSplit(n_splits=n_cortes, gap=gap).split(np.arange(n)))


def tabla_cortes(fechas: ArrayLike, cortes: list[tuple[np.ndarray, np.ndarray]]) -> pd.DataFrame:
    """Traduce los cortes a periodos con fechas reales (tab05a_cortes.csv).

    Parámetros
    ----------
    fechas : array-like de fechas
        Fecha de cada fila del desarrollo, en el mismo orden que los índices de `cortes`.
    cortes : list[tuple[np.ndarray, np.ndarray]]
        Resultado de `cortes_walk_forward`.

    Devuelve
    --------
    pd.DataFrame
        Índice ``corte`` (1, 2, …) y columnas ``entrena_desde``, ``entrena_hasta``,
        ``filas_entrenamiento``, ``valida_desde``, ``valida_hasta`` y ``filas_validacion``.
    """
    fechas = pd.Series(pd.to_datetime(fechas)).reset_index(drop=True)
    return pd.DataFrame([{
        "corte": i, "entrena_desde": fechas[tr[0]].date(), "entrena_hasta": fechas[tr[-1]].date(),
        "filas_entrenamiento": len(tr), "valida_desde": fechas[va[0]].date(),
        "valida_hasta": fechas[va[-1]].date(), "filas_validacion": len(va),
    } for i, (tr, va) in enumerate(cortes, start=1)]).set_index("corte")


def particion_cronologica(dataset: pd.DataFrame, fraccion_test: float = FRACCION_TEST, purga: int = 1) -> np.ndarray:
    """Asigna cada fila a desarrollo, purga o test con un corte cronológico.

    El último `fraccion_test` de las filas es test; nunca se mezcla el orden. Se purgan las últimas
    `purga` filas del desarrollo: su objetivo usa la TRM siguiente, que ya pertenece al periodo de
    test (con horizonte de un día, basta una fila).

    Parámetros
    ----------
    dataset : pd.DataFrame
        Conjunto de modelado ordenado por fecha (solo se usa su número de filas).
    fraccion_test : float, por defecto `FRACCION_TEST` (0.20)
        Fracción final que va a test; el número de filas se redondea (1.080 de 5.401).
    purga : int, por defecto 1
        Filas del final del desarrollo que no se usan en ninguna muestra.

    Devuelve
    --------
    np.ndarray
        Arreglo de texto con ``"desarrollo"``, ``"purga"`` o ``"test"`` para cada fila, en orden.
    """
    n = len(dataset)
    n_test = int(round(n * fraccion_test))
    inicio_test = n - n_test
    muestra = np.array(["desarrollo"] * n, dtype=object)
    muestra[inicio_test - purga:inicio_test] = "purga"
    muestra[inicio_test:] = "test"
    return muestra


def metricas(y_ret: ArrayLike, yhat_ret: ArrayLike, trm_hoy: ArrayLike) -> dict[str, float]:
    """Calcula las métricas del contrato con la Entrega 1 para un conjunto de predicciones.

    El RMSE y el MAE se miden en pesos sobre la TRM reconstruida (TRM_{t+1} = TRM_t · e^y), porque
    ese es el error que le importa a quien usa la TRM. El R² se mide sobre retornos: sobre niveles
    daría ≈ 0,99 para cualquier modelo, incluso para copiar la TRM de hoy.

    Parámetros
    ----------
    y_ret : array-like de forma (n,)
        Retorno logarítmico real del día siguiente, ln(TRM_{t+1} / TRM_t).
    yhat_ret : array-like de forma (n,)
        Retorno predicho por el modelo, en la misma escala que `y_ret`.
    trm_hoy : array-like de forma (n,)
        TRM vigente en t, en pesos; se usa para pasar los retornos a pesos.

    Devuelve
    --------
    dict[str, float]
        ``rmse_pesos`` (métrica principal), ``mae_pesos`` y ``r2_retornos``.
    """
    from sklearn.metrics import r2_score
    y_ret, yhat_ret, trm_hoy = map(np.asarray, (y_ret, yhat_ret, trm_hoy))
    real, pred = trm_hoy * np.exp(y_ret), trm_hoy * np.exp(yhat_ret)
    return {"rmse_pesos": float(np.sqrt(np.mean((real - pred) ** 2))),
            "mae_pesos": float(np.mean(np.abs(real - pred))),
            "r2_retornos": float(r2_score(y_ret, yhat_ret))}
