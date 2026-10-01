"""Partición cronológica, cortes walk-forward y (en el notebook 05) validación anidada y métricas."""
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from .config import FRACCION_TEST, GAP, N_CORTES_EXTERNOS


def cortes_walk_forward(n, n_cortes=N_CORTES_EXTERNOS, gap=GAP):
    """Cortes del bucle externo: ventana expansiva, validación siempre posterior al entrenamiento."""
    return list(TimeSeriesSplit(n_splits=n_cortes, gap=gap).split(np.arange(n)))


def tabla_cortes(fechas, cortes):
    """Periodo de entrenamiento y de validación de cada corte (fechas reales)."""
    fechas = pd.Series(pd.to_datetime(fechas)).reset_index(drop=True)
    return pd.DataFrame([{
        "corte": i, "entrena_desde": fechas[tr[0]].date(), "entrena_hasta": fechas[tr[-1]].date(),
        "filas_entrenamiento": len(tr), "valida_desde": fechas[va[0]].date(),
        "valida_hasta": fechas[va[-1]].date(), "filas_validacion": len(va),
    } for i, (tr, va) in enumerate(cortes, start=1)]).set_index("corte")


def particion_cronologica(dataset, fraccion_test=FRACCION_TEST, purga=1):
    """Corte cronológico: el último `fraccion_test` es test; nunca se mezcla el orden.

    Se purgan las últimas `purga` filas del desarrollo: su objetivo usa la TRM siguiente, que ya
    pertenece al periodo de test (con h = 1, basta una fila).
    Devuelve un arreglo con 'desarrollo', 'purga' o 'test' por fila.
    """
    n = len(dataset)
    n_test = int(round(n * fraccion_test))
    inicio_test = n - n_test
    muestra = np.array(["desarrollo"] * n, dtype=object)
    muestra[inicio_test - purga:inicio_test] = "purga"
    muestra[inicio_test:] = "test"
    return muestra


def metricas(y_ret, yhat_ret, trm_hoy):
    """Métricas del contrato con la Entrega 1.

    RMSE y MAE en PESOS sobre el nivel reconstruido (TRM_{t+1} = TRM_t · e^y) y R² sobre RETORNOS
    (proporción de la variación diaria explicada; sobre niveles daría ≈ 0,99 para cualquier modelo).
    """
    from sklearn.metrics import r2_score
    y_ret, yhat_ret, trm_hoy = map(np.asarray, (y_ret, yhat_ret, trm_hoy))
    real, pred = trm_hoy * np.exp(y_ret), trm_hoy * np.exp(yhat_ret)
    return {"rmse_pesos": float(np.sqrt(np.mean((real - pred) ** 2))),
            "mae_pesos": float(np.mean(np.abs(real - pred))),
            "r2_retornos": float(r2_score(y_ret, yhat_ret))}
