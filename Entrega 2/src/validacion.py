"""Partición cronológica, cortes walk-forward y métricas (rúbricas 7.2 y 7.4).

La partición (notebook 02) separa desarrollo y test sin mezclar el orden del tiempo; los cortes
walk-forward (notebook 05) son el bucle externo de la validación anidada; las métricas son las del
contrato con la Entrega 1 y se usan igual en validación (notebook 05) y en test (notebook 06).
Las pruebas formales de comparación (Diebold-Mariano y bootstrap por bloques) se usan en el
notebook 08 para decidir si dos modelos son distinguibles.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from .config import FRACCION_TEST, GAP, N_CORTES_EXTERNOS, SEMILLA

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


def errores_en_pesos(y_ret: ArrayLike, yhat_ret: ArrayLike, trm_hoy: ArrayLike) -> np.ndarray:
    """Calcula el error de cada día en pesos: TRM real de mañana menos TRM predicha.

    Usa la misma reconstrucción que `metricas` (TRM_{t+1} = TRM_t · e^y), así que
    ``sqrt(mean(errores²))`` es exactamente el ``rmse_pesos`` de `metricas`.

    Parámetros
    ----------
    y_ret : array-like de forma (n,)
        Retorno logarítmico real del día siguiente.
    yhat_ret : array-like de forma (n,)
        Retorno predicho.
    trm_hoy : array-like de forma (n,)
        TRM vigente en t, en pesos.

    Devuelve
    --------
    np.ndarray de forma (n,)
        Error en pesos de cada día (positivo = el modelo se quedó corto).
    """
    y_ret, yhat_ret, trm_hoy = map(np.asarray, (y_ret, yhat_ret, trm_hoy))
    return trm_hoy * np.exp(y_ret) - trm_hoy * np.exp(yhat_ret)


def prueba_diebold_mariano(e1: ArrayLike, e2: ArrayLike, rezagos: int | None = None) -> dict[str, float]:
    """Prueba si dos modelos tienen el mismo error cuadrático medio (prueba de Diebold-Mariano).

    Compara, día a día, la pérdida cuadrática de dos modelos evaluados sobre los mismos días:
    d_t = e1_t² − e2_t². H0: E[d_t] = 0 (los dos modelos predicen igual de bien). Como los errores
    diarios pueden estar autocorrelacionados y tener varianza cambiante, la varianza de la media de
    d_t se estima con Newey-West (HAC) y el estadístico se corrige para muestras finitas con el
    factor de Harvey, Leybourne y Newbold (horizonte de un día); el p-valor usa una t con n − 1
    grados de libertad.

    Parámetros
    ----------
    e1, e2 : array-like de forma (n,)
        Errores en pesos de los dos modelos, alineados día a día (ver `errores_en_pesos`).
    rezagos : int o None, por defecto None
        Rezagos de Newey-West. Con None se usa la regla floor(4 · (n/100)^(2/9)) (8 con 3.600 días).

    Devuelve
    --------
    dict[str, float]
        ``diferencia_mse`` (media de d_t; negativa = e1 mejor), ``estadistico`` (DM con la corrección),
        ``p_valor`` (bilateral), ``n`` y ``rezagos``.

    Lanza
    -----
    ValueError
        Si `e1` y `e2` no tienen el mismo largo.

    Notas
    -----
    Un p-valor alto no prueba que los modelos sean iguales; dice que con estos datos no se pueden
    distinguir. Referencias: Diebold y Mariano (1995); Harvey, Leybourne y Newbold (1997).
    """
    from scipy import stats
    e1, e2 = np.asarray(e1, dtype=float), np.asarray(e2, dtype=float)
    if e1.shape != e2.shape:
        raise ValueError(f"e1 y e2 deben tener el mismo largo ({e1.shape} frente a {e2.shape})")
    d = e1 ** 2 - e2 ** 2
    n = len(d)
    rezagos = int(np.floor(4 * (n / 100) ** (2 / 9))) if rezagos is None else rezagos
    u = d - d.mean()
    varianza = u @ u / n
    for k in range(1, rezagos + 1):
        varianza += 2 * (1 - k / (rezagos + 1)) * (u[k:] @ u[:-k]) / n
    estadistico = d.mean() / np.sqrt(varianza / n) * np.sqrt((n - 1) / n)
    return {"diferencia_mse": float(d.mean()), "estadistico": float(estadistico),
            "p_valor": float(2 * stats.t.sf(abs(estadistico), n - 1)), "n": n, "rezagos": rezagos}


def bootstrap_bloques_diferencia_rmse(e1: ArrayLike, e2: ArrayLike, grupos: ArrayLike | None = None,
                                      largo_bloque: int = 20, repeticiones: int = 2000,
                                      semilla: int = SEMILLA) -> dict[str, float]:
    """Calcula un intervalo de confianza para la diferencia de RMSE entre dos modelos (bootstrap por bloques).

    Remuestrea bloques de días seguidos (no días sueltos), para conservar la autocorrelación y los
    periodos de volatilidad de los errores. Si se dan `grupos` (por ejemplo, el corte de validación),
    remuestrea dentro de cada grupo, así cada réplica conserva el peso de cada periodo.

    Parámetros
    ----------
    e1, e2 : array-like de forma (n,)
        Errores en pesos de los dos modelos, alineados día a día y en orden temporal.
    grupos : array-like de forma (n,) o None, por defecto None
        Etiqueta del grupo de cada día; con None todo es un solo grupo.
    largo_bloque : int, por defecto 20
        Días por bloque (unas cuatro semanas hábiles).
    repeticiones : int, por defecto 2000
        Número de réplicas bootstrap.
    semilla : int, por defecto `SEMILLA`
        Semilla del generador, para que el intervalo sea repetible.

    Devuelve
    --------
    dict[str, float]
        ``diferencia_rmse`` (RMSE(e1) − RMSE(e2) con los datos originales; negativa = e1 mejor),
        ``ic95_inf`` e ``ic95_sup`` (percentiles 2,5 y 97,5 de las réplicas) y
        ``prob_e1_mejor`` (fracción de réplicas en que e1 tiene menor RMSE).

    Lanza
    -----
    ValueError
        Si `e1`, `e2` y `grupos` no tienen el mismo largo, o si algún grupo tiene menos días que
        `largo_bloque`.
    """
    e1, e2 = np.asarray(e1, dtype=float), np.asarray(e2, dtype=float)
    grupos = np.zeros(len(e1)) if grupos is None else np.asarray(grupos)
    if not (e1.shape == e2.shape == grupos.shape):
        raise ValueError("e1, e2 y grupos deben tener el mismo largo")
    indices = [np.flatnonzero(grupos == g) for g in pd.unique(grupos)]
    if min(len(i) for i in indices) < largo_bloque:
        raise ValueError(f"cada grupo debe tener al menos {largo_bloque} días")
    rmse = lambda e: np.sqrt(np.mean(e ** 2))
    rng = np.random.default_rng(semilla)
    replicas = np.empty(repeticiones)
    for b in range(repeticiones):
        elegidos = []
        for idx in indices:
            n = len(idx)
            inicios = rng.integers(0, n - largo_bloque + 1, int(np.ceil(n / largo_bloque)))
            elegidos.append(idx[np.concatenate([np.arange(i, i + largo_bloque) for i in inicios])[:n]])
        sel = np.concatenate(elegidos)
        replicas[b] = rmse(e1[sel]) - rmse(e2[sel])
    lo, hi = np.percentile(replicas, [2.5, 97.5])
    return {"diferencia_rmse": float(rmse(e1) - rmse(e2)), "ic95_inf": float(lo), "ic95_sup": float(hi),
            "prob_e1_mejor": float(np.mean(replicas < 0))}
