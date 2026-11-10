"""Funciones del análisis exploratorio (notebook 03). Todas reciben datos de DESARROLLO.

El EDA se hace antes de entrenar y nunca mira el test: cada función responde una de las preguntas
E1–E9 del notebook 03, y sus conclusiones justifican decisiones del modelado (objetivo en retornos,
métricas, rangos de hiperparámetros e hipótesis H1–H6).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy import stats

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


def pruebas_estacionariedad(series: dict[str, pd.Series]) -> pd.DataFrame:
    """Aplica ADF (H0: raíz unitaria) y KPSS (H0: estacionaria) a cada serie (pregunta E1).

    Conclusión robusta: 'estacionaria' solo si ADF rechaza y KPSS no rechaza al 5 %.

    Parámetros
    ----------
    series : dict[str, pd.Series]
        Nombre → serie (niveles o retornos); los vacíos se descartan.

    Devuelve
    --------
    pd.DataFrame
        Índice ``serie`` y columnas ``p_ADF``, ``p_KPSS`` y ``conclusion``: "estacionaria",
        "no estacionaria" (ADF no rechaza y KPSS rechaza) o "mixta" (las pruebas no coinciden).

    Notas
    -----
    ADF usa el número de rezagos que elige el AIC; KPSS, constante y rezagos automáticos. Cuando el
    p-valor de KPSS sale de su tabla, statsmodels devuelve el extremo de la tabla y avisa; el aviso
    se silencia porque el extremo basta para concluir.
    """
    from statsmodels.tsa.stattools import adfuller, kpss
    import warnings
    filas = []
    for nombre, s in series.items():
        s = s.dropna()
        p_adf = adfuller(s, autolag="AIC")[1]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")      # KPSS avisa cuando el p-valor sale del rango tabulado
            p_kpss = kpss(s, regression="c", nlags="auto")[1]
        filas.append({"serie": nombre, "p_ADF": p_adf, "p_KPSS": p_kpss,
                      "conclusion": "estacionaria" if (p_adf < 0.05 and p_kpss >= 0.05)
                      else ("no estacionaria" if (p_adf >= 0.05 and p_kpss < 0.05) else "mixta")})
    return pd.DataFrame(filas).set_index("serie")


def resumen_distribucion(y: pd.Series) -> pd.Series:
    """Resume momentos, normalidad y colas de una serie de retornos (pregunta E2).

    Parámetros
    ----------
    y : pd.Series
        Retornos diarios, sin vacíos.

    Devuelve
    --------
    pd.Series
        ``n``, ``media``, ``desv_est``, ``volatilidad_anual_pct`` (desviación × √252 × 100),
        ``asimetria``, ``curtosis_exceso`` (0 en una normal), ``pct_dias_mas_4sd``,
        ``veces_lo_esperado_si_normal`` (frecuencia de días a más de 4 desviaciones frente a la de
        una normal) y ``p_jarque_bera``.
    """
    z = (y - y.mean()) / y.std()
    jb = stats.jarque_bera(y)
    esperado_4sd = 2 * stats.norm.sf(4)
    return pd.Series({
        "n": len(y), "media": y.mean(), "desv_est": y.std(),
        "volatilidad_anual_pct": 100 * y.std() * np.sqrt(252),
        "asimetria": stats.skew(y), "curtosis_exceso": stats.kurtosis(y),
        "pct_dias_mas_4sd": 100 * (z.abs() > 4).mean(),
        "veces_lo_esperado_si_normal": (z.abs() > 4).mean() / esperado_4sd,
        "p_jarque_bera": jb.pvalue,
    })


def concentracion_error(y: pd.Series, fracciones: tuple[float, ...] = (0.01, 0.05)) -> pd.Series:
    """Mide qué parte del error cuadrático de predecir 'sin cambio' (ŷ = 0) aportan los días más extremos.

    Si pocos días concentran gran parte del error, el RMSE queda dominado por ellos; por eso el
    proyecto reporta el RMSE junto con el MAE (pregunta E2).

    Parámetros
    ----------
    y : pd.Series
        Retornos diarios.
    fracciones : tuple[float, ...], por defecto (0.01, 0.05)
        Fracciones de los días más extremos que se evalúan (1 % y 5 %); siempre se toma al menos un día.

    Devuelve
    --------
    pd.Series
        Nombre ``pct_del_error_cuadratico``; índice ``top_<p>pct_dias`` con el porcentaje del error
        total que aportan esos días.
    """
    e2 = np.sort((y ** 2).to_numpy())[::-1]
    total = e2.sum()
    return pd.Series({f"top_{int(100*f)}pct_dias": 100 * e2[:max(1, int(len(e2) * f))].sum() / total
                      for f in fracciones}, name="pct_del_error_cuadratico")


def acf_con_banda(y: ArrayLike, rezagos: int = 20) -> pd.DataFrame:
    """Calcula la autocorrelación simple y parcial con su banda de 95 % (±1,96/√n).

    Aplicada a los retornos mide la memoria de la serie (E4); aplicada a los retornos al cuadrado,
    el agrupamiento de la volatilidad (E3).

    Parámetros
    ----------
    y : array-like
        Serie sin vacíos, en orden temporal.
    rezagos : int, por defecto 20
        Último rezago que se calcula (unas cuatro semanas hábiles).

    Devuelve
    --------
    pd.DataFrame
        Índice ``rezago`` (1 a `rezagos`) y columnas ``acf``, ``pacf`` (método de Yule-Walker) y
        ``banda_95``; un valor fuera de ±banda_95 es significativo al 5 %.
    """
    from statsmodels.tsa.stattools import acf, pacf
    a = acf(y, nlags=rezagos, fft=True)[1:]
    p = pacf(y, nlags=rezagos, method="ywm")[1:]
    return pd.DataFrame({"acf": a, "pacf": p, "banda_95": 1.96 / np.sqrt(len(y))},
                        index=pd.Index(range(1, rezagos + 1), name="rezago"))


def correlacion_con_ic(x: ArrayLike, y: ArrayLike) -> dict[str, float]:
    """Calcula la correlación de Pearson con IC 95 % (transformación de Fisher) y la de Spearman (pregunta E5).

    Si Pearson y Spearman coinciden, la relación no depende de unos pocos valores extremos.

    Parámetros
    ----------
    x : array-like
        Variable predictora, sin vacíos.
    y : array-like
        Objetivo, alineado fila a fila con `x`.

    Devuelve
    --------
    dict[str, float]
        ``pearson``, ``ic95_inf``, ``ic95_sup`` y ``spearman``. Si el intervalo contiene el 0, la
        correlación no es significativa al 5 %.
    """
    r = np.corrcoef(x, y)[0, 1]
    se = 1 / np.sqrt(len(x) - 3)
    lo, hi = np.tanh(np.arctanh(r) - 1.96 * se), np.tanh(np.arctanh(r) + 1.96 * se)
    return {"pearson": r, "ic95_inf": lo, "ic95_sup": hi, "spearman": stats.spearmanr(x, y).statistic}


def medias_por_decil(x: pd.Series, y: pd.Series, q: int = 10) -> pd.DataFrame:
    """Calcula la media del objetivo en cada decil de la variable para ver la forma de la relación.

    Si las medias suben o bajan en línea recta, un modelo lineal captura la relación; si hay curvas
    o escalones, un modelo de árboles podría aprovecharlos (preguntas E5 y E9, Figura E4).

    Parámetros
    ----------
    x : pd.Series
        Variable predictora.
    y : pd.Series
        Objetivo, con el mismo índice que `x`.
    q : int, por defecto 10
        Número de grupos por cuantiles; si hay valores repetidos pueden salir menos.

    Devuelve
    --------
    pd.DataFrame
        Una fila por grupo, de menor a mayor `x`, con ``x_media``, ``y_media``, ``y_se`` (error
        estándar de la media) y ``n``.
    """
    g = pd.qcut(x, q, duplicates="drop")
    t = pd.DataFrame({"x": x, "y": y}).groupby(g, observed=True).agg(x_media=("x", "mean"), y_media=("y", "mean"),
                                                                       y_se=("y", "sem"), n=("y", "size"))
    return t.reset_index(drop=True)


def vif(X: pd.DataFrame) -> pd.Series:
    """Calcula el factor de inflación de varianza de cada variable (redundancia lineal, pregunta E8).

    Un VIF cercano a 1 indica que la variable no se puede predecir con las demás; por encima de 5
    se suele considerar redundante.

    Parámetros
    ----------
    X : pd.DataFrame
        Variables predictoras sin vacíos, una por columna.

    Devuelve
    --------
    pd.Series
        Nombre ``VIF`` e índice con los nombres de las columnas de `X`.

    Notas
    -----
    Se calcula como la diagonal de la inversa de la matriz de correlación; si dos columnas son
    perfectamente colineales, numpy no puede invertirla y lanza ``LinAlgError``.
    """
    Xc = (X - X.mean()) / X.std()
    inv = np.linalg.inv(np.corrcoef(Xc.T.to_numpy()))
    return pd.Series(np.diag(inv), index=X.columns, name="VIF")
