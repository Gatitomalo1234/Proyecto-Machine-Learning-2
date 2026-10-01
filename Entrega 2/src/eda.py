"""Funciones del análisis exploratorio (notebook 03). Todas reciben datos de DESARROLLO."""
import numpy as np
import pandas as pd
from scipy import stats


def pruebas_estacionariedad(series):
    """ADF (H0: raíz unitaria) y KPSS (H0: estacionaria) para cada serie.

    Conclusión robusta: 'estacionaria' solo si ADF rechaza y KPSS no rechaza al 5 %.
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


def resumen_distribucion(y):
    """Momentos, normalidad y colas de una serie de retornos."""
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


def concentracion_error(y, fracciones=(0.01, 0.05)):
    """Qué parte del error cuadrático de predecir 'sin cambio' (ŷ = 0) aportan los días más extremos."""
    e2 = np.sort((y ** 2).to_numpy())[::-1]
    total = e2.sum()
    return pd.Series({f"top_{int(100*f)}pct_dias": 100 * e2[:max(1, int(len(e2) * f))].sum() / total
                      for f in fracciones}, name="pct_del_error_cuadratico")


def acf_con_banda(y, rezagos=20):
    """Autocorrelación simple y parcial con banda de 95 % (±1,96/√n)."""
    from statsmodels.tsa.stattools import acf, pacf
    a = acf(y, nlags=rezagos, fft=True)[1:]
    p = pacf(y, nlags=rezagos, method="ywm")[1:]
    return pd.DataFrame({"acf": a, "pacf": p, "banda_95": 1.96 / np.sqrt(len(y))},
                        index=pd.Index(range(1, rezagos + 1), name="rezago"))


def correlacion_con_ic(x, y):
    """Pearson con IC 95 % (transformación de Fisher) y Spearman."""
    r = np.corrcoef(x, y)[0, 1]
    se = 1 / np.sqrt(len(x) - 3)
    lo, hi = np.tanh(np.arctanh(r) - 1.96 * se), np.tanh(np.arctanh(r) + 1.96 * se)
    return {"pearson": r, "ic95_inf": lo, "ic95_sup": hi, "spearman": stats.spearmanr(x, y).statistic}


def medias_por_decil(x, y, q=10):
    """Media del objetivo en cada decil de la variable: muestra la forma de la relación."""
    g = pd.qcut(x, q, duplicates="drop")
    t = pd.DataFrame({"x": x, "y": y}).groupby(g, observed=True).agg(x_media=("x", "mean"), y_media=("y", "mean"),
                                                                       y_se=("y", "sem"), n=("y", "size"))
    return t.reset_index(drop=True)


def vif(X):
    """Factor de inflación de varianza de cada variable (redundancia lineal)."""
    Xc = (X - X.mean()) / X.std()
    inv = np.linalg.inv(np.corrcoef(Xc.T.to_numpy()))
    return pd.Series(np.diag(inv), index=X.columns, name="VIF")
