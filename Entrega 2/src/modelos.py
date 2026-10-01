"""Modelos, espacios de búsqueda y validación anidada (notebook 05).

Contrato con la Entrega 1:
- Línea base única: regresión lineal sobre los rezagos 1, 5 y 10 de la TRM (Pipeline con escalado).
- Modelos avanzados: LightGBM, CatBoost y XGBoost, cada uno con dos conjuntos de variables
  (solo TRM y TRM + macro) → 6 configuraciones.
- Validación anidada: bucle externo walk-forward (desempeño con variabilidad) y bucle interno
  TimeSeriesSplit + RandomizedSearchCV (hiperparámetros). Todo dentro de Pipeline.
"""
import time

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from .config import GAP, N_BUSQUEDA, N_CORTES_INTERNOS, SEMILLA
from .validacion import metricas

# ---------------------------------------------------------------------------
# Espacios de búsqueda (Tabla 4). La señal es débil (EDA, E9): se favorecen árboles poco
# profundos, tasas de aprendizaje bajas, hojas con muchas observaciones y regularización.
# ---------------------------------------------------------------------------
ESPACIOS = {
    "LightGBM": {
        "modelo__n_estimators": [100, 200, 400, 800],
        "modelo__learning_rate": [0.01, 0.03, 0.05, 0.1],
        "modelo__max_depth": [2, 3, 4, 6],
        "modelo__num_leaves": [4, 8, 15, 31],
        "modelo__min_child_samples": [20, 50, 100, 200],
        "modelo__subsample": [0.6, 0.8, 1.0],
        "modelo__colsample_bytree": [0.6, 0.8, 1.0],
        "modelo__reg_lambda": [0.0, 1.0, 5.0, 10.0],
    },
    "CatBoost": {
        "modelo__iterations": [100, 200, 400, 800],
        "modelo__learning_rate": [0.01, 0.03, 0.05, 0.1],
        "modelo__depth": [2, 3, 4, 6],
        "modelo__min_data_in_leaf": [20, 50, 100, 200],
        "modelo__subsample": [0.6, 0.8, 1.0],
        "modelo__rsm": [0.6, 0.8, 1.0],
        "modelo__l2_leaf_reg": [1.0, 3.0, 5.0, 10.0],
    },
    "XGBoost": {
        "modelo__n_estimators": [100, 200, 400, 800],
        "modelo__learning_rate": [0.01, 0.03, 0.05, 0.1],
        "modelo__max_depth": [2, 3, 4, 6],
        "modelo__min_child_weight": [1, 5, 10, 20],
        "modelo__subsample": [0.6, 0.8, 1.0],
        "modelo__colsample_bytree": [0.6, 0.8, 1.0],
        "modelo__reg_lambda": [0.0, 1.0, 5.0, 10.0],
    },
}

JUSTIFICACION_ESPACIOS = {
    "número de árboles (100–800)": "Se combina con la tasa de aprendizaje: tasas bajas necesitan más árboles",
    "tasa de aprendizaje (0,01–0,1)": "Tasas bajas generalizan mejor cuando la señal es débil (EDA E9: R² ≈ 0,03)",
    "profundidad máxima (2–6)": "La relación en la media es casi lineal (E9) y no hay redundancia (E8): interacciones profundas serían ruido",
    "mínimo de observaciones por hoja (20–200) / peso mínimo (1–20)": "Evita reglas basadas en pocos días extremos (E2: el 1 % de los días genera el 21 % del error)",
    "submuestreo de filas y columnas (0,6–1,0)": "Reduce la varianza y la dependencia de una sola variable",
    "regularización L2 (0–10)": "Encoge las predicciones hacia cero, lo prudente con señal débil",
}


def crear_modelo(nombre):
    """Pipeline de cada modelo. Hilos = 1 para que el resultado sea idéntico en cada corrida."""
    if nombre == "lineal":
        return Pipeline([("escala", StandardScaler()), ("modelo", LinearRegression())])
    base = {
        "LightGBM": LGBMRegressor(random_state=SEMILLA, n_jobs=1, verbose=-1, subsample_freq=1),
        "CatBoost": CatBoostRegressor(random_seed=SEMILLA, thread_count=1, verbose=0, allow_writing_files=False,
                                      bootstrap_type="Bernoulli"),
        "XGBoost": XGBRegressor(random_state=SEMILLA, n_jobs=1),
    }[nombre]
    return Pipeline([("modelo", base)])      # los árboles no necesitan escalado


def busqueda(nombre, n_jobs=-1):
    """Bucle interno: RandomizedSearchCV con TimeSeriesSplit (validación siempre posterior)."""
    return RandomizedSearchCV(
        crear_modelo(nombre), ESPACIOS[nombre], n_iter=N_BUSQUEDA,
        cv=TimeSeriesSplit(n_splits=N_CORTES_INTERNOS, gap=GAP),
        scoring="neg_mean_squared_error", random_state=SEMILLA, n_jobs=n_jobs, refit=True)


def evaluar_walk_forward(datos, variables, nombre, cortes, n_jobs=-1):
    """Bucle externo: para cada corte ajusta (y, si es boosting, busca hiperparámetros) solo con su
    entrenamiento y mide en su validación. Devuelve métricas por corte, predicciones e hiperparámetros."""
    filas, preds, params = [], [], []
    X, y, trm = datos[variables], datos["y"], datos["trm"]
    for i, (tr, va) in enumerate(cortes, start=1):
        t0 = time.perf_counter()
        if nombre == "lineal":
            modelo = crear_modelo("lineal").fit(X.iloc[tr], y.iloc[tr])
            elegido = {}
        else:
            bus = busqueda(nombre, n_jobs=n_jobs).fit(X.iloc[tr], y.iloc[tr])
            modelo, elegido = bus.best_estimator_, {k.replace("modelo__", ""): v for k, v in bus.best_params_.items()}
        yhat = modelo.predict(X.iloc[va])
        filas.append({"corte": i, **metricas(y.iloc[va], yhat, trm.iloc[va]), "segundos": time.perf_counter() - t0})
        preds.append(pd.DataFrame({"corte": i, "fecha": datos["fecha"].iloc[va].to_numpy(),
                                   "y": y.iloc[va].to_numpy(), "yhat": yhat}))
        params.append({"corte": i, **elegido})
    return pd.DataFrame(filas), pd.concat(preds, ignore_index=True), pd.DataFrame(params)


def ajustar_final(datos, variables, nombre, n_jobs=-1):
    """Modelo final con TODO el desarrollo: repite la búsqueda interna sobre el desarrollo completo."""
    X, y = datos[variables], datos["y"]
    if nombre == "lineal":
        return crear_modelo("lineal").fit(X, y), {}
    bus = busqueda(nombre, n_jobs=n_jobs).fit(X, y)
    return bus.best_estimator_, {k.replace("modelo__", ""): v for k, v in bus.best_params_.items()}


# ---------------------------------------------------------------------------
# Prueba de robustez: búsqueda bayesiana con Optuna (TPE) en el mismo esquema anidado.
# Usa EXACTAMENTE los mismos valores posibles que ESPACIOS, para que lo único que cambie sea
# la estrategia de búsqueda (aleatoria frente a bayesiana).
# ---------------------------------------------------------------------------
def busqueda_optuna(nombre, X, y, n_trials, semilla=SEMILLA):
    """Bucle interno con Optuna: minimiza el error cuadrático medio en TimeSeriesSplit(3, gap 5)."""
    import optuna
    from sklearn.base import clone
    from sklearn.metrics import mean_squared_error
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    cv = list(TimeSeriesSplit(n_splits=N_CORTES_INTERNOS, gap=GAP).split(X))
    espacio = ESPACIOS[nombre]

    def objetivo(trial):
        params = {k: trial.suggest_categorical(k.replace("modelo__", ""), v) for k, v in espacio.items()}
        errores = []
        for tr, va in cv:
            m = clone(crear_modelo(nombre)).set_params(**params)
            m.fit(X.iloc[tr], y.iloc[tr])
            errores.append(mean_squared_error(y.iloc[va], m.predict(X.iloc[va])))
        return float(np.mean(errores))

    estudio = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=semilla))
    estudio.optimize(objetivo, n_trials=n_trials)
    mejores = {f"modelo__{k}": v for k, v in estudio.best_params.items()}
    return clone(crear_modelo(nombre)).set_params(**mejores).fit(X, y), estudio.best_params


def evaluar_walk_forward_optuna(datos, variables, nombre, cortes, n_trials):
    """Mismo bucle externo que evaluar_walk_forward, con Optuna en el bucle interno."""
    X, y, trm = datos[variables], datos["y"], datos["trm"]
    filas = []
    for i, (tr, va) in enumerate(cortes, start=1):
        modelo, _ = busqueda_optuna(nombre, X.iloc[tr].reset_index(drop=True), y.iloc[tr].reset_index(drop=True), n_trials)
        filas.append({"corte": i, **metricas(y.iloc[va], modelo.predict(X.iloc[va]), trm.iloc[va])})
    return pd.DataFrame(filas)
