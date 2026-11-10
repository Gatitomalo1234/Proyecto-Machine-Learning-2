"""Núcleo de la aplicación: del historial de la TRM a la predicción de mañana y su explicación (rúbrica 8.3).

La aplicación (`app/`) no tiene lógica de modelamiento propia: llama a estas funciones, que usan las
mismas fórmulas del proyecto. Flujo:

1. `obtener_trm`: últimas TRM publicadas (en vivo o, si la fuente falla, la copia congelada).
2. `variables_desde_trm`: los retornos logarítmicos de hace 1, 5 y 10 días (las variables del modelo).
3. `predecir`: retorno y TRM de mañana con el modelo final y con la línea base, con un rango de error.
4. `explicar`: cuánto aporta cada variable a la predicción (SHAP en CatBoost; aportes exactos en la lineal).
5. `chequear_coherencia`: si los datos de uso se parecen a los de entrenamiento.

Los modelos son los de `models/modelo_final_despliegue.joblib`, y `cargar_modelos` verifica sus huellas.
"""
from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .config import DATA_PROC, MODELOS, VARS_TRM

if TYPE_CHECKING:
    from pathlib import Path

    from sklearn.pipeline import Pipeline

MODELO_FINAL = "CatBoost · solo_trm"
LINEA_BASE = "Línea base (lineal)"
MIN_HISTORIA = 11                     # TRM necesarias para el rezago de 10 días (TRM_t … TRM_{t-10})
VENTANA_VOLATILIDAD = 20              # TRM para medir el régimen de volatilidad (notebook 07)
VENTANA_RANGO = 40                    # TRM para escalar el rango del 80 % (elegida solo con la validación, notebook 10)
TOLERANCIA_CONTROL = 1e-9             # diferencia máxima admitida en las predicciones de control (retorno)


def obtener_trm(n: int = 60) -> tuple[pd.DataFrame, str, str | None]:
    """Consigue las últimas TRM publicadas: en vivo si la fuente responde; si no, la copia congelada.

    Parámetros
    ----------
    n : int, por defecto 60
        Número de TRM recientes que se piden.

    Devuelve
    --------
    tuple[pd.DataFrame, str, str | None]
        Tabla con ``fecha``, ``vigencia_hasta`` y ``valor`` (orden cronológico); el origen
        (``"en vivo"`` o ``"copia congelada"``); y el error de la fuente si se usó la copia (None si no).
    """
    from .descarga import leer, trm_reciente
    try:
        return trm_reciente(n), "en vivo", None
    except Exception as error:                      # sin internet o fuente caída: la app sigue funcionando
        return leer("trm_vigencias").tail(n).reset_index(drop=True), "copia congelada", str(error)


def variables_desde_trm(trm: pd.Series) -> pd.DataFrame:
    """Calcula las variables del modelo a partir de una serie de TRM publicadas.

    Usa exactamente las fórmulas del notebook 02 (`variables.agregar_variables`):
    r_trm_lk = ln(TRM_{t-k+1} / TRM_{t-k}) para k = 1, 5 y 10. Las pruebas comprueban que, sobre la
    copia congelada, el resultado es idéntico (al bit) a las columnas de `dataset.csv`.

    Parámetros
    ----------
    trm : pd.Series
        Una TRM por publicación, en orden cronológico (como la fuente de la Superfinanciera).

    Devuelve
    --------
    pd.DataFrame
        Mismo índice que `trm`, con las columnas ``r_trm_l1``, ``r_trm_l5`` y ``r_trm_l10``; las primeras
        filas quedan vacías mientras no hay historia suficiente.
    """
    r = np.log(trm).diff()                           # ln(TRM_t / TRM_{t-1})
    return pd.DataFrame({"r_trm_l1": r, "r_trm_l5": r.shift(4), "r_trm_l10": r.shift(9)})


def huella(modelo: Pipeline, X: pd.DataFrame) -> str:
    """Calcula la huella de un modelo: el MD5 de sus predicciones redondeadas a 12 decimales.

    Dos modelos con la misma huella predicen exactamente lo mismo sobre `X`; así se verifica que el
    modelo cargado es el que se entrenó (el archivo .joblib cambia en bytes aunque el modelo no cambie).

    Parámetros
    ----------
    modelo : sklearn.pipeline.Pipeline
        Modelo ajustado.
    X : pd.DataFrame
        Datos sobre los que se calcula (los de entrenamiento del modelo).

    Devuelve
    --------
    str
        MD5 en hexadecimal.
    """
    return hashlib.md5(np.round(modelo.predict(X), 12).tobytes()).hexdigest()


def datos_entrenamiento() -> pd.DataFrame:
    """Lee las filas con las que se entrenaron los modelos de la aplicación (2003–2026).

    Devuelve
    --------
    pd.DataFrame
        Las filas de `dataset.csv` con las 3 variables y el objetivo completos (5.401 días).
    """
    d = pd.read_csv(DATA_PROC / "dataset.csv", parse_dates=["fecha"])
    return d.dropna(subset=VARS_TRM + ["y"]).reset_index(drop=True)


def cargar_modelos(ruta: Path | None = None, verificar: bool = True) -> dict[str, Pipeline]:
    """Carga el modelo final y la línea base de la aplicación y verifica que sean los entrenados.

    Parámetros
    ----------
    ruta : Path o None, por defecto None
        Archivo .joblib; con None, ``models/modelo_final_despliegue.joblib``.
    verificar : bool, por defecto True
        Si es True, compara las predicciones de cada modelo en 25 días de control con las guardadas en
        ``models/modelo_final.json``, con tolerancia `TOLERANCIA_CONTROL`. Se usa tolerancia y no el MD5
        exacto porque otro sistema operativo o procesador puede cambiar el último decimal sin que el
        modelo cambie (p. ej. Streamlit Community Cloud corre en Linux).

    Devuelve
    --------
    dict[str, Pipeline]
        ``{MODELO_FINAL: …, LINEA_BASE: …}``.

    Lanza
    -----
    ValueError
        Si las predicciones de control de algún modelo no coinciden (el archivo no es el modelo documentado).
    """
    import joblib
    modelos = joblib.load(ruta or MODELOS / "modelo_final_despliegue.joblib")
    if verificar:
        control = json.loads((MODELOS / "modelo_final.json").read_text(encoding="utf-8"))["modelo_despliegue"]["predicciones_control"]
        X = datos_entrenamiento()[VARS_TRM].iloc[control["posiciones"]]
        for nombre, modelo in modelos.items():
            if nombre not in control or np.max(np.abs(modelo.predict(X) - np.array(control[nombre]))) > TOLERANCIA_CONTROL:
                raise ValueError(f"{nombre}: la huella (predicciones de control) no coincide con models/modelo_final.json")
    return modelos


def errores_historicos(modelo: str = MODELO_FINAL) -> dict[str, float]:
    """Resume el error del modelo en el test (2022–2026) para dar un rango a cada predicción.

    Parámetros
    ----------
    modelo : str, por defecto `MODELO_FINAL`
        Columna de ``predicciones_test.csv`` (el modelo congelado, evaluado una sola vez).

    Devuelve
    --------
    dict[str, float]
        ``mae_pesos`` y ``rmse_pesos`` del test; los percentiles 10 y 90 del error en retorno en el test
        (``p10_retorno``, ``p90_retorno``, rango fijo); y los percentiles 10 y 90 del error dividido por la
        volatilidad de las últimas `VENTANA_RANGO` TRM (``z10``, ``z90``, rango adaptativo), calculados
        con todos los errores fuera de muestra del modelo (validación 2006–2022 y test 2022–2026).

    Notas
    -----
    El rango adaptativo (notebook 10) se ensancha cuando el mercado está agitado: con el rango fijo, la
    TRM real cayó dentro solo el 67 % de los días de volatilidad alta del test; con el adaptativo, el
    80 %. La ventana de 40 TRM se eligió solo con la validación.
    """
    from .validacion import errores_en_pesos
    t = pd.read_csv(DATA_PROC / "predicciones_test.csv", parse_dates=["fecha"])
    e_ret = t["y"] - t[modelo]
    e_pesos = errores_en_pesos(t["y"], t[modelo], t["trm"])
    v = pd.read_csv(DATA_PROC / "predicciones_validacion.csv", parse_dates=["fecha"])
    v = v[v["modelo"] == modelo]
    vol = datos_entrenamiento().set_index("fecha")["r_trm_l1"].rolling(VENTANA_RANGO).std()
    z = pd.concat([(v["y"] - v["yhat"]) / v["fecha"].map(vol), e_ret / t["fecha"].map(vol)]).dropna()
    return {"mae_pesos": float(np.mean(np.abs(e_pesos))), "rmse_pesos": float(np.sqrt(np.mean(e_pesos ** 2))),
            "p10_retorno": float(np.quantile(e_ret, 0.10)), "p90_retorno": float(np.quantile(e_ret, 0.90)),
            "z10": float(np.quantile(z, 0.10)), "z90": float(np.quantile(z, 0.90))}


def predecir(modelos: dict[str, Pipeline], trm: pd.Series, error: dict[str, float] | None = None) -> pd.DataFrame:
    """Predice el movimiento y la TRM del día siguiente a la última TRM de la serie.

    Parámetros
    ----------
    modelos : dict[str, Pipeline]
        Nombre → modelo ajustado sobre `VARS_TRM` (p. ej. el resultado de `cargar_modelos`).
    trm : pd.Series
        TRM publicadas en orden cronológico; se usa la última como "hoy" (al menos `MIN_HISTORIA`).
    error : dict[str, float] o None, por defecto None
        Resultado de `errores_historicos`; si se da, se agrega el rango del 80 % para cada modelo:
        adaptativo (escalado por la volatilidad de las últimas `VENTANA_RANGO` TRM) si la serie tiene al
        menos `VENTANA_RANGO` + 1 TRM; si no, el rango fijo.

    Devuelve
    --------
    pd.DataFrame
        Una fila por modelo: ``retorno`` (ln de mañana/hoy), ``movimiento_pb``, ``trm_hoy``,
        ``trm_manana``, ``movimiento_pesos`` y, si hay `error`, ``trm_min_80``, ``trm_max_80`` y
        ``rango`` ("adaptativo" o "fijo").

    Lanza
    -----
    ValueError
        Si la serie tiene menos de `MIN_HISTORIA` TRM.
    """
    if len(trm) < MIN_HISTORIA:
        raise ValueError(f"se necesitan al menos {MIN_HISTORIA} TRM publicadas y llegaron {len(trm)}")
    x = variables_desde_trm(trm.reset_index(drop=True)).iloc[[-1]]
    hoy = float(trm.iloc[-1])
    adaptativo = error is not None and "z10" in error and len(trm) > VENTANA_RANGO
    if adaptativo:
        vol = float(np.log(trm.reset_index(drop=True)).diff().tail(VENTANA_RANGO).std())
        bajo, alto = error["z10"] * vol, error["z90"] * vol
    elif error is not None:
        bajo, alto = error["p10_retorno"], error["p90_retorno"]
    filas = []
    for nombre, modelo in modelos.items():
        ret = float(modelo.predict(x[VARS_TRM])[0])
        fila = {"modelo": nombre, "retorno": ret, "movimiento_pb": 1e4 * ret, "trm_hoy": hoy,
                "trm_manana": hoy * np.exp(ret), "movimiento_pesos": hoy * np.exp(ret) - hoy}
        if error is not None:
            fila["trm_min_80"] = hoy * np.exp(ret + bajo)
            fila["trm_max_80"] = hoy * np.exp(ret + alto)
            fila["rango"] = "adaptativo" if adaptativo else "fijo"
        filas.append(fila)
    return pd.DataFrame(filas).set_index("modelo")


def explicar(modelos: dict[str, Pipeline], trm: pd.Series) -> pd.DataFrame:
    """Descompone la predicción de cada modelo en el aporte de cada variable, en puntos básicos.

    Para CatBoost se usan valores SHAP (TreeExplainer): valor base + aportes = predicción. Para la
    regresión lineal el aporte es exacto: coeficiente × (valor − media del entrenamiento) / desviación,
    y el valor base es el intercepto.

    Parámetros
    ----------
    modelos : dict[str, Pipeline]
        Debe incluir `MODELO_FINAL` y `LINEA_BASE`.
    trm : pd.Series
        TRM publicadas en orden cronológico (al menos `MIN_HISTORIA`).

    Devuelve
    --------
    pd.DataFrame
        Índice ``variable`` (las 3 variables y ``base``) y una columna por modelo con el aporte en pb;
        la suma de cada columna es la predicción del modelo en pb. Incluye la columna ``valor_pct``
        con el valor de cada variable (movimiento en %).
    """
    import shap
    x = variables_desde_trm(trm.reset_index(drop=True)).iloc[[-1]][VARS_TRM]
    cat = modelos[MODELO_FINAL].named_steps["modelo"]
    exp = shap.TreeExplainer(cat)
    aportes = {MODELO_FINAL: [*exp.shap_values(x)[0], float(np.ravel(exp.expected_value)[0])]}
    lin = modelos[LINEA_BASE]
    esc, reg = lin.named_steps["escala"], lin.named_steps["modelo"]
    aportes[LINEA_BASE] = [*(reg.coef_ * (x.to_numpy()[0] - esc.mean_) / esc.scale_), float(reg.intercept_)]
    tabla = pd.DataFrame({k: 1e4 * np.array(v) for k, v in aportes.items()}, index=pd.Index(VARS_TRM + ["base"], name="variable"))
    tabla["valor_pct"] = [*(100 * x.to_numpy()[0]), np.nan]
    return tabla


def referencia_entrenamiento() -> dict[str, object]:
    """Resume los datos de entrenamiento contra los que se comparan los datos de uso.

    Devuelve
    --------
    dict[str, object]
        ``rango``: percentiles 0,5 y 99,5 de cada variable (DataFrame); ``vol_terciles``: los cortes
        de los terciles de la volatilidad de 20 TRM (en retorno diario); ``hueco_max_dias``: el mayor
        número de días entre dos TRM seguidas en el entrenamiento.
    """
    d = datos_entrenamiento()
    vol = d["r_trm_l1"].rolling(VENTANA_VOLATILIDAD).std().dropna()
    return {"rango": d[VARS_TRM].quantile([0.005, 0.995]),
            "vol_terciles": tuple(vol.quantile([1 / 3, 2 / 3])),
            "hueco_max_dias": int(d["fecha"].diff().dt.days.max())}


def chequear_coherencia(tabla_trm: pd.DataFrame, referencia: dict[str, object], hoy: pd.Timestamp,
                        modelo_verificado: bool = True, dias_max: int = 4) -> pd.DataFrame:
    """Verifica que los datos de uso sean coherentes con los de entrenamiento (rúbrica 8.3).

    Parámetros
    ----------
    tabla_trm : pd.DataFrame
        Últimas TRM con ``fecha`` y ``valor`` (resultado de `obtener_trm`).
    referencia : dict[str, object]
        Resultado de `referencia_entrenamiento`.
    hoy : pd.Timestamp
        Fecha de consulta, para medir qué tan reciente es la última TRM.
    modelo_verificado : bool, por defecto True
        Si las huellas de los modelos coincidieron al cargarlos.
    dias_max : int, por defecto 4
        Días calendario máximos entre la última TRM y hoy para considerarla vigente (cubre un fin de
        semana largo con festivo).

    Devuelve
    --------
    pd.DataFrame
        Una fila por chequeo (``datos_recientes``, ``historia_completa``, ``dentro_de_lo_visto``,
        ``regimen_volatilidad``, ``modelo_correcto``) con ``estado`` ("ok", "aviso" o "alerta") y
        ``detalle`` en español sencillo.
    """
    t = tabla_trm.sort_values("fecha").reset_index(drop=True)
    filas = []
    dias = int((pd.Timestamp(hoy).normalize() - t["fecha"].iloc[-1]).days)
    filas.append(("datos_recientes", "ok" if dias <= dias_max else "aviso",
                  f"La última TRM es del {t['fecha'].iloc[-1]:%d/%m/%Y} ({dias} días antes de la consulta)."))
    hueco = int(t["fecha"].diff().dt.days.max()) if len(t) > 1 else 0
    completa = len(t) >= MIN_HISTORIA and hueco <= referencia["hueco_max_dias"]
    filas.append(("historia_completa", "ok" if completa else "alerta",
                  f"{len(t)} TRM disponibles (se necesitan {MIN_HISTORIA}); mayor salto entre publicaciones: "
                  f"{hueco} días (en el entrenamiento, {referencia['hueco_max_dias']})."))
    x = variables_desde_trm(t["valor"]).iloc[-1]
    rango = referencia["rango"]
    fuera = [v for v in VARS_TRM if not (rango.loc[0.005, v] <= x[v] <= rango.loc[0.995, v])]
    filas.append(("dentro_de_lo_visto", "ok" if not fuera else "alerta",
                  "Los tres movimientos están dentro del rango del entrenamiento." if not fuera else
                  f"Fuera del 99 % central del entrenamiento: {', '.join(fuera)}. La predicción es una extrapolación."))
    vol = float(t["valor"].pipe(np.log).diff().tail(VENTANA_VOLATILIDAD).std())
    bajo, alto = referencia["vol_terciles"]
    nivel = "baja" if vol < bajo else ("media" if vol < alto else "alta")
    filas.append(("regimen_volatilidad", "aviso" if nivel == "alta" else "ok",
                  f"Volatilidad de las últimas {VENTANA_VOLATILIDAD} TRM: {100 * vol * np.sqrt(252):.1f} % anual ({nivel}). ".replace(".", ",", 1)
                  + ("Con volatilidad alta, el movimiento de hoy se repite menos mañana." if nivel == "alta" else "")))
    filas.append(("modelo_correcto", "ok" if modelo_verificado else "alerta",
                  "Los modelos cargados coinciden con su huella documentada." if modelo_verificado else
                  "Los modelos cargados no coinciden con su huella documentada."))
    return pd.DataFrame(filas, columns=["chequeo", "estado", "detalle"]).set_index("chequeo")
