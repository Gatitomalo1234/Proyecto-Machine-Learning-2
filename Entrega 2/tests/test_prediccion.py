"""Pruebas del núcleo de la aplicación (`src/prediccion.py`) y de la TRM en vivo.

Lo central: el preprocesamiento que usa la aplicación es idéntico al del entrenamiento, y sus
predicciones coinciden con las del notebook 06 para fechas conocidas.
Ejecutar desde la carpeta Entrega 2/: ``python -m pytest`` (``-m "not red"`` para omitir las que usan internet).
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import pytest

from src import config as cfg
from src import prediccion as p
from src.descarga import leer


@pytest.fixture(scope="module")
def copia() -> pd.DataFrame:
    """Copia congelada de la TRM de la Superfinanciera (la misma fuente que se consulta en vivo)."""
    return leer("trm_vigencias")


@pytest.fixture(scope="module")
def modelos() -> dict:
    """Modelos de la aplicación, verificados con su huella."""
    return p.cargar_modelos()


@pytest.fixture(scope="module")
def referencia() -> dict:
    """Resumen de los datos de entrenamiento."""
    return p.referencia_entrenamiento()


def test_variables_identicas_al_entrenamiento(copia: pd.DataFrame) -> None:
    """Las variables calculadas por la aplicación son idénticas, al bit, a las de dataset.csv."""
    datos = pd.read_csv(cfg.DATA_PROC / "dataset.csv", parse_dates=["fecha"]).set_index("fecha")
    x = p.variables_desde_trm(copia.set_index("fecha")["valor"]).reindex(datos.index)
    assert x.notna().all().all()
    for v in cfg.VARS_TRM:
        assert (x[v] == datos[v]).all(), v


def test_prediccion_coincide_con_el_notebook(copia: pd.DataFrame) -> None:
    """Con los modelos congelados, `predecir` da lo mismo que el notebook 06 en fechas del test."""
    congelados = joblib.load(cfg.MODELOS / "modelos_finales.joblib")
    usados = {k: congelados[k] for k in (p.MODELO_FINAL, p.LINEA_BASE)}
    test = pd.read_csv(cfg.DATA_PROC / "predicciones_test.csv", parse_dates=["fecha"])
    for _, fila in test.iloc[[0, 500, len(test) - 1]].iterrows():
        historia = copia.loc[copia["fecha"] <= fila["fecha"], "valor"]
        pred = p.predecir(usados, historia)
        assert np.isclose(pred.loc[p.MODELO_FINAL, "retorno"], fila[p.MODELO_FINAL], rtol=0, atol=1e-12)
        assert np.isclose(pred.loc[p.LINEA_BASE, "retorno"], fila[p.LINEA_BASE], rtol=0, atol=1e-12)
        assert np.isclose(pred.loc[p.MODELO_FINAL, "trm_hoy"], fila["trm"])


def test_huella_detecta_un_modelo_equivocado(tmp_path, modelos: dict) -> None:
    """Si el archivo de modelos no es el documentado, `cargar_modelos` se detiene."""
    cambiados = {p.MODELO_FINAL: modelos[p.LINEA_BASE], p.LINEA_BASE: modelos[p.LINEA_BASE]}
    joblib.dump(cambiados, tmp_path / "otro.joblib")
    with pytest.raises(ValueError, match="huella"):
        p.cargar_modelos(tmp_path / "otro.joblib")


def test_explicacion_suma_la_prediccion(copia: pd.DataFrame, modelos: dict) -> None:
    """Valor base + aportes = predicción, en los dos modelos."""
    historia = copia["valor"].tail(60)
    pred, exp = p.predecir(modelos, historia), p.explicar(modelos, historia)
    for m in (p.MODELO_FINAL, p.LINEA_BASE):
        assert np.isclose(exp[m].sum(), pred.loc[m, "movimiento_pb"], atol=1e-6)


def test_rango_de_error_contiene_la_prediccion(copia: pd.DataFrame, modelos: dict) -> None:
    """El rango del 80 % rodea a la predicción y mide decenas de pesos (MAE del test ≈ 27)."""
    error = p.errores_historicos()
    assert 20 < error["mae_pesos"] < 35
    pred = p.predecir(modelos, copia["valor"].tail(60), error)
    assert (pred["trm_min_80"] < pred["trm_manana"]).all() and (pred["trm_manana"] < pred["trm_max_80"]).all()


def test_predecir_exige_historia(copia: pd.DataFrame, modelos: dict) -> None:
    """Con menos de 11 TRM no se puede calcular el rezago de 10 días."""
    with pytest.raises(ValueError, match="al menos"):
        p.predecir(modelos, copia["valor"].tail(5))


def test_coherencia_en_condiciones_normales(copia: pd.DataFrame, referencia: dict) -> None:
    """En un periodo tranquilo, con datos recientes y normales, todos los chequeos están en verde."""
    vol = np.log(copia["valor"]).diff().rolling(20).std()
    fin = int(vol.iloc[100:].idxmin())                         # el periodo más tranquilo de la historia
    t = copia.iloc[fin - 59:fin + 1]
    c = p.chequear_coherencia(t, referencia, hoy=t["fecha"].iloc[-1])
    assert (c["estado"] == "ok").all(), c


def test_coherencia_detecta_problemas(copia: pd.DataFrame, referencia: dict) -> None:
    """Datos viejos, historia corta, un salto fuera de lo visto, mercado agitado y modelo no verificado."""
    t = copia.tail(60).reset_index(drop=True)
    hoy = t["fecha"].iloc[-1]
    assert p.chequear_coherencia(t, referencia, hoy + pd.Timedelta(days=30)).loc["datos_recientes", "estado"] == "aviso"
    assert p.chequear_coherencia(t.tail(5), referencia, hoy).loc["historia_completa", "estado"] == "alerta"
    salto = t.copy()
    salto.loc[salto.index[-1], "valor"] *= 1.10                                  # +10 % en un día
    assert p.chequear_coherencia(salto, referencia, hoy).loc["dentro_de_lo_visto", "estado"] == "alerta"
    agitado = t.copy()
    agitado["valor"] = agitado["valor"] * np.where(np.arange(len(t)) % 2 == 0, 1.02, 0.98)   # ±2 % día a día
    assert p.chequear_coherencia(agitado, referencia, hoy).loc["regimen_volatilidad", "estado"] == "aviso"
    assert p.chequear_coherencia(t, referencia, hoy, modelo_verificado=False).loc["modelo_correcto", "estado"] == "alerta"


def test_sin_internet_usa_la_copia(monkeypatch) -> None:
    """Si la fuente no responde, la aplicación sigue con la copia congelada y lo informa."""
    import src.descarga as d

    def falla(*args, **kwargs):
        """Simula una caída de la red."""
        raise RuntimeError("sin red")

    monkeypatch.setattr(d, "_get", falla)
    tabla, origen, error = p.obtener_trm(30)
    assert origen == "copia congelada" and "sin red" in error and len(tabla) == 30


@pytest.mark.red
def test_trm_en_vivo_coincide_con_la_copia(copia: pd.DataFrame) -> None:
    """La TRM en vivo tiene el mismo formato y los mismos valores que la copia en los días comunes."""
    from src.descarga import trm_reciente
    try:
        vivo = trm_reciente(120)
    except RuntimeError:
        pytest.skip("datos.gov.co no responde")
    comun = vivo.merge(copia, on="fecha", suffixes=("_vivo", "_copia"))
    assert len(comun) > 0
    assert (comun["valor_vivo"] == comun["valor_copia"]).all()
    assert (comun["vigencia_hasta_vivo"] == comun["vigencia_hasta_copia"]).all()
    assert vivo["fecha"].is_monotonic_increasing and not vivo["fecha"].duplicated().any()
