"""Aplicación del Proyecto Integrador ML2: predicción de la próxima TRM (rúbrica 8.3).

Toma las últimas TRM publicadas (en vivo desde datos.gov.co o, si falla, la copia congelada), calcula
las variables del modelo con las mismas fórmulas del entrenamiento, predice la próxima TRM con el
modelo final (CatBoost · solo TRM) y con la línea base, explica la predicción y verifica que los
datos de uso sean coherentes con los de entrenamiento. Toda la lógica está en `src/prediccion.py`.

Ejecutar desde la carpeta Entrega 2/:
    streamlit run app/app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
for ruta in (RAIZ, RAIZ / "app"):
    if str(ruta) not in sys.path:
        sys.path.insert(0, str(ruta))

import numpy as np                                                   # noqa: E402
import pandas as pd                                                  # noqa: E402
import plotly.graph_objects as go                                    # noqa: E402
import streamlit as st                                               # noqa: E402

import estilo as e                                                   # noqa: E402
from src import prediccion as p                                      # noqa: E402
from src.config import DATA_PROC, MODELOS                            # noqa: E402
from src.descarga import leer                                        # noqa: E402

REPO = "https://github.com/Gatitomalo1234/Proyecto-Machine-Learning-2"
NOMBRES = {"r_trm_l1": "Movimiento de hoy", "r_trm_l5": "Movimiento de hace 5 TRM", "r_trm_l10": "Movimiento de hace 10 TRM",
           "base": "Punto de partida (promedio)"}
CHEQUEOS = {"datos_recientes": "Datos recientes", "historia_completa": "Historia completa",
            "dentro_de_lo_visto": "Dentro de lo visto", "regimen_volatilidad": "Régimen de volatilidad",
            "modelo_correcto": "Modelo correcto"}


# ---------------------------------------------------------------------------
# Carga (en caché: los modelos una vez; la TRM en vivo cada hora)
# ---------------------------------------------------------------------------
@st.cache_resource
def recursos() -> dict[str, object]:
    """Carga una sola vez los modelos verificados, la referencia de entrenamiento y los errores históricos.

    Devuelve
    --------
    dict[str, object]
        ``modelos``, ``verificado`` (bool), ``referencia``, ``error``, ``ficha`` (modelo_final.json) y
        ``copia`` (la TRM congelada, para simular fechas pasadas).
    """
    try:
        modelos, verificado = p.cargar_modelos(), True
    except ValueError:
        modelos, verificado = p.cargar_modelos(verificar=False), False
    return {"modelos": modelos, "verificado": verificado, "referencia": p.referencia_entrenamiento(),
            "error": p.errores_historicos(), "copia": leer("trm_vigencias"),
            "ficha": json.loads((MODELOS / "modelo_final.json").read_text(encoding="utf-8"))}


@st.cache_data(ttl=3600, show_spinner="Consultando la TRM en datos.gov.co…")
def trm_actual() -> tuple[pd.DataFrame, str, str | None]:
    """Consulta las últimas TRM publicadas (se guarda una hora para no consultar en cada clic).

    Devuelve
    --------
    tuple[pd.DataFrame, str, str | None]
        Lo mismo que `src.prediccion.obtener_trm`.
    """
    return p.obtener_trm(60)


# ---------------------------------------------------------------------------
# Gráficos (Plotly, monocromo)
# ---------------------------------------------------------------------------
def diseno(fig: go.Figure, alto: int = 340) -> go.Figure:
    """Aplica el estilo monocromo de la aplicación a una figura de Plotly.

    Parámetros
    ----------
    fig : plotly.graph_objects.Figure
        Figura que se ajusta.
    alto : int, por defecto 340
        Altura en píxeles.

    Devuelve
    --------
    plotly.graph_objects.Figure
        La misma figura, con fondo transparente, letra Inter y cuadrícula tenue.
    """
    fig.update_layout(height=alto, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Inter, sans-serif", color="#cfcfd4", size=12),
                      margin=dict(l=10, r=10, t=30, b=10), showlegend=True,
                      legend=dict(orientation="h", y=1.12, x=0, font=dict(size=11)), hovermode="x unified")
    fig.update_xaxes(showgrid=False, linecolor="rgba(255,255,255,.25)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,.07)", zeroline=False)
    return fig


def grafico_proxima(tabla: pd.DataFrame, pred: pd.DataFrame) -> go.Figure:
    """Dibuja las últimas TRM y la próxima predicha, con su rango del 80 %.

    Parámetros
    ----------
    tabla : pd.DataFrame
        Últimas TRM (``fecha``, ``valor``).
    pred : pd.DataFrame
        Resultado de `src.prediccion.predecir`.

    Devuelve
    --------
    plotly.graph_objects.Figure
        Línea de la TRM y punto de la próxima TRM con su rango.
    """
    f = pred.loc[p.MODELO_FINAL]
    sig = tabla["fecha"].iloc[-1] + pd.offsets.BDay(1)
    fig = go.Figure()
    fig.add_scatter(x=tabla["fecha"], y=tabla["valor"], mode="lines", name="TRM publicada", line=dict(color="#f2f2f4", width=1.6))
    fig.add_scatter(x=[tabla["fecha"].iloc[-1], sig], y=[f["trm_hoy"], f["trm_manana"]], mode="lines", showlegend=False,
                    line=dict(color="#f2f2f4", width=1.2, dash="dot"))
    fig.add_scatter(x=[sig], y=[f["trm_manana"]], mode="markers", name="Próxima TRM (modelo final)",
                    marker=dict(color="#f2f2f4", size=9, line=dict(color=e.FONDO, width=2)),
                    error_y=dict(type="data", symmetric=False, array=[f["trm_max_80"] - f["trm_manana"]],
                                 arrayminus=[f["trm_manana"] - f["trm_min_80"]], color="#9a9aa2", thickness=1.2, width=10))
    fig.add_scatter(x=[sig], y=[pred.loc[p.LINEA_BASE, "trm_manana"]], mode="markers", name="Línea base",
                    marker=dict(color="#9a9aa2", size=7, symbol="diamond-open"))
    fig = diseno(fig)
    fig.update_xaxes(tickformat="%d/%m")
    return fig


def grafico_aportes(exp: pd.DataFrame) -> go.Figure:
    """Dibuja la descomposición de la predicción del modelo final (cascada SHAP).

    Parámetros
    ----------
    exp : pd.DataFrame
        Resultado de `src.prediccion.explicar`.

    Devuelve
    --------
    plotly.graph_objects.Figure
        Cascada: punto de partida + aporte de cada movimiento = predicción, en puntos básicos.
    """
    orden = ["base", "r_trm_l10", "r_trm_l5", "r_trm_l1"]
    valores = [exp.loc[v, p.MODELO_FINAL] for v in orden]
    fig = go.Figure(go.Waterfall(
        x=["Base", "Hace 10", "Hace 5", "Hoy", "Predicción"], y=valores + [sum(valores)],
        measure=["relative"] * len(orden) + ["total"], text=[e.num(v, 1, True) + " pb" for v in valores + [sum(valores)]],
        textposition="outside", connector=dict(line=dict(color="rgba(255,255,255,.25)", width=1)),
        increasing=dict(marker=dict(color="rgba(242,242,244,.85)")), decreasing=dict(marker=dict(color="rgba(242,242,244,.30)")),
        totals=dict(marker=dict(color="#9a9aa2"))))
    fig = diseno(fig, 330)
    fig.update_layout(showlegend=False, yaxis_title="puntos básicos (1 pb = 0,01 %)")
    return fig


def grafico_test() -> go.Figure:
    """Compara, en el test (2022–2026), el movimiento real de la TRM con el que predijo el modelo final.

    Devuelve
    --------
    plotly.graph_objects.Figure
        Barras del movimiento real diario y línea del movimiento predicho, en puntos básicos.
    """
    t = pd.read_csv(DATA_PROC / "predicciones_test.csv", parse_dates=["fecha"])
    fig = go.Figure()
    fig.add_bar(x=t["fecha"], y=1e4 * t["y"], name="Movimiento real", marker=dict(color="rgba(255,255,255,.22)"))
    fig.add_scatter(x=t["fecha"], y=1e4 * t[p.MODELO_FINAL], name="Movimiento predicho", mode="lines",
                    line=dict(color="#f2f2f4", width=1.2))
    fig = diseno(fig, 320)
    fig.update_layout(yaxis_title="puntos básicos", bargap=0)
    fig.update_xaxes(tickformat="%m/%Y")
    return fig


# ---------------------------------------------------------------------------
# Bloques reutilizables
# ---------------------------------------------------------------------------
def bloque_semaforo(coherencia: pd.DataFrame) -> str:
    """Construye la tabla del semáforo de coherencia entre los datos de uso y los de entrenamiento.

    Parámetros
    ----------
    coherencia : pd.DataFrame
        Resultado de `src.prediccion.chequear_coherencia`.

    Devuelve
    --------
    str
        HTML de la tabla.
    """
    cuerpo = [[e.punto(f["estado"]) + CHEQUEOS[c], f["estado"].upper(), f["detalle"]] for c, f in coherencia.iterrows()]
    return e.filas(["Chequeo", "Estado", "Detalle"], cuerpo, "1.3fr .6fr 3fr")


def bloque_aportes(exp: pd.DataFrame) -> str:
    """Construye la tabla de aportes de cada variable en los dos modelos.

    Parámetros
    ----------
    exp : pd.DataFrame
        Resultado de `src.prediccion.explicar`.

    Devuelve
    --------
    str
        HTML de la tabla, con barras proporcionales al aporte del modelo final.
    """
    maximo = exp[p.MODELO_FINAL].abs().max()
    cuerpo = []
    for v in ["r_trm_l1", "r_trm_l5", "r_trm_l10", "base"]:
        valor = "—" if v == "base" else e.num(exp.loc[v, "valor_pct"], 2, True) + " %"
        cuerpo.append([NOMBRES[v], valor, e.num(exp.loc[v, p.MODELO_FINAL], 1, True) + " pb",
                       e.barra(exp.loc[v, p.MODELO_FINAL], maximo), e.num(exp.loc[v, p.LINEA_BASE], 1, True) + " pb"])
    return e.filas(["Variable", "Valor", "Aporte · modelo final", "", "Aporte · línea base"], cuerpo, "1.6fr .8fr 1fr 1.4fr 1fr")


def resultado(tabla: pd.DataFrame, r: dict[str, object], hoy: pd.Timestamp, semilla: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Calcula predicción, explicación y coherencia para una tabla de TRM y muestra la tarjeta principal.

    Parámetros
    ----------
    tabla : pd.DataFrame
        Últimas TRM (``fecha``, ``valor``).
    r : dict[str, object]
        Resultado de `recursos`.
    hoy : pd.Timestamp
        Fecha de consulta (para el chequeo de datos recientes).
    semilla : int
        Semilla de la malla decorativa de la tarjeta.

    Devuelve
    --------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Predicción, explicación y coherencia.
    """
    pred = p.predecir(r["modelos"], tabla["valor"], r["error"])
    exp = p.explicar(r["modelos"], tabla["valor"])
    coh = p.chequear_coherencia(tabla, r["referencia"], hoy, r["verificado"])
    f, b = pred.loc[p.MODELO_FINAL], pred.loc[p.LINEA_BASE]
    ultima = tabla["fecha"].iloc[-1]
    izquierda = (e.etiquetas(["Medir", "Analizar", "Predecir", e.cuadritos()])
                 + f'<div class="titulo">TRM de<br>la próxima<br>publicación</div>'
                 + e.kpi(f"$ {e.num(f['trm_manana'])}", f"Modelo final (CatBoost). Rango del 80 %: $ {e.num(f['trm_min_80'], 0)} a "
                                                         f"$ {e.num(f['trm_max_80'], 0)}."))
    derecha = (e.kpi(f"$ {e.num(f['trm_hoy'])}", f"TRM vigente desde el {ultima:%d/%m/%Y}.", chica=True)
               + e.kpi(("+" if f["movimiento_pesos"] >= 0 else "−") + f"$ {e.num(abs(f['movimiento_pesos']))}", f"Movimiento esperado en pesos ({e.num(f['movimiento_pb'], 1, True)} pb).", chica=True)
               + e.kpi(f"$ {e.num(b['trm_manana'])}", "Línea base (regresión lineal), como referencia.", chica=True))
    c1, c2 = st.columns([1.6, 1])
    c1.markdown(e.tarjeta(izquierda, malla=semilla, alto=360), unsafe_allow_html=True)
    c2.markdown(e.tarjeta(derecha, clara=True, alto=360), unsafe_allow_html=True)
    return pred, exp, coh


# ---------------------------------------------------------------------------
# Pantallas
# ---------------------------------------------------------------------------
def pantalla_manana(r: dict[str, object]) -> None:
    """Pantalla 1: la próxima TRM con los datos de hoy.

    Parámetros
    ----------
    r : dict[str, object]
        Resultado de `recursos`.
    """
    tabla, origen, error = trm_actual()
    st.markdown(f'<span class="etiqueta-origen">Datos: {origen}</span>', unsafe_allow_html=True)
    if error:
        st.markdown(f'<p class="nota">La fuente en vivo no respondió ({error}); se usa la copia congelada del proyecto.</p>',
                    unsafe_allow_html=True)
    pred, _, coh = resultado(tabla, r, pd.Timestamp.today(), semilla=3)
    st.plotly_chart(grafico_proxima(tabla, pred), width="stretch")
    malos = coh[coh["estado"] != "ok"]
    resumen = ("Los 5 chequeos de coherencia están en verde." if malos.empty else
               f"{len(malos)} chequeo(s) piden cuidado: " + ", ".join(CHEQUEOS[c].lower() for c in malos.index) + ". Ver la pestaña Confianza.")
    st.markdown(f'<p class="nota">{resumen} La próxima TRM es la siguiente que publique la Superintendencia Financiera '
                f'(después de un viernes, la del lunes o la del martes si es festivo).</p>', unsafe_allow_html=True)


def pantalla_por_que(r: dict[str, object]) -> None:
    """Pantalla 2: por qué el modelo predice lo que predice.

    Parámetros
    ----------
    r : dict[str, object]
        Resultado de `recursos`.
    """
    tabla, _, _ = trm_actual()
    exp = p.explicar(r["modelos"], tabla["valor"])
    pred = p.predecir(r["modelos"], tabla["valor"])
    hoy_pct = exp.loc["r_trm_l1", "valor_pct"]
    aporte = exp.loc["r_trm_l1", p.MODELO_FINAL]
    total = pred.loc[p.MODELO_FINAL, "movimiento_pb"]
    texto = (f"Hoy la TRM se movió <b>{e.num(hoy_pct, 2, True)} %</b>. La TRM es el promedio de las operaciones de un día, "
             f"así que parte del movimiento de hoy suele aparecer también en el de mañana. Por eso el movimiento de hoy aporta "
             f"<b>{e.num(aporte, 1, True)} pb</b> de los <b>{e.num(total, 1, True)} pb</b> que espera el modelo. Los movimientos de "
             f"hace 5 y 10 TRM casi no pesan: la memoria de la TRM es de un día.")
    st.markdown(e.tarjeta(e.etiquetas(["Por qué", e.cuadritos()]) + '<div class="titulo">¿Por qué esa<br>predicción?</div>'
                          + f'<p class="texto">{texto}</p>', malla=7), unsafe_allow_html=True)
    c1, c2 = st.columns([1.1, 1])
    with c1:
        st.markdown(bloque_aportes(exp), unsafe_allow_html=True)
        st.markdown('<p class="nota">Modelo final: valores SHAP (cuánto mueve cada variable la predicción desde el promedio). '
                    'Línea base: aporte exacto de cada coeficiente. En los dos, la suma de los aportes es la predicción. '
                    'El modelo final aprendió además que un movimiento muy grande no se repite en proporción (efecto en forma de S).</p>',
                    unsafe_allow_html=True)
    c2.plotly_chart(grafico_aportes(exp), width="stretch")


def pantalla_confianza(r: dict[str, object]) -> None:
    """Pantalla 3: chequeos de coherencia y desempeño histórico, contado con honestidad.

    Parámetros
    ----------
    r : dict[str, object]
        Resultado de `recursos`.
    """
    tabla, _, _ = trm_actual()
    coh = p.chequear_coherencia(tabla, r["referencia"], pd.Timestamp.today(), r["verificado"])
    st.markdown('<h3>Coherencia entre entrenamiento y uso</h3>', unsafe_allow_html=True)
    st.markdown(bloque_semaforo(coh) + '<div style="height:26px"></div>', unsafe_allow_html=True)
    ev = r["ficha"]["evaluacion"]["test (1.080 días, una vez)"]
    c1, c2, c3 = st.columns(3)
    c1.markdown(e.tarjeta(e.kpi(f"$ {e.num(ev['mae_pesos'])}", "Error típico por día en el test (2022–2026): MAE."), clara=True, alto=210),
                unsafe_allow_html=True)
    c2.markdown(e.tarjeta(e.kpi(e.num(100 * ev["r2_retornos"], 1) + " %", "Del movimiento diario que explica el modelo (R² sobre retornos)."),
                          clara=True, alto=210), unsafe_allow_html=True)
    c3.markdown(e.tarjeta(e.kpi(f"p = {e.num(ev['p_dm_vs_base'])}", "Frente a la línea base (Diebold-Mariano): empate estadístico."),
                          clara=True, alto=210), unsafe_allow_html=True)
    st.plotly_chart(grafico_test(), width="stretch")
    st.markdown('<p class="nota"><b>Cómo leer esto.</b> El modelo predice movimientos pequeños, porque lo único predecible de la TRM '
                'a un día es la continuación del movimiento de hoy. Los movimientos grandes (barras) no se anticipan con el dato de ayer. '
                'Con volatilidad alta, el movimiento de hoy se repite menos (correlación 0,14 frente a 0,23 en periodos tranquilos). '
                'El modelo final empata con una regresión lineal: su valor está en hacerlo con rigor, no en una ventaja grande.</p>',
                unsafe_allow_html=True)


def pantalla_simular(r: dict[str, object]) -> None:
    """Pantalla 4: probar el modelo con una fecha pasada o con TRM escritas a mano.

    Parámetros
    ----------
    r : dict[str, object]
        Resultado de `recursos`.
    """
    copia: pd.DataFrame = r["copia"]
    modo = st.radio("Entrada", ["Una fecha pasada", "Escribir las últimas TRM"], horizontal=True)
    if modo == "Una fecha pasada":
        minimo, maximo = copia["fecha"].iloc[70].date(), copia["fecha"].iloc[-2].date()
        fecha = st.date_input("Fecha", value=pd.Timestamp("2022-11-04").date(), min_value=minimo, max_value=maximo)
        tabla = copia[copia["fecha"] <= pd.Timestamp(fecha)].tail(60).reset_index(drop=True)
        pred, exp, coh = resultado(tabla, r, tabla["fecha"].iloc[-1], semilla=11)
        siguiente = copia[copia["fecha"] > tabla["fecha"].iloc[-1]].head(1)
        if not siguiente.empty:
            real = float(siguiente["valor"].iloc[0])
            f = pred.loc[p.MODELO_FINAL]
            dentro = f["trm_min_80"] <= real <= f["trm_max_80"]
            st.markdown(f'<p class="texto">La TRM que se publicó después fue <b>$ {e.num(real)}</b>: el modelo final se equivocó en '
                        f'<b>{e.num(f["trm_manana"] - real, 2, True)} pesos</b> y la línea base en '
                        f'<b>{e.num(pred.loc[p.LINEA_BASE, "trm_manana"] - real, 2, True)} pesos</b>. '
                        f'{"Cayó dentro" if dentro else "Cayó fuera"} del rango del 80 %.</p>', unsafe_allow_html=True)
    else:
        tabla_vivo, _, _ = trm_actual()
        base = tabla_vivo.tail(p.MIN_HISTORIA)[["fecha", "valor"]].reset_index(drop=True)
        st.markdown('<p class="nota">Cambie cualquier valor (por ejemplo, la TRM de hoy) y la predicción se recalcula.</p>',
                    unsafe_allow_html=True)
        editada = st.data_editor(base, num_rows="fixed", width="stretch", disabled=["fecha"],
                                 column_config={"fecha": st.column_config.DateColumn("Fecha", format="DD/MM/YYYY"),
                                                "valor": st.column_config.NumberColumn("TRM", min_value=1.0, format="%.2f")})
        pred, exp, coh = resultado(editada, r, editada["fecha"].iloc[-1], semilla=13)
    c1, c2 = st.columns([1, 1])
    c1.markdown(bloque_aportes(exp), unsafe_allow_html=True)
    c2.markdown(bloque_semaforo(coh), unsafe_allow_html=True)


def pantalla_modelo(r: dict[str, object]) -> None:
    """Pantalla 5: la ficha del modelo final.

    Parámetros
    ----------
    r : dict[str, object]
        Resultado de `recursos`.
    """
    st.markdown(e.tarjeta(e.etiquetas(["Modelo", "Datos", "Validación", e.cuadritos()])
                          + '<div class="titulo">Ficha del<br>modelo final</div>'
                          + f'<p class="nota">Proyecto Integrador · Machine Learning II · Universidad Externado de Colombia. '
                          f'Código y datos: <a href="{REPO}" style="color:#f2f2f4">{REPO}</a></p>', malla=21), unsafe_allow_html=True)
    st.markdown((MODELOS / "ficha_modelo_final.md").read_text(encoding="utf-8").split("\n", 2)[2])


def main() -> None:
    """Arma la página: estilo, encabezado y las cinco pestañas."""
    st.set_page_config(page_title="TRM · próxima publicación", page_icon="◼", layout="wide")
    st.markdown(e.css(), unsafe_allow_html=True)
    st.markdown(e.encabezado("Proyecto Integrador ML2", "TRM · LightGBM / CatBoost / XGBoost"), unsafe_allow_html=True)
    r = recursos()
    pestanas = st.tabs(["Mañana", "Por qué", "Confianza", "Simular", "Modelo"])
    for pestana, pantalla in zip(pestanas, [pantalla_manana, pantalla_por_que, pantalla_confianza, pantalla_simular, pantalla_modelo]):
        with pestana:
            pantalla(r)


main()
