"""Aplicación del Proyecto Integrador ML2: predicción de la próxima TRM (rúbrica 8.3).

Toma las últimas TRM publicadas (en vivo desde datos.gov.co o, si falla, la copia congelada), calcula
las variables del modelo con las mismas fórmulas del entrenamiento, predice la próxima TRM con el
modelo final (CatBoost · solo TRM) y con la línea base, explica la predicción y verifica que los
datos de uso sean coherentes con los de entrenamiento. Toda la lógica está en `src/prediccion.py`.

Ejecutar desde la carpeta Entrega 2/:
    streamlit run app/app.py
"""
from __future__ import annotations

import importlib
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

# Streamlit Cloud aplica un push volviendo a correr app.py sin reiniciar Python: sin esto seguiría usando
# las versiones viejas de src/prediccion.py y estilo.py que ya tenía en memoria.
e, p = importlib.reload(e), importlib.reload(p)
from src.config import DATA_PROC, MODELOS                            # noqa: E402
from src.descarga import leer                                        # noqa: E402

REPO = "https://github.com/Gatitomalo1234/Proyecto-Machine-Learning-2"
NOMBRES = {"r_trm_l1": "Movimiento de hoy", "r_trm_l5": "Movimiento de hace 5 TRM", "r_trm_l10": "Movimiento de hace 10 TRM",
           "base": "Punto de partida (promedio)"}
SIN_BOTONES = {"displayModeBar": False}                            # sin la barra de botones de Plotly
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
                      margin=dict(l=10, r=10, t=30, b=10), showlegend=True, separators=",.",
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
    if "vigencia_hasta" in tabla:                    # la próxima TRM empieza a regir el día después de que vence la última
        sig = tabla["vigencia_hasta"].iloc[-1] + pd.Timedelta(days=1)
    else:
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
    fig.update_yaxes(tickformat=",.0f")
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


def signo_pesos(valor: float) -> str:
    """Escribe una cantidad de pesos con signo, en formato colombiano (p. ej. "+$ 4,65" o "−$ 0,52").

    Parámetros
    ----------
    valor : float
        Pesos.

    Devuelve
    --------
    str
        Texto con el signo, el símbolo y el número.
    """
    return ("+" if valor >= 0 else "−") + f"$ {e.num(abs(valor))}"


def grafico_pesos(pesos: pd.Series, exp: pd.DataFrame, alto: int = 270) -> go.Figure:
    """Dibuja en barras horizontales cuántos pesos aporta cada movimiento a la predicción del modelo final.

    Parámetros
    ----------
    pesos : pd.Series
        Resultado de `src.prediccion.aportes_en_pesos` (suma = movimiento esperado en pesos).
    exp : pd.DataFrame
        Resultado de `src.prediccion.explicar` (para mostrar el valor de cada movimiento).
    alto : int, por defecto 270
        Altura en píxeles.

    Devuelve
    --------
    plotly.graph_objects.Figure
        Una barra por variable, con su aporte en pesos escrito al lado.
    """
    orden = ["base", "r_trm_l10", "r_trm_l5", "r_trm_l1"]                    # de abajo hacia arriba
    nombres = [NOMBRES[v] + ("" if v == "base" else f" ({e.num(exp.loc[v, 'valor_pct'], 2, True)} %)") for v in orden]
    valores = [float(pesos[v]) for v in orden]
    m = max(abs(v) for v in valores) or 1.0
    adentro = [abs(v) > 0.45 * m for v in valores]                         # barras grandes: el monto va adentro
    fig = go.Figure(go.Bar(x=valores, y=nombres, orientation="h", text=[signo_pesos(v) for v in valores],
                           textposition=["inside" if a else "outside" for a in adentro], insidetextanchor="middle",
                           textfont=dict(color=[e.FONDO if a and v >= 0 else "#f2f2f4" for a, v in zip(adentro, valores)]),
                           cliponaxis=False, hoverinfo="skip",
                           marker=dict(color=["rgba(242,242,244,.85)" if v >= 0 else "rgba(242,242,244,.30)" for v in valores])))
    fig = diseno(fig, alto)
    fig.update_layout(showlegend=False, hovermode=False, margin=dict(l=10, r=60, t=10, b=10))
    fig.update_xaxes(range=[-2.2 * m if min(valores) < 0 else -0.2 * m, 2.2 * m], zeroline=True,
                     zerolinecolor="rgba(255,255,255,.35)", showticklabels=False)
    return fig


def grafico_curva(curva: pd.DataFrame, x_hoy: float, y_hoy: float, visto: tuple[float, float], alto: int = 300) -> go.Figure:
    """Dibuja la curva en S: cuánto se espera que se mueva la TRM mañana según cuánto se movió hoy.

    Parámetros
    ----------
    curva : pd.DataFrame
        Resultado de `src.prediccion.curva_respuesta`.
    x_hoy, y_hoy : float
        Movimiento de hoy y movimiento esperado de mañana (en %), para marcar el punto de hoy.
    visto : tuple[float, float]
        Movimiento de hoy mínimo y máximo (en %) del 99 % central del entrenamiento; lo de afuera se sombrea.
    alto : int, por defecto 300
        Altura en píxeles.

    Devuelve
    --------
    plotly.graph_objects.Figure
        Curva del modelo final, recta de la línea base, el punto de hoy y, si aparece, la zona fuera de lo visto.
    """
    fig = go.Figure()
    for desde, hasta in [(curva["movimiento_hoy_pct"].min(), visto[0]), (visto[1], curva["movimiento_hoy_pct"].max())]:
        if desde < hasta:                                # zona fuera de lo visto en el entrenamiento: el modelo extrapola
            fig.add_vrect(x0=desde, x1=hasta, fillcolor="rgba(184,112,107,.12)", line_width=0,
                          annotation_text="fuera de lo visto", annotation_position="top left",
                          annotation_font=dict(size=10, color="#9a9aa2"))
    fig.add_scatter(x=curva["movimiento_hoy_pct"], y=curva[p.MODELO_FINAL], name="Modelo final", mode="lines",
                    line=dict(color="#f2f2f4", width=2))
    fig.add_scatter(x=curva["movimiento_hoy_pct"], y=curva[p.LINEA_BASE], name="Línea base", mode="lines",
                    line=dict(color="#9a9aa2", width=1.2, dash="dot"))
    fig.add_scatter(x=[x_hoy], y=[y_hoy], name="Hoy", mode="markers+text", text=["Hoy"], textposition="top left",
                    textfont=dict(color="#f2f2f4"), marker=dict(color="#f2f2f4", size=11, line=dict(color=e.FONDO, width=2)))
    fig = diseno(fig, alto)
    fig.update_layout(hovermode="closest", margin=dict(l=70, r=10, t=30, b=10))
    fig.update_xaxes(title="movimiento de hoy", ticksuffix=" %", zeroline=True, zerolinecolor="rgba(255,255,255,.2)")
    fig.update_yaxes(title="movimiento esperado mañana", ticksuffix=" %", zeroline=True, zerolinecolor="rgba(255,255,255,.2)")
    return fig


def grafico_simulacion(tabla: pd.DataFrame, pred: pd.DataFrame, siguiente: pd.DataFrame) -> go.Figure:
    """Dibuja lo que vio el modelo, lo que predijo y lo que pasó, para una fecha pasada.

    Parámetros
    ----------
    tabla : pd.DataFrame
        Las TRM que vio el modelo (``fecha``, ``vigencia_hasta``, ``valor``); se dibujan las últimas 25.
    pred : pd.DataFrame
        Resultado de `src.prediccion.predecir` con su rango del 80 %.
    siguiente : pd.DataFrame
        La TRM que se publicó después (una fila; vacía si no se conoce).

    Devuelve
    --------
    plotly.graph_objects.Figure
        Línea de la historia, predicción con su rango y punto real (verde si cayó en el rango, rojo si no).
    """
    f = pred.loc[p.MODELO_FINAL]
    tabla = tabla.tail(25)                               # las últimas 25 TRM: la predicción no queda apretada en la orilla
    fecha = siguiente["fecha"].iloc[0] if not siguiente.empty else tabla["vigencia_hasta"].iloc[-1] + pd.Timedelta(days=1)
    fig = go.Figure()
    fig.add_scatter(x=tabla["fecha"], y=tabla["valor"], mode="lines", name="Lo que vio", line=dict(color="#9a9aa2", width=1.4))
    fig.add_scatter(x=[tabla["fecha"].iloc[-1], fecha], y=[f["trm_hoy"], f["trm_manana"]], mode="lines", showlegend=False,
                    line=dict(color="#f2f2f4", width=1.2, dash="dot"))
    fig.add_scatter(x=[fecha], y=[f["trm_manana"]], mode="markers", name="Lo que predijo (rango 80 %)",
                    marker=dict(color="#f2f2f4", size=9, symbol="diamond"),
                    error_y=dict(type="data", symmetric=False, array=[f["trm_max_80"] - f["trm_manana"]],
                                 arrayminus=[f["trm_manana"] - f["trm_min_80"]], color="#cfcfd4", thickness=1.4, width=12))
    if not siguiente.empty:
        real = float(siguiente["valor"].iloc[0])
        dentro = f["trm_min_80"] <= real <= f["trm_max_80"]
        fig.add_scatter(x=[fecha], y=[real], mode="markers", name="Lo que pasó",
                        marker=dict(color=e.COLOR_ESTADO["ok" if dentro else "alerta"], size=12, line=dict(color=e.FONDO, width=2)))
    fig = diseno(fig, 340)
    fig.update_xaxes(tickformat="%d/%m/%y")
    fig.update_yaxes(tickformat=",.0f")
    return fig


def grafico_prueba(bt: pd.DataFrame, fecha: pd.Timestamp) -> go.Figure:
    """Dibuja la prueba día a día alrededor de una fecha: rango del 80 %, TRM real y predicción.

    Parámetros
    ----------
    bt : pd.DataFrame
        Resultado de `src.prediccion.prueba_alrededor`.
    fecha : pd.Timestamp
        Fecha elegida (se marca con una línea vertical).

    Devuelve
    --------
    plotly.graph_objects.Figure
        Banda del rango, línea de la TRM real, predicciones y, en rojo, los días fuera del rango.
    """
    fig = go.Figure()
    fig.add_scatter(x=bt["fecha"], y=bt["max_80"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=bt["fecha"], y=bt["min_80"], mode="lines", line=dict(width=0), fill="tonexty",
                    fillcolor="rgba(255,255,255,.12)", name="Rango del 80 %", hoverinfo="skip")
    fig.add_scatter(x=bt["fecha"], y=bt["prediccion"], mode="lines", name="Predicción", line=dict(color="#9a9aa2", width=1.2, dash="dot"))
    fig.add_scatter(x=bt["fecha"], y=bt["real"], mode="lines+markers", name="TRM publicada",
                    line=dict(color="#f2f2f4", width=1.6), marker=dict(size=4))
    fuera = bt[~bt["dentro"]]
    fig.add_scatter(x=fuera["fecha"], y=fuera["real"], mode="markers", name="Fuera del rango",
                    marker=dict(color=e.COLOR_ESTADO["alerta"], size=9))
    fig.add_vline(x=pd.Timestamp(fecha), line=dict(color="rgba(255,255,255,.35)", width=1, dash="dash"))
    fig = diseno(fig, 330)
    fig.update_xaxes(tickformat="%d/%m/%y")
    fig.update_yaxes(tickformat=",.0f")
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


def mostrar_fichas(coherencia: pd.DataFrame) -> None:
    """Muestra el semáforo compacto: 5 fichas de color, el detalle de las que no están en verde y el resto en un desplegable.

    Parámetros
    ----------
    coherencia : pd.DataFrame
        Resultado de `src.prediccion.chequear_coherencia`.
    """
    st.markdown(e.fichas([(CHEQUEOS[c], f["estado"]) for c, f in coherencia.iterrows()]), unsafe_allow_html=True)
    for c, f in coherencia[coherencia["estado"] != "ok"].iterrows():
        st.markdown(f'<p class="nota">{e.punto(f["estado"])}<b>{CHEQUEOS[c]}.</b> {f["detalle"]}</p>', unsafe_allow_html=True)
    with st.expander("Detalle de los 5 chequeos"):
        st.markdown(bloque_semaforo(coherencia), unsafe_allow_html=True)


def parte(fraccion: float) -> str:
    """Escribe una fracción como "la mitad", "un tercio", "una sexta parte", etc.

    Parámetros
    ----------
    fraccion : float
        Número entre 0 y 1.

    Devuelve
    --------
    str
        La fracción en palabras (la más cercana de la forma 1/n).
    """
    n = int(round(1 / fraccion))
    palabras = {2: "la mitad", 3: "un tercio", 4: "una cuarta parte", 5: "una quinta parte", 6: "una sexta parte",
                7: "una séptima parte", 8: "una octava parte", 9: "una novena parte", 10: "una décima parte"}
    return palabras.get(n, "una parte pequeña")


def frase_prediccion(exp: pd.DataFrame, pred: pd.DataFrame) -> str:
    """Resume en una frase el movimiento de hoy y el que espera el modelo para mañana.

    Parámetros
    ----------
    exp : pd.DataFrame
        Resultado de `src.prediccion.explicar`.
    pred : pd.DataFrame
        Resultado de `src.prediccion.predecir` con la misma serie.

    Devuelve
    --------
    str
        Frase en HTML, p. ej. "Hoy la TRM subió 0,83 %. El modelo espera que mañana suba 0,14 % (+$ 4,65):
        repite una sexta parte del movimiento de hoy."
    """
    f = pred.loc[p.MODELO_FINAL]
    hoy = 100 * float(np.expm1(exp.loc["r_trm_l1", "valor_pct"] / 100))
    manana = 100 * float(np.expm1(f["retorno"]))
    texto = (f"Hoy la TRM {'subió' if hoy >= 0 else 'bajó'} <b>{e.num(abs(hoy))} %</b>. El modelo espera que mañana "
             f"{'suba' if manana >= 0 else 'baje'} <b>{e.num(abs(manana))} %</b> (<b>{signo_pesos(f['movimiento_pesos'])}</b>)")
    fraccion = manana / hoy if abs(hoy) >= 0.05 else np.nan
    return texto + (f": repite {parte(fraccion)} del movimiento de hoy." if 0.05 <= fraccion <= 0.6 else ".")


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
                                                         f"$ {e.num(f['trm_max_80'], 0)}"
                                                         + (" (se ensancha con la volatilidad de las últimas 40 TRM)." if f["rango"] == "adaptativo"
                                                            else " (rango fijo: hacen falta 41 TRM para medir la volatilidad).")))
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
    st.plotly_chart(grafico_proxima(tabla, pred), width="stretch", config=SIN_BOTONES)
    malos = coh[coh["estado"] != "ok"]
    resumen = ("Los 5 chequeos de coherencia están en verde." if malos.empty else
               f"{len(malos)} chequeo(s) piden cuidado: " + ", ".join(CHEQUEOS[c].lower() for c in malos.index) + ". Ver la pestaña Confianza.")
    st.markdown(f'<p class="nota">{resumen} La próxima TRM es la siguiente que publique la Superintendencia Financiera. '
                f'La del fin de semana rige desde el sábado hasta el lunes (o hasta el martes si el lunes es festivo).</p>',
                unsafe_allow_html=True)


def pantalla_por_que(r: dict[str, object]) -> None:
    """Pantalla 2: por qué el modelo predice lo que predice, en una frase, dos gráficos y tres ideas.

    Parámetros
    ----------
    r : dict[str, object]
        Resultado de `recursos`.
    """
    tabla, _, _ = trm_actual()
    pred = p.predecir(r["modelos"], tabla["valor"])
    exp = p.explicar(r["modelos"], tabla["valor"])
    st.markdown(e.tarjeta(e.etiquetas(["Por qué", e.cuadritos()]) + '<div class="titulo">¿Por qué esa<br>predicción?</div>'
                          + f'<p class="texto grande">{frase_prediccion(exp, pred)}</p>', malla=7), unsafe_allow_html=True)
    explicacion_visual(r, tabla["valor"], pred, exp)
    principal = max(p.VARS_TRM, key=lambda v: abs(exp.loc[v, p.MODELO_FINAL]))
    st.markdown(e.claves([f"<b>Lo que más pesa:</b> {NOMBRES[principal].lower()}.",
                          "<b>Hace 5 y 10 TRM pesan poco:</b> la memoria de la TRM es de un día.",
                          "<b>Un movimiento muy grande no se repite completo:</b> por eso la curva se aplana en los extremos."]),
                unsafe_allow_html=True)


def explicacion_visual(r: dict[str, object], trm: pd.Series, pred: pd.DataFrame, exp: pd.DataFrame) -> None:
    """Muestra lado a lado los aportes en pesos y la curva en S con el punto de hoy.

    Parámetros
    ----------
    r : dict[str, object]
        Resultado de `recursos`.
    trm : pd.Series
        TRM con las que se predijo.
    pred : pd.DataFrame
        Resultado de `src.prediccion.predecir`.
    exp : pd.DataFrame
        Resultado de `src.prediccion.explicar`.
    """
    x_hoy = 100 * float(np.expm1(exp.loc["r_trm_l1", "valor_pct"] / 100))
    y_hoy = 100 * float(np.expm1(pred.loc[p.MODELO_FINAL, "retorno"]))
    rango = r["referencia"]["rango"]["r_trm_l1"]
    visto = (100 * float(np.expm1(rango.loc[0.005])), 100 * float(np.expm1(rango.loc[0.995])))   # 99 % central del entrenamiento
    curva = p.curva_respuesta(r["modelos"], trm, min(visto[0], x_hoy - 0.3), max(visto[1], x_hoy + 0.3))
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("<h3>Qué empuja la predicción</h3>", unsafe_allow_html=True)
        st.plotly_chart(grafico_pesos(p.aportes_en_pesos(exp, pred), exp), width="stretch", config=SIN_BOTONES)
        st.markdown(f'<p class="nota">Suman {signo_pesos(pred.loc[p.MODELO_FINAL, "movimiento_pesos"])}. La regresión lineal llega '
                    f'a casi lo mismo: {signo_pesos(pred.loc[p.LINEA_BASE, "movimiento_pesos"])}.</p>', unsafe_allow_html=True)
    with c2:
        st.markdown("<h3>Cómo responde el modelo</h3>", unsafe_allow_html=True)
        st.plotly_chart(grafico_curva(curva, x_hoy, y_hoy, visto), width="stretch", config=SIN_BOTONES)
        st.markdown('<p class="nota">Si hoy la TRM se mueve X %, ¿cuánto espera el modelo para mañana? El punto es hoy. La curva cubre el 99 % de los movimientos vistos en el entrenamiento.</p>',
                    unsafe_allow_html=True)


def pantalla_confianza(r: dict[str, object]) -> None:
    """Pantalla 3: chequeos de coherencia, desempeño histórico y la ficha del modelo.

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
    st.plotly_chart(grafico_test(), width="stretch", config=SIN_BOTONES)
    st.markdown('<p class="nota"><b>Cómo leer esto</b><br>'
                '· La línea es pequeña: lo único predecible es que mañana repite una parte del movimiento de hoy.<br>'
                '· Los saltos grandes (barras) no se ven venir.<br>'
                '· En días agitados el rango se ensancha: así contiene la TRM real el 80 % de las veces (con rango fijo, el 67 %).<br>'
                '· Empata con la regresión lineal.</p>',
                unsafe_allow_html=True)
    with st.expander("Ficha del modelo"):
        st.markdown(bloque_ficha(r["ficha"]), unsafe_allow_html=True)


def bloque_ficha(ficha: dict[str, object]) -> str:
    """Construye la ficha corta del modelo final (4 datos y el enlace al repositorio).

    Parámetros
    ----------
    ficha : dict[str, object]
        Contenido de ``models/modelo_final.json``.

    Devuelve
    --------
    str
        HTML de la ficha.
    """
    h = ficha["hiperparametros"]
    ev = ficha["evaluacion"]["test (1.080 días, una vez)"]
    entrenado = ficha["modelo_despliegue"]["entrenado_con"]
    cuerpo = [["Modelo", f"CatBoost con solo la TRM (profundidad {h['depth']}, {h['iterations']} iteraciones, "
                         f"tasa de aprendizaje {e.num(h['learning_rate'])})"],
              ["Variables", "Movimiento de la TRM de hoy, de hace 5 y de hace 10 publicaciones (retornos logarítmicos)"],
              ["Entrenamiento", f"{pd.Timestamp(entrenado['fecha_desde']).year}–{pd.Timestamp(entrenado['fecha_hasta']).year} "
                                f"({e.num(entrenado['dias'], 0)} días), con los hiperparámetros congelados"],
              ["Error", f"MAE de $ {e.num(ev['mae_pesos'])} por día en el test 2022–2026 (evaluado una sola vez); "
                        f"empate con la regresión lineal (p = {e.num(ev['p_dm_vs_base'])})"],
              ["Código", f'<a href="{REPO}" style="color:#f2f2f4">{REPO}</a>']]
    return e.filas(["Dato", "Valor"], cuerpo, "1fr 4fr")


@st.cache_data(show_spinner="Repitiendo la predicción día a día…")
def prueba_cerca(fecha: pd.Timestamp) -> pd.DataFrame:
    """Prueba día a día alrededor de una fecha pasada (en caché por fecha).

    Parámetros
    ----------
    fecha : pd.Timestamp
        Fecha central.

    Devuelve
    --------
    pd.DataFrame
        Resultado de `src.prediccion.prueba_alrededor` sobre la copia congelada.
    """
    r = recursos()
    return p.prueba_alrededor(r["modelos"], r["copia"], fecha, r["error"])


def pantalla_simular(r: dict[str, object]) -> None:
    """Pantalla 4: probar el modelo con una fecha pasada o con un movimiento de hoy a elección.

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
        mostrar_fichas(coh)
        c1, c2 = st.columns([1.35, 1])
        with c1:
            st.markdown("<h3>Lo que vio, lo que predijo, lo que pasó</h3>", unsafe_allow_html=True)
            st.plotly_chart(grafico_simulacion(tabla, pred, siguiente), width="stretch", config=SIN_BOTONES)
        with c2:
            st.markdown("<h3>Qué empujó la predicción</h3>", unsafe_allow_html=True)
            st.plotly_chart(grafico_pesos(p.aportes_en_pesos(exp, pred), exp, alto=340), width="stretch", config=SIN_BOTONES)
        bt = prueba_cerca(tabla["fecha"].iloc[-1])
        st.markdown("<h3>Día a día alrededor de la fecha</h3>", unsafe_allow_html=True)
        st.plotly_chart(grafico_prueba(bt, tabla["fecha"].iloc[-1]), width="stretch", config=SIN_BOTONES)
        st.markdown(f'<p class="nota">El modelo repite su predicción en cada una de las {len(bt)} TRM cercanas, con solo los datos '
                    f'que había ese día. La TRM real cayó dentro del rango del 80 % en {int(bt["dentro"].sum())} de ellas.</p>',
                    unsafe_allow_html=True)
    else:
        tabla_vivo, _, _ = trm_actual()
        historia = tabla_vivo.tail(p.MIN_HISTORIA)[["fecha", "valor"]].reset_index(drop=True)
        real_hoy = 100 * (historia["valor"].iloc[-1] / historia["valor"].iloc[-2] - 1)
        movimiento = st.slider("¿Y si hoy la TRM se mueve…? (%)", -5.0, 5.0, float(np.clip(round(real_hoy, 1), -5, 5)), 0.1, format="%.1f %%")
        with st.expander(f"Editar las {p.MIN_HISTORIA - 1} TRM anteriores"):
            previas = st.data_editor(historia.iloc[:-1], num_rows="fixed", width="stretch", disabled=["fecha"],
                                     column_config={"fecha": st.column_config.DateColumn("Fecha", format="DD/MM/YYYY"),
                                                    "valor": st.column_config.NumberColumn("TRM", min_value=1.0, format="%.2f")})
        if pd.to_numeric(previas["valor"], errors="coerce").isna().any():          # celda borrada: no se puede predecir
            st.markdown(f'<p class="nota">Falta alguna TRM: escriba las {p.MIN_HISTORIA - 1} anteriores para ver la predicción.</p>',
                        unsafe_allow_html=True)
            return
        hoy = pd.DataFrame({"fecha": [historia["fecha"].iloc[-1]], "valor": [float(previas["valor"].iloc[-1]) * (1 + movimiento / 100)]})
        tabla = pd.concat([previas, hoy], ignore_index=True)
        pred, exp, coh = resultado(tabla, r, tabla["fecha"].iloc[-1], semilla=13)
        mostrar_fichas(coh)
        explicacion_visual(r, tabla["valor"], pred, exp)


def main() -> None:
    """Arma la página: estilo, encabezado y las cuatro pestañas."""
    st.set_page_config(page_title="TRM · próxima publicación", page_icon="◼", layout="wide")
    st.markdown(e.css(), unsafe_allow_html=True)
    st.markdown(e.encabezado("Proyecto Integrador ML2", "TRM · CatBoost frente a la línea base"), unsafe_allow_html=True)
    r = recursos()
    pestanas = st.tabs(["Mañana", "Por qué", "Confianza", "Simular"])
    for pestana, pantalla in zip(pestanas, [pantalla_manana, pantalla_por_que, pantalla_confianza, pantalla_simular]):
        with pestana:
            pantalla(r)


main()
