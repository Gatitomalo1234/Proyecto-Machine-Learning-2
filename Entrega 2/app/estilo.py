"""Estilo visual de la aplicación: tema oscuro monocromo, tipografía ancha y mallas de alambre.

Traduce la referencia de diseño elegida por el usuario (tarjetas oscuras en degradé de grafito, títulos
en mayúsculas con una fuente ancha, cifras grandes con una línea fina debajo, etiquetas pequeñas y
espaciadas, líneas de 1 px, mallas de alambre y un ícono de cuadritos). Todo se genera aquí, sin
imágenes externas: las mallas son SVG calculados con numpy.

Las funciones devuelven fragmentos de HTML que `app.py` muestra con ``st.markdown(..., unsafe_allow_html=True)``.
"""
from __future__ import annotations

import numpy as np

FONDO, TEXTO, TENUE, LINEA = "#0b0b0c", "#f2f2f4", "#9a9aa2", "rgba(255,255,255,0.16)"
COLOR_ESTADO = {"ok": "#7fa58a", "aviso": "#c9a95c", "alerta": "#b8706b"}   # único toque de color, apagado


def css() -> str:
    """Devuelve la hoja de estilos de la aplicación.

    Define el fondo, las fuentes (Michroma para títulos y cifras; Inter para el texto), las pestañas
    en mayúsculas pequeñas, las tarjetas en degradé, los indicadores y las filas de tabla.

    Devuelve
    --------
    str
        Bloque ``<style>`` listo para ``st.markdown``.
    """
    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Michroma&family=Inter:wght@300;400;500&display=swap');
.stApp {{ background: radial-gradient(circle at 15% -10%, #1d1d21 0%, {FONDO} 55%); color: {TEXTO}; }}
html, body, .stApp, .stMarkdown, p, label, li {{ font-family: 'Inter', sans-serif; font-weight: 300; }}
h2, h3 {{ font-size: 20px !important; margin: 18px 0 12px 0 !important; }}
h1, h2, h3, .titulo {{ font-family: 'Michroma', sans-serif !important; text-transform: uppercase;
    letter-spacing: .05em; color: {TEXTO}; font-weight: 400 !important; }}
header[data-testid="stHeader"] {{ background: transparent; }}
#MainMenu, footer {{ visibility: hidden; }}
.block-container {{ max-width: 1180px; padding-top: 2.2rem; }}
[role="tablist"] {{ gap: 2.6rem; border-bottom: 1px solid {LINEA}; }}
[data-testid="stTab"] p {{ font-family: 'Inter', sans-serif !important; font-size: 11px !important; letter-spacing: .28em;
    text-transform: uppercase; color: {TENUE}; font-weight: 400; }}
[data-testid="stTab"][aria-selected="true"] p {{ color: {TEXTO}; }}
.stAppDeployButton, [data-testid="stAppDeployButton"], [data-testid="stToolbar"] {{ display: none !important; }}
.encabezado {{ display: flex; align-items: center; gap: 18px; color: #cfcfd4; font-size: 13px;
    letter-spacing: .04em; margin-bottom: 1.4rem; }}
.encabezado .raya {{ flex: 1; border-top: 1px solid {LINEA}; }}
.etiquetas {{ display: flex; justify-content: space-between; font-size: 10px; letter-spacing: .22em;
    text-transform: uppercase; color: {TENUE}; margin-bottom: 18px; position: relative; z-index: 2; }}
.tarjeta {{ position: relative; overflow: hidden; border: 1px solid {LINEA}; padding: 26px 30px;
    background: linear-gradient(135deg, #121214 0%, #1e1e22 55%, #34343a 100%); margin-bottom: 18px; }}
.tarjeta.clara {{ background: linear-gradient(160deg, #2a2a2f 0%, #4a4a50 100%); }}
.tarjeta .fondo {{ position: absolute; inset: 0; opacity: .38; pointer-events: none; }}
.tarjeta .fondo svg {{ position: absolute; right: -110px; top: -20px; width: 62%; height: auto; }}
.tarjeta .contenido {{ position: relative; z-index: 2; }}
.tarjeta .titulo {{ font-size: 30px; line-height: 1.25; margin: 6px 0 10px 0; }}
.kpi {{ margin: 6px 0 14px 0; }}
.kpi .cifra {{ font-family: 'Michroma', sans-serif; font-size: 38px; color: {TEXTO}; line-height: 1.15; }}
.kpi .cifra.chica {{ font-size: 22px; }}
.kpi .raya {{ border-top: 1px solid {LINEA}; margin: 10px 0 8px 0; width: 85%; }}
.kpi .desc {{ color: #c8c8ce; font-size: 13px; line-height: 1.45; }}
.nota {{ color: {TENUE}; font-size: 12.5px; line-height: 1.55; }}
.texto {{ color: #d9d9de; font-size: 15px; line-height: 1.65; }}
.fila {{ display: grid; align-items: center; border-top: 1px solid {LINEA}; padding: 11px 2px; font-size: 13.5px; color: #dcdce0; }}
.fila.cabeza {{ font-size: 10px; letter-spacing: .2em; text-transform: uppercase; color: {TENUE}; border-top: none; }}
.barra {{ height: 6px; background: rgba(255,255,255,.55); }}
.barra.neg {{ background: rgba(255,255,255,.22); }}
.punto {{ display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 10px; }}
.etiqueta-origen {{ display: inline-block; border: 1px solid {LINEA}; padding: 3px 10px; font-size: 10px;
    letter-spacing: .2em; text-transform: uppercase; color: #cfcfd4; }}
div[data-testid="stMetric"] {{ background: transparent; }}
[data-testid="stWidgetLabel"] p {{ font-size: 11px !important; letter-spacing: .2em; text-transform: uppercase; color: {TENUE}; }}
</style>
"""


def malla_svg(semilla: int = 0, ancho: int = 640, alto: int = 420, lineas: int = 28) -> str:
    """Dibuja una malla de alambre ondulada (superficie 3D en perspectiva) como SVG.

    Es el elemento decorativo de la referencia: una superficie z = f(u, v) con ondas suaves, rotada y
    proyectada en 2D, dibujada con líneas finas blancas semitransparentes.

    Parámetros
    ----------
    semilla : int, por defecto 0
        Cambia la forma de las ondas, para que cada tarjeta tenga una malla distinta.
    ancho, alto : int, por defecto 640 y 420
        Tamaño del lienzo SVG en píxeles.
    lineas : int, por defecto 28
        Número de líneas en cada dirección.

    Devuelve
    --------
    str
        Código SVG completo.
    """
    rng = np.random.default_rng(semilla)
    fa, fb, fase = rng.uniform(1.5, 3.2), rng.uniform(1.2, 2.6), rng.uniform(0, np.pi)
    u, v = np.meshgrid(np.linspace(-1.3, 1.3, lineas), np.linspace(-1.3, 1.3, lineas))
    z = 0.42 * np.sin(fa * u + fase) * np.cos(fb * v) * np.exp(-0.45 * (u ** 2 + v ** 2)) + 0.18 * np.sin(2.2 * v + u)
    a, b = np.radians(58), np.radians(28 + 20 * rng.uniform())
    x1, y1 = u * np.cos(b) - v * np.sin(b), u * np.sin(b) + v * np.cos(b)
    y2, z2 = y1 * np.cos(a) - z * np.sin(a), y1 * np.sin(a) + z * np.cos(a)
    persp = 3.2 / (3.2 + z2)
    px, py = ancho / 2 + 175 * x1 * persp, alto / 2 + 175 * y2 * persp

    def trazo(xs: np.ndarray, ys: np.ndarray) -> str:
        """Convierte una línea de la malla en una polilínea SVG.

        Parámetros
        ----------
        xs, ys : np.ndarray
            Coordenadas en píxeles de los puntos de la línea.

        Devuelve
        --------
        str
            Elemento ``<polyline>``.
        """
        puntos = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
        return f'<polyline points="{puntos}" fill="none" stroke="#ffffff" stroke-opacity="0.42" stroke-width="0.6"/>'

    trazos = [trazo(px[i], py[i]) for i in range(lineas)] + [trazo(px[:, j], py[:, j]) for j in range(lineas)]
    circulos = (f'<circle cx="{ancho * 0.42:.0f}" cy="{alto * 0.48:.0f}" r="{alto * 0.46:.0f}" fill="none" '
                f'stroke="#ffffff" stroke-opacity="0.14" stroke-width="0.8"/>')
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{ancho}" height="{alto}" viewBox="0 0 {ancho} {alto}">'
            + circulos + "".join(trazos) + "</svg>")


def cuadritos() -> str:
    """Devuelve el ícono de cuadritos alternados de la referencia, como SVG en línea.

    Devuelve
    --------
    str
        SVG de 18 × 10 píxeles.
    """
    celdas = "".join(f'<rect x="{3 * i}" y="{5 * j}" width="3" height="5" fill="#fff" fill-opacity="{0.9 if (i + j) % 2 == 0 else 0.25}"/>'
                     for i in range(6) for j in range(2))
    return f'<svg width="18" height="10" viewBox="0 0 18 10" xmlns="http://www.w3.org/2000/svg">{celdas}</svg>'


def encabezado(izquierda: str, derecha: str) -> str:
    """Construye el encabezado "texto ——— texto" de la referencia.

    Parámetros
    ----------
    izquierda, derecha : str
        Textos a cada lado de la línea.

    Devuelve
    --------
    str
        HTML del encabezado.
    """
    return f'<div class="encabezado"><span>{izquierda}</span><span class="raya"></span><span>{derecha}</span></div>'


def etiquetas(textos: list[str]) -> str:
    """Construye la fila de etiquetas pequeñas y espaciadas de la parte superior de una tarjeta.

    Parámetros
    ----------
    textos : list[str]
        Etiquetas, de izquierda a derecha.

    Devuelve
    --------
    str
        HTML de la fila.
    """
    return '<div class="etiquetas">' + "".join(f"<span>{t}</span>" for t in textos) + "</div>"


def tarjeta(contenido: str, malla: int | None = None, clara: bool = False, alto: int | None = None) -> str:
    """Envuelve contenido en una tarjeta oscura con degradé y, opcionalmente, una malla de fondo.

    Parámetros
    ----------
    contenido : str
        HTML interior.
    malla : int o None, por defecto None
        Semilla de la malla de alambre de fondo; None para no dibujarla.
    clara : bool, por defecto False
        Usa el degradé gris más claro (como los paneles de cifras de la referencia).
    alto : int o None, por defecto None
        Altura mínima en píxeles.

    Devuelve
    --------
    str
        HTML de la tarjeta.
    """
    fondo = f'<div class="fondo">{malla_svg(malla)}</div>' if malla is not None else ""   # SVG en línea: Streamlit filtra los fondos data:
    estilo_alto = f' style="min-height:{alto}px"' if alto else ""
    return f'<div class="tarjeta{" clara" if clara else ""}"{estilo_alto}>{fondo}<div class="contenido">{contenido}</div></div>'


def kpi(cifra: str, descripcion: str, chica: bool = False) -> str:
    """Construye un indicador: cifra grande, línea fina y una descripción corta.

    Parámetros
    ----------
    cifra : str
        Texto de la cifra (ya formateado).
    descripcion : str
        Qué significa la cifra.
    chica : bool, por defecto False
        Usa un tamaño de cifra menor.

    Devuelve
    --------
    str
        HTML del indicador.
    """
    return (f'<div class="kpi"><div class="cifra{" chica" if chica else ""}">{cifra}</div>'
            f'<div class="raya"></div><div class="desc">{descripcion}</div></div>')


def punto(estado: str) -> str:
    """Devuelve el punto de color del semáforo para un estado.

    Parámetros
    ----------
    estado : str
        "ok", "aviso" o "alerta".

    Devuelve
    --------
    str
        HTML del punto.
    """
    return f'<span class="punto" style="background:{COLOR_ESTADO.get(estado, TENUE)}"></span>'


def filas(cabeza: list[str], cuerpo: list[list[str]], columnas: str) -> str:
    """Construye una tabla de filas separadas por líneas finas (como la tercera tarjeta de la referencia).

    Parámetros
    ----------
    cabeza : list[str]
        Títulos de las columnas.
    cuerpo : list[list[str]]
        Celdas (HTML) de cada fila.
    columnas : str
        Plantilla CSS de las columnas (p. ej. ``"2fr 1fr 3fr"``).

    Devuelve
    --------
    str
        HTML de la tabla.
    """
    def fila(celdas: list[str], clase: str = "") -> str:
        """Construye una fila de la tabla.

        Parámetros
        ----------
        celdas : list[str]
            HTML de cada celda.
        clase : str, por defecto ""
            Clase CSS adicional ("cabeza" para los títulos).

        Devuelve
        --------
        str
            HTML de la fila.
        """
        return f'<div class="fila {clase}" style="grid-template-columns:{columnas}">' + "".join(f"<div>{c}</div>" for c in celdas) + "</div>"
    return fila(cabeza, "cabeza") + "".join(fila(c) for c in cuerpo)


def barra(valor: float, maximo: float) -> str:
    """Dibuja una barra horizontal proporcional al valor absoluto (gris claro si es positivo, tenue si es negativo).

    Parámetros
    ----------
    valor : float
        Valor que se representa.
    maximo : float
        Valor absoluto que ocupa el ancho completo.

    Devuelve
    --------
    str
        HTML de la barra.
    """
    ancho = 0 if maximo == 0 else min(85, 85 * abs(valor) / maximo)
    return f'<div class="barra{" neg" if valor < 0 else ""}" style="width:{ancho:.0f}%"></div>'


def num(x: float, decimales: int = 2, signo: bool = False) -> str:
    """Escribe un número en formato colombiano: punto de miles y coma decimal.

    Parámetros
    ----------
    x : float
        Número.
    decimales : int, por defecto 2
        Decimales que se muestran.
    signo : bool, por defecto False
        Si es True, antepone "+" a los positivos.

    Devuelve
    --------
    str
        El número formateado (p. ej. 3269.88 → "3.269,88").
    """
    s = f"{x:{'+' if signo else ''},.{decimales}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")
