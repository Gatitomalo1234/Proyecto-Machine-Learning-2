"""Descarga de las series crudas desde sus fuentes oficiales y lectura de los archivos crudos.

Principios:
- Cada serie se descarga de su fuente oficial por API (sin copiar y pegar a mano).
- El archivo se guarda en data/raw/ TAL COMO LLEGA; nunca se edita. La limpieza ocurre después.
- Cada descarga se valida leyéndola de nuevo antes de darla por buena.
- Las funciones leer_* convierten un archivo crudo en una tabla (fecha, valor) y se usan en los notebooks.

La descarga la ejecuta scripts/descargar_datos.py una sola vez; los notebooks solo leen la copia
congelada con `leer`. Aparte están los datos del artículo base (`FUENTES_ARTICULO`): los cinco pares
de divisas que el autor publicó, para reproducir su método con sus propios datos (notebook 08b).
"""
from __future__ import annotations

import hashlib
import io
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import pandas as pd
import requests

from .config import DATA_RAW

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Catálogo de fuentes
# ---------------------------------------------------------------------------
_SDMX = ("https://totoro.banrep.gov.co/nsi-jax-ws/rest/data/ESTAT,{flujo},1.0/all/ALL/"
         "?startPeriod=2003&dimensionAtObservation=TIME_PERIOD&detail=full")
_FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={serie}"
_SOCRATA_TRM = ("https://www.datos.gov.co/resource/32sa-8pi3.csv"
                "?$limit=50000&$order=vigenciadesde&$where=vigenciadesde>='2003-01-01'")

FUENTES = {
    "trm": {
        "archivo": "trm_banrep.xml",
        "url": _SDMX.format(flujo="DF_TRM_DAILY_HIST"),
        "descripcion": "TRM diaria (COP por USD), Banco de la República, servicio web SDMX, flujo DF_TRM_DAILY_HIST",
    },
    "trm_vigencias": {
        "archivo": "trm_vigencias_datosgov.csv",
        "url": _SOCRATA_TRM,
        "descripcion": "TRM con fechas de vigencia (una fila por TRM publicada), Superfinanciera en datos.gov.co, conjunto 32sa-8pi3",
    },
    "tpm": {
        "archivo": "tpm_banrep.xml",
        "url": _SDMX.format(flujo="DF_CBR_DAILY_HIST"),
        "descripcion": "Tasa de política monetaria vigente al día (%), Banco de la República, servicio web SDMX, flujo DF_CBR_DAILY_HIST",
    },
    "brent": {
        "archivo": "brent_fred.csv",
        "url": _FRED.format(serie="DCOILBRENTEU"),
        "descripcion": "Petróleo Brent, USD por barril, FRED serie DCOILBRENTEU",
    },
    "dff": {
        "archivo": "dff_fred.csv",
        "url": _FRED.format(serie="DFF"),
        "descripcion": "Tasa efectiva de fondos federales (%), FRED serie DFF",
    },
    "dxy": {
        "archivo": "dxy_yahoo.csv",
        "url": "yfinance: DX-Y.NYB",
        "descripcion": "Índice del dólar (DXY), cierre diario, Yahoo Finance DX-Y.NYB vía yfinance",
    },
}


# ---------------------------------------------------------------------------
# Descarga
# ---------------------------------------------------------------------------
def _get(url: str, intentos: int = 4, espera: float = 5, timeout: float = 120) -> bytes:
    """Hace un GET con reintentos y espera creciente ante fallas de red o del servidor.

    Parámetros
    ----------
    url : str
        Dirección que se descarga.
    intentos : int, por defecto 4
        Número máximo de intentos.
    espera : float, por defecto 5
        Segundos base de espera; tras el intento i se espera ``espera * i`` (5, 10, 15 y 20 s).
    timeout : float, por defecto 120
        Segundos máximos de cada intento.

    Devuelve
    --------
    bytes
        El contenido de la respuesta, sin ninguna transformación.

    Lanza
    -----
    RuntimeError
        Si fallan todos los intentos; encadena el último error de ``requests`` como causa.
    """
    ultimo_error = None
    for i in range(1, intentos + 1):
        try:
            r = requests.get(url, timeout=timeout, headers={"User-Agent": "proyecto-trm-ml2/1.0"})
            r.raise_for_status()
            return r.content
        except requests.RequestException as e:
            ultimo_error = e
            print(f"  intento {i}/{intentos} falló: {e}")
            time.sleep(espera * i)
    raise RuntimeError(f"No se pudo descargar {url}") from ultimo_error


def _bajar_dxy() -> bytes:
    """Descarga el índice del dólar (DXY) diario desde 2003 con yfinance.

    El DXY no tiene una API oficial gratuita; Yahoo Finance (ticker DX-Y.NYB) es la fuente de la
    Entrega 1. Se piden precios sin ajustar (``auto_adjust=False``) y se quita el nivel del ticker
    que yfinance agrega a las columnas.

    Devuelve
    --------
    bytes
        Tabla OHLC completa en formato CSV (UTF-8), con la columna de fecha llamada ``fecha``.

    Lanza
    -----
    RuntimeError
        Si yfinance devuelve una tabla vacía.
    """
    import yfinance as yf
    d = yf.download("DX-Y.NYB", start="2003-01-01", progress=False, auto_adjust=False, threads=False)
    if d.empty:
        raise RuntimeError("yfinance devolvió una tabla vacía para DX-Y.NYB")
    d.columns = [c[0] if isinstance(c, tuple) else c for c in d.columns]   # quita el nivel del ticker
    d = d.reset_index().rename(columns={"Date": "fecha"})
    return d.to_csv(index=False).encode("utf-8")


def descargar(nombre: str) -> bytes:
    """Descarga una serie de `FUENTES` y devuelve sus bytes crudos.

    Parámetros
    ----------
    nombre : str
        Clave de `FUENTES`: "trm", "trm_vigencias", "tpm", "brent", "dff" o "dxy". El DXY se
        descarga con yfinance; las demás, por su API con `_get`.

    Devuelve
    --------
    bytes
        El archivo tal como lo entrega la fuente.
    """
    if nombre == "dxy":
        return _bajar_dxy()
    return _get(FUENTES[nombre]["url"])


def guardar(nombre: str, contenido: bytes) -> dict[str, str | int]:
    """Escribe el archivo crudo en data/raw/, lo valida leyéndolo y devuelve sus metadatos.

    Parámetros
    ----------
    nombre : str
        Clave de `FUENTES`; define el nombre del archivo y el lector con que se valida.
    contenido : bytes
        Resultado de `descargar`. Se escribe sin modificar y reemplaza el archivo si ya existía.

    Devuelve
    --------
    dict[str, str | int]
        La entrada de ``metadatos.json`` para esta serie: ``archivo``, ``fuente``, ``url``,
        ``descargado_utc``, ``md5``, ``bytes``, ``filas``, ``fecha_min`` y ``fecha_max``.

    Notas
    -----
    Si el archivo no se puede leer, el error del lector detiene la descarga: así no queda guardada
    una copia dañada como si fuera buena.
    """
    ruta = DATA_RAW / FUENTES[nombre]["archivo"]
    ruta.write_bytes(contenido)
    tabla = LECTORES[nombre](ruta)
    return {
        "archivo": ruta.name,
        "fuente": FUENTES[nombre]["descripcion"],
        "url": FUENTES[nombre]["url"],
        "descargado_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "md5": md5(ruta),
        "bytes": ruta.stat().st_size,
        "filas": int(len(tabla)),
        "fecha_min": str(tabla["fecha"].min().date()),
        "fecha_max": str(tabla["fecha"].max().date()),
    }


def md5(ruta: Path) -> str:
    """Calcula la huella MD5 de un archivo.

    Sirve para detectar si la copia congelada de los datos crudos fue modificada.

    Parámetros
    ----------
    ruta : Path
        Archivo del que se calcula la huella.

    Devuelve
    --------
    str
        Huella MD5 en hexadecimal (32 caracteres).
    """
    return hashlib.md5(ruta.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Lectura de los archivos crudos (sin limpieza: solo formato)
# ---------------------------------------------------------------------------
_NS = {"g": "http://www.sdmx.org/resources/sdmxml/schemas/v2_1/data/generic"}   # espacio de nombres del XML SDMX genérico


def leer_sdmx(ruta: Path) -> pd.DataFrame:
    """Convierte un XML SDMX del Banrep (TRM o TPM) en una tabla (fecha, valor).

    Parámetros
    ----------
    ruta : Path
        Archivo XML en formato SDMX genérico, con una sola serie.

    Devuelve
    --------
    pd.DataFrame
        Columnas ``fecha`` (leída con el formato AAAAMMDD) y ``valor`` (número; lo que no se pueda
        convertir queda como NaN), ordenada por fecha.

    Lanza
    -----
    ValueError
        Si el archivo no trae exactamente una serie.
    """
    series = ET.parse(ruta).getroot().findall(".//g:Series", _NS)
    if len(series) != 1:
        raise ValueError(f"{ruta.name}: se esperaba 1 serie y llegaron {len(series)}")
    filas = [(o.find("g:ObsDimension", _NS).get("value"), o.find("g:ObsValue", _NS).get("value"))
             for o in series[0].findall("g:Obs", _NS)]
    df = pd.DataFrame(filas, columns=["fecha", "valor"])
    df["fecha"] = pd.to_datetime(df["fecha"], format="%Y%m%d")
    df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
    return df.sort_values("fecha").reset_index(drop=True)


def leer_fred(ruta: Path) -> pd.DataFrame:
    """Convierte un CSV de FRED (Brent o DFF) en una tabla (fecha, valor).

    Parámetros
    ----------
    ruta : Path
        CSV descargado de ``fredgraph.csv``, con dos columnas (fecha y serie).

    Devuelve
    --------
    pd.DataFrame
        Columnas ``fecha`` y ``valor``. Los "." que FRED usa para los días sin dato quedan como NaN.
    """
    df = pd.read_csv(ruta, na_values=".")
    df.columns = ["fecha", "valor"]
    df["fecha"] = pd.to_datetime(df["fecha"])
    return df


def leer_vigencias(ruta: Path) -> pd.DataFrame:
    """Convierte el CSV de la Superfinanciera (datos.gov.co) en una tabla de TRM con vigencias.

    Este archivo trae una fila por TRM publicada, con el día en que empieza y termina a regir; de él
    salen las fechas de la TRM (la que empieza a regir el sábado vale hasta el lunes, o hasta el
    martes si el lunes es festivo).

    Parámetros
    ----------
    ruta : Path
        CSV del conjunto 32sa-8pi3, con columnas ``vigenciadesde``, ``vigenciahasta`` y ``valor``.

    Devuelve
    --------
    pd.DataFrame
        Columnas ``fecha`` (vigencia desde), ``vigencia_hasta`` y ``valor``, ordenada por fecha.
    """
    df = pd.read_csv(ruta, parse_dates=["vigenciadesde", "vigenciahasta"])
    return (df.rename(columns={"vigenciadesde": "fecha", "vigenciahasta": "vigencia_hasta"})
              [["fecha", "vigencia_hasta", "valor"]].sort_values("fecha").reset_index(drop=True))


def leer_dxy(ruta: Path) -> pd.DataFrame:
    """Convierte el CSV guardado desde yfinance en una tabla (fecha, valor = cierre).

    Parámetros
    ----------
    ruta : Path
        CSV escrito por `_bajar_dxy`.

    Devuelve
    --------
    pd.DataFrame
        Columnas ``fecha`` y ``valor`` (precio de cierre sin ajustar).
    """
    df = pd.read_csv(ruta, parse_dates=["fecha"])
    return df.rename(columns={"Close": "valor"})[["fecha", "valor"]]


LECTORES = {"trm": leer_sdmx, "tpm": leer_sdmx, "trm_vigencias": leer_vigencias,
            "brent": leer_fred, "dff": leer_fred, "dxy": leer_dxy}   # lector de cada archivo crudo


def leer(nombre: str) -> pd.DataFrame:
    """Lee la copia congelada de una serie desde data/raw/.

    Parámetros
    ----------
    nombre : str
        Clave de `FUENTES` ("trm", "trm_vigencias", "tpm", "brent", "dff" o "dxy").

    Devuelve
    --------
    pd.DataFrame
        La serie con el formato de su lector (ver `leer_sdmx`, `leer_fred`, `leer_vigencias` y `leer_dxy`).
    """
    return LECTORES[nombre](DATA_RAW / FUENTES[nombre]["archivo"])


# ---------------------------------------------------------------------------
# Datos del artículo base (Eslami Ghayour y Yousefi, 2023), para el notebook 08b
# ---------------------------------------------------------------------------
_REPO_ARTICULO = ("https://raw.githubusercontent.com/arasheslamii/"
                  "1-Performance-comparison-of-Gradient-Boosting-algorithms-for-Forex-market-Prediction-A-Study-on-Lig/"
                  "506912ceffde3cf998bbaf8ba27ad31572a31a93/{archivo}")   # fijado al commit del 2023-05-14
ARTICULO_RAW = DATA_RAW / "articulo"                                    # copia congelada de esos archivos

FUENTES_ARTICULO = {
    par: {"archivo": archivo, "url": _REPO_ARTICULO.format(archivo=archivo),
          "descripcion": f"{par}, precio de cierre diario; repositorio público del autor del artículo base"}
    for par, archivo in [("GBP/USD", "GU.csv"), ("EUR/USD", "EU.csv"), ("AUD/USD", "AU.csv"),
                         ("NZD/USD", "NU.csv"), ("USD/CHF", "UC.csv")]
}   # EU-C.csv del mismo repositorio es idéntico a EU.csv y no se usa


def descargar_articulo(nombre: str) -> bytes:
    """Descarga un par de divisas del repositorio del autor del artículo base.

    Parámetros
    ----------
    nombre : str
        Clave de `FUENTES_ARTICULO`: "GBP/USD", "EUR/USD", "AUD/USD", "NZD/USD" o "USD/CHF".

    Devuelve
    --------
    bytes
        El CSV tal como está en el repositorio, en la versión fijada por el commit de `_REPO_ARTICULO`.
    """
    return _get(FUENTES_ARTICULO[nombre]["url"])


def leer_csv_articulo(ruta: Path) -> pd.DataFrame:
    """Convierte un CSV del repositorio del autor en una tabla (fecha, valor).

    Los archivos traen las columnas ``Date`` (mes/día/año) y ``Close``; el de EUR/USD trae solo
    ``Close``. Se conserva el orden del archivo, que es el cronológico.

    Parámetros
    ----------
    ruta : Path
        Archivo CSV del repositorio del autor.

    Devuelve
    --------
    pd.DataFrame
        Columnas ``fecha`` (NaT si el archivo no trae fechas) y ``valor`` (precio de cierre).

    Lanza
    -----
    ValueError
        Si el archivo no tiene la columna ``Close``.
    """
    df = pd.read_csv(ruta)
    if "Close" not in df.columns:
        raise ValueError(f"{ruta.name}: falta la columna Close")
    fechas = pd.to_datetime(df["Date"], format="%m/%d/%Y") if "Date" in df.columns else pd.NaT
    return pd.DataFrame({"fecha": fechas, "valor": pd.to_numeric(df["Close"], errors="coerce")})


def guardar_articulo(nombre: str, contenido: bytes) -> dict[str, str | int | None]:
    """Escribe un CSV del artículo en data/raw/articulo/, lo valida leyéndolo y devuelve sus metadatos.

    Parámetros
    ----------
    nombre : str
        Clave de `FUENTES_ARTICULO`.
    contenido : bytes
        Resultado de `descargar_articulo`; se escribe sin modificar.

    Devuelve
    --------
    dict[str, str | int | None]
        ``archivo``, ``fuente``, ``url``, ``descargado_utc``, ``md5``, ``bytes``, ``filas``,
        ``fecha_min`` y ``fecha_max`` (None si el archivo no trae fechas).
    """
    ARTICULO_RAW.mkdir(parents=True, exist_ok=True)
    ruta = ARTICULO_RAW / FUENTES_ARTICULO[nombre]["archivo"]
    ruta.write_bytes(contenido)
    tabla = leer_csv_articulo(ruta)
    con_fecha = tabla["fecha"].notna().any()
    return {
        "archivo": ruta.name,
        "fuente": FUENTES_ARTICULO[nombre]["descripcion"],
        "url": FUENTES_ARTICULO[nombre]["url"],
        "descargado_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "md5": md5(ruta),
        "bytes": ruta.stat().st_size,
        "filas": int(len(tabla)),
        "fecha_min": str(tabla["fecha"].min().date()) if con_fecha else None,
        "fecha_max": str(tabla["fecha"].max().date()) if con_fecha else None,
    }


def leer_articulo(nombre: str) -> pd.DataFrame:
    """Lee la copia congelada de un par de divisas del artículo desde data/raw/articulo/.

    Parámetros
    ----------
    nombre : str
        Clave de `FUENTES_ARTICULO`.

    Devuelve
    --------
    pd.DataFrame
        Columnas ``fecha`` y ``valor`` (ver `leer_csv_articulo`).
    """
    return leer_csv_articulo(ARTICULO_RAW / FUENTES_ARTICULO[nombre]["archivo"])


# ---------------------------------------------------------------------------
# TRM en vivo (aplicación, rúbrica 8.3)
# ---------------------------------------------------------------------------
_SOCRATA_TRM_RECIENTE = ("https://www.datos.gov.co/resource/32sa-8pi3.json"
                         "?$order=vigenciadesde%20DESC&$limit={n}")   # misma fuente y formato que la copia congelada


def trm_reciente(n: int = 60, intentos: int = 2, timeout: float = 15) -> pd.DataFrame:
    """Descarga las últimas TRM publicadas desde datos.gov.co (Superintendencia Financiera).

    Es la misma fuente y el mismo formato (una fila por TRM publicada, con su vigencia) que la copia
    congelada `trm_vigencias` con que se construyó el conjunto de entrenamiento, así que las variables
    que se calculan con ella son idénticas a las del entrenamiento.

    Parámetros
    ----------
    n : int, por defecto 60
        Número de TRM recientes que se piden (la app necesita al menos 11 para los rezagos y 21
        para la volatilidad de 20 días).
    intentos : int, por defecto 2
        Intentos antes de rendirse; pocos, para que la app no se quede esperando.
    timeout : float, por defecto 15
        Segundos máximos de cada intento.

    Devuelve
    --------
    pd.DataFrame
        Columnas ``fecha`` (vigencia desde), ``vigencia_hasta`` y ``valor``, en orden cronológico.

    Lanza
    -----
    RuntimeError
        Si la fuente no responde (lo lanza `_get`) o si la respuesta no trae las columnas esperadas.
    """
    import json
    filas = json.loads(_get(_SOCRATA_TRM_RECIENTE.format(n=n), intentos=intentos, espera=1, timeout=timeout))
    df = pd.DataFrame(filas)
    if not {"vigenciadesde", "vigenciahasta", "valor"} <= set(df.columns):
        raise RuntimeError("datos.gov.co devolvió una respuesta sin las columnas esperadas")
    df = pd.DataFrame({"fecha": pd.to_datetime(df["vigenciadesde"]).dt.normalize(),
                       "vigencia_hasta": pd.to_datetime(df["vigenciahasta"]).dt.normalize(),
                       "valor": pd.to_numeric(df["valor"])})
    return df.sort_values("fecha").reset_index(drop=True)
