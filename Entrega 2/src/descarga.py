"""Descarga de las series crudas desde sus fuentes oficiales y lectura de los archivos crudos.

Principios:
- Cada serie se descarga de su fuente oficial por API (sin copiar y pegar a mano).
- El archivo se guarda en data/raw/ TAL COMO LLEGA; nunca se edita. La limpieza ocurre después.
- Cada descarga se valida leyéndola de nuevo antes de darla por buena.
- Las funciones leer_* convierten un archivo crudo en una tabla (fecha, valor) y se usan en los notebooks.
"""
import hashlib
import io
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

import pandas as pd
import requests

from .config import DATA_RAW

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
def _get(url, intentos=4, espera=5, timeout=120):
    """GET con reintentos y espera creciente ante fallas de red o del servidor."""
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


def _bajar_dxy():
    import yfinance as yf
    d = yf.download("DX-Y.NYB", start="2003-01-01", progress=False, auto_adjust=False, threads=False)
    if d.empty:
        raise RuntimeError("yfinance devolvió una tabla vacía para DX-Y.NYB")
    d.columns = [c[0] if isinstance(c, tuple) else c for c in d.columns]   # quita el nivel del ticker
    d = d.reset_index().rename(columns={"Date": "fecha"})
    return d.to_csv(index=False).encode("utf-8")


def descargar(nombre):
    """Descarga una serie y devuelve sus bytes crudos."""
    if nombre == "dxy":
        return _bajar_dxy()
    return _get(FUENTES[nombre]["url"])


def guardar(nombre, contenido):
    """Escribe el archivo crudo en data/raw/ y devuelve sus metadatos (MD5, tamaño, momento de descarga)."""
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


def md5(ruta):
    return hashlib.md5(ruta.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Lectura de los archivos crudos (sin limpieza: solo formato)
# ---------------------------------------------------------------------------
_NS = {"g": "http://www.sdmx.org/resources/sdmxml/schemas/v2_1/data/generic"}


def leer_sdmx(ruta):
    """XML SDMX del Banrep → DataFrame (fecha, valor). Exige una sola serie."""
    series = ET.parse(ruta).getroot().findall(".//g:Series", _NS)
    if len(series) != 1:
        raise ValueError(f"{ruta.name}: se esperaba 1 serie y llegaron {len(series)}")
    filas = [(o.find("g:ObsDimension", _NS).get("value"), o.find("g:ObsValue", _NS).get("value"))
             for o in series[0].findall("g:Obs", _NS)]
    df = pd.DataFrame(filas, columns=["fecha", "valor"])
    df["fecha"] = pd.to_datetime(df["fecha"], format="%Y%m%d")
    df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
    return df.sort_values("fecha").reset_index(drop=True)


def leer_fred(ruta):
    """CSV de FRED → DataFrame (fecha, valor). Los '.' (días sin dato) quedan como NaN."""
    df = pd.read_csv(ruta, na_values=".")
    df.columns = ["fecha", "valor"]
    df["fecha"] = pd.to_datetime(df["fecha"])
    return df


def leer_vigencias(ruta):
    """CSV de datos.gov.co → DataFrame (fecha = vigencia desde, vigencia_hasta, valor)."""
    df = pd.read_csv(ruta, parse_dates=["vigenciadesde", "vigenciahasta"])
    return (df.rename(columns={"vigenciadesde": "fecha", "vigenciahasta": "vigencia_hasta"})
              [["fecha", "vigencia_hasta", "valor"]].sort_values("fecha").reset_index(drop=True))


def leer_dxy(ruta):
    """CSV guardado desde yfinance → DataFrame (fecha, valor = cierre)."""
    df = pd.read_csv(ruta, parse_dates=["fecha"])
    return df.rename(columns={"Close": "valor"})[["fecha", "valor"]]


LECTORES = {"trm": leer_sdmx, "tpm": leer_sdmx, "trm_vigencias": leer_vigencias,
            "brent": leer_fred, "dff": leer_fred, "dxy": leer_dxy}


def leer(nombre):
    """Lee la copia congelada de una serie."""
    return LECTORES[nombre](DATA_RAW / FUENTES[nombre]["archivo"])
