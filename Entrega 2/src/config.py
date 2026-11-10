"""Configuración única del proyecto: semilla, periodo, rutas y variables.

Todos los notebooks y scripts importan de aquí, así una decisión se cambia en un solo lugar.
Contiene: la semilla, el periodo de estudio, las rutas de datos y salidas, las variables y los
modelos acordados en la Entrega 1, y los parámetros de la partición y la validación.
Al importarse crea las carpetas de salida que falten (data/raw, data/processed, models,
reports/figuras y reports/tablas).
"""
from __future__ import annotations

from pathlib import Path

# --- Reproducibilidad ---
SEMILLA = 42

# --- Periodo de estudio (Entrega 1: "desde diciembre de 2003 hasta hoy") ---
FECHA_INICIO = "2003-12-01"
# Fijada a mano el día de la descarga de los datos (ver data/raw/metadatos.json). NO se vuelve a cambiar.
FECHA_FIN = "2026-09-29"

# --- Rutas (funcionan desde la raíz, desde notebooks/ o desde scripts/) ---
RAIZ = Path(__file__).resolve().parents[1]
DATA_RAW = RAIZ / "data" / "raw"
DATA_PROC = RAIZ / "data" / "processed"
MODELOS = RAIZ / "models"
FIGURAS = RAIZ / "reports" / "figuras"
TABLAS = RAIZ / "reports" / "tablas"
CONFIG_CONGELADA = RAIZ / "config_congelada.json"

# --- Variables (contrato con la Entrega 1) ---
OBJETIVO = "y"                          # ln(TRM_{t+1} / TRM_t)
VARS_TRM = ["r_trm_l1", "r_trm_l5", "r_trm_l10"]   # línea base = regresión lineal sobre estas
VARS_MACRO = ["r_brent", "r_dxy", "dif_tasas"]      # dif_tasas = TPM − DFF, en nivel
VARS_TODAS = VARS_TRM + VARS_MACRO

# --- Modelos (contrato con la Entrega 1) ---
LINEA_BASE = "lineal"                                  # regresión lineal sobre VARS_TRM (única línea base)
MODELOS_BOOSTING = ["LightGBM", "CatBoost", "XGBoost"]  # modelos avanzados de la Entrega 1
CONJUNTOS = {"solo_trm": VARS_TRM, "trm_macro": VARS_TODAS}  # 3 × 2 = 6 configuraciones

# --- Experimento de ventanas de rezago (notebook 07; pedido del profesor tras la Entrega 2) ---
VARS_DIARIOS = [f"r_trm_l{k}" for k in range(1, 21)]          # movimiento de cada uno de los últimos 20 días
VARS_VENTANA = ["r_trm_acum5", "r_trm_acum10", "r_trm_acum20", "r_trm_acum60", "dist_media20",
                "vol_trm5", "vol_trm20", "vol_trm60"]          # tendencia y volatilidad recientes
CONJUNTOS_VENTANAS = {"solo_trm": VARS_TRM, "diarios_1_20": VARS_DIARIOS,
                      "trm_ventanas": VARS_TRM + VARS_VENTANA}  # solo_trm = referencia

# --- Partición y validación ---
FRACCION_TEST = 0.20          # corte cronológico 80/20, como el artículo y la Entrega 1
N_CORTES_EXTERNOS = 5         # walk-forward externo (desempeño con variabilidad)
N_CORTES_INTERNOS = 3         # TimeSeriesSplit interno (hiperparámetros)
GAP = 5                       # filas de separación entre entrenamiento y validación
N_BUSQUEDA = 15               # combinaciones de RandomizedSearchCV


def exigir_fecha_fin() -> str:
    """Devuelve la fecha final del periodo y detiene la ejecución si todavía no se ha fijado.

    La fecha final se fija a mano el día de la descarga y no se vuelve a cambiar: las fuentes
    corrigen datos pasados, así que el periodo debe quedar congelado junto con los archivos crudos.

    Devuelve
    --------
    str
        `FECHA_FIN` en formato ``"AAAA-MM-DD"``.

    Lanza
    -----
    RuntimeError
        Si `FECHA_FIN` es None (antes de la primera descarga).
    """
    if FECHA_FIN is None:
        raise RuntimeError("FECHA_FIN no está fijada en src/config.py; se fija el día de la descarga.")
    return FECHA_FIN


for _c in (DATA_RAW, DATA_PROC, MODELOS, FIGURAS, TABLAS):
    _c.mkdir(parents=True, exist_ok=True)
