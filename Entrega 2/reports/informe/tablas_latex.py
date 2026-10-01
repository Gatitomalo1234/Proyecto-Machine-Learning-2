"""Genera las tablas numéricas del informe directamente desde reports/tablas/*.csv.

Así las cifras del informe coinciden por construcción con las salidas de los notebooks.
Uso: python reports/informe/tablas_latex.py
"""
from pathlib import Path

import pandas as pd

AQUI = Path(__file__).resolve().parent
TABLAS = AQUI.parent / "tablas"
SALIDA = AQUI / "tablas"
SALIDA.mkdir(exist_ok=True)


def num(x, d=2):
    """Número con coma decimal y punto de miles (formato del informe)."""
    s = f"{x:,.{d}f}"
    s = s.replace(",", "X").replace(".", ",").replace("X", ".")
    return "$-$" + s[1:] if s.startswith("-") else s


def fecha(s):
    a, m, _ = str(s).split("-")
    meses = "ene feb mar abr may jun jul ago sep oct nov dic".split()
    return f"{meses[int(m) - 1]} {a}"


# --------------------------------------------------------------------------- Tabla 1
cal = pd.read_csv(TABLAS / "tab01_calidad.csv").set_index("serie")
FUENTES = {
    "TRM": "Banco de la República (SDMX)",
    "Brent": r"FRED, \texttt{DCOILBRENTEU}",
    "DFF": r"FRED, \texttt{DFF}",
    "DXY": r"Yahoo Finance, \texttt{DX-Y.NYB}",
    "TPM": "Banco de la República (SDMX)",
}
filas = []
for serie in ["TRM", "Brent", "DXY", "TPM", "DFF"]:
    r = cal.loc[serie]
    filas.append(f"{serie} & {FUENTES[serie]} & {fecha(r.desde)} -- {fecha(r.hasta)} & {num(r.observaciones, 0)} & "
                 f"{num(r.faltantes, 0)} & {num(r.fechas_duplicadas, 0)} & {num(r.mayor_hueco_dias, 0)} & "
                 f"{num(r.cambios_extremos, 0)} \\\\")
(SALIDA / "tabla1_calidad.tex").write_text("\n".join([
    r"\begin{tabular}{llcrrrrr}", r"\toprule",
    r"Serie & Fuente & Periodo & Obs. & Falt. & Dupl. & Hueco máx. & Extremos \\", r"\midrule",
    *filas, r"\bottomrule", r"\end{tabular}"]) + "\n")

# --------------------------------------------------------------------------- Tabla 4
val = pd.read_csv(TABLAS / "tab05_validacion.csv").set_index("modelo")
test = pd.read_csv(TABLAS / "tab07_test.csv").set_index("modelo")
aporte = pd.read_csv(TABLAS / "tab06_aporte_macro.csv").set_index("algoritmo")
orden = ["Línea base (lineal)"] + [f"{m} · {c}" for m in ["LightGBM", "CatBoost", "XGBoost"] for c in ["solo_trm", "trm_macro"]]
mejor_val, mejor_test = val["rmse_media"].idxmin(), test["rmse_pesos"].idxmin()
NOMBRE = {"solo_trm": "solo TRM", "trm_macro": "TRM + macro"}
filas = []
for i, m in enumerate(orden):
    v, t = val.loc[m], test.loc[m]
    if m.startswith("Línea"):
        alg, conj, gana, macro = "Línea base (lineal)", "solo TRM", "--", "--"
    else:
        alg, c = m.split(" · ")
        conj = NOMBRE[c]
        gana = f"{int(v.cortes_que_gana_a_linea_base_rmse)} de 5"
        macro = f"{int(aporte.loc[alg, 'cortes_donde_macro_baja_rmse'])} de 5" if c == "trm_macro" else ""
        alg = alg if c == "solo_trm" else ""
    rv = num(v.rmse_media)
    rv = rf"\textbf{{{rv}}}" if m == mejor_val else rv
    rt = num(t.rmse_pesos)
    rt = rf"\textbf{{{rt}}}" if m == mejor_test else rt
    delta = "--" if m.startswith("Línea") else ("+" if t.delta_rmse_vs_base >= 0 else "") + num(t.delta_rmse_vs_base)
    if i in (1, 3, 5):
        filas.append(r"\addlinespace[2pt]")
    filas.append(f"{alg} & {conj} & {rv} $\\pm$ {num(v.rmse_desv)} & {num(v.r2_media, 3)} & {gana} & {macro} & "
                 f"{rt} & {num(t.mae_pesos)} & {num(t.r2_retornos, 3)} & {delta} \\\\")
(SALIDA / "tabla4_resultados.tex").write_text("\n".join([
    r"\begin{tabular}{llcccc|cccc}", r"\toprule",
    r" & & \multicolumn{4}{c|}{Validación (desarrollo, 5 cortes)} & \multicolumn{4}{c}{Test (una sola vez)} \\",
    r"\cmidrule(lr){3-6}\cmidrule(lr){7-10}",
    r"Algoritmo & Variables & RMSE (media $\pm$ desv.) & $R^2$ & Gana a la base & Macro mejora & RMSE & MAE & $R^2$ & $\Delta$RMSE \\",
    r"\midrule", *filas, r"\bottomrule", r"\end{tabular}"]) + "\n")
print("Tablas generadas en", SALIDA)

# --------------------------------------------------------------------------- Tabla 4: hiperparámetros
import json
import ast

RAIZ = AQUI.parent.parent
conf = json.loads((RAIZ / "config_congelada.json").read_text())["modelos"]
esp = pd.read_csv(TABLAS / "tab04b_espacios_busqueda.csv", index_col=0)
ALG = ["LightGBM", "CatBoost", "XGBoost"]
# fila del informe → nombre del hiperparámetro en cada algoritmo
FILAS = [
    ("Número de árboles", "Cuántas correcciones se suman", {"LightGBM": "n_estimators", "CatBoost": "iterations", "XGBoost": "n_estimators"}),
    ("Tasa de aprendizaje", "Cuánto corrige cada árbol", {a: "learning_rate" for a in ALG}),
    ("Profundidad", "Cuántas variables se combinan en una regla", {"LightGBM": "max_depth", "CatBoost": "depth", "XGBoost": "max_depth"}),
    ("Mínimo por hoja", "Días mínimos para crear una regla", {"LightGBM": "min_child_samples", "CatBoost": "min_data_in_leaf", "XGBoost": "min_child_weight"}),
    ("Hojas máximas", "Tamaño del árbol (solo LightGBM)", {"LightGBM": "num_leaves"}),
    ("Filas por árbol", "Fracción de días que ve cada árbol", {a: "subsample" for a in ALG}),
    ("Variables por árbol", "Fracción de variables que ve cada árbol", {"LightGBM": "colsample_bytree", "CatBoost": "rsm", "XGBoost": "colsample_bytree"}),
    ("Regularización L2", "Cuánto se encogen las predicciones", {"LightGBM": "reg_lambda", "CatBoost": "l2_leaf_reg", "XGBoost": "reg_lambda"}),
]


def corto(x):
    """0.03 → 0,03; 200.0 → 200."""
    x = float(x)
    return str(int(x)) if x.is_integer() else str(x).replace(".", ",")


def posibles(mapa):
    vals = {a: ast.literal_eval(esp.loc[p, a]) for a, p in mapa.items()}
    distintos = {tuple(v) for v in vals.values()}
    fmt = lambda v: "; ".join(corto(x) for x in v)
    if len(distintos) == 1:
        return fmt(next(iter(vals.values())))
    base = vals.get("LightGBM", next(iter(vals.values())))
    otros = [f"{a}: {'peso ' if mapa[a] == 'min_child_weight' else ''}{fmt(v)}" for a, v in vals.items() if v != base]
    return fmt(base) + " (" + "; ".join(otros) + ")"


filas = []
for nombre, que, mapa in FILAS:
    celdas = []
    for a in ALG:
        if a not in mapa:
            celdas += ["--", "--"]
            continue
        for c in ["solo_trm", "trm_macro"]:
            celdas.append(corto(conf[f"{a} · {c}"]["hiperparametros"][mapa[a]]))
    filas.append(f"{nombre} & {que} & {posibles(mapa)} & " + " & ".join(celdas) + r" \\")
(SALIDA / "tabla_hiperparametros.tex").write_text("\n".join([
    r"\begin{tabular}{l >{\raggedright\arraybackslash}p{5.6cm} >{\raggedright\arraybackslash}p{5.4cm} cc cc cc}", r"\toprule",
    r" & & & \multicolumn{2}{c}{LightGBM} & \multicolumn{2}{c}{CatBoost} & \multicolumn{2}{c}{XGBoost} \\",
    r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
    r"Hiperparámetro & Qué controla & Valores posibles & solo & macro & solo & macro & solo & macro \\",
    r"\midrule", *filas, r"\bottomrule", r"\end{tabular}"]) + "\n")

# --------------------------------------------------------------------------- Tabla 5: robustez del ajuste
rob = pd.read_csv(TABLAS / "tab05g_robustez_optuna.csv", header=[0, 1], index_col=0).dropna(how="all")
rmse = rob["rmse_media"]
rmse.columns = ["azar15", "optuna15", "optuna50"]
orden_cfg = [f"{a} · {c}" for a in ALG for c in ["solo_trm", "trm_macro"]]
base_val = num(val.loc["Línea base (lineal)", "rmse_media"])
filas = []
for col, etiqueta in [("azar15", "Al azar, 15 intentos (usado)"), ("optuna15", "Optuna, 15 intentos"), ("optuna50", "Optuna, 50 intentos")]:
    celdas = [num(rmse.loc[k, col]) for k in orden_cfg]
    filas.append(f"{etiqueta} & " + " & ".join(celdas) + f" & {base_val if col == 'azar15' else ''} \\\\")
cambio = [num(max(abs(rmse.loc[k, "optuna15"] - rmse.loc[k, "azar15"]), abs(rmse.loc[k, "optuna50"] - rmse.loc[k, "azar15"]))) for k in orden_cfg]
filas += [r"\midrule", "Cambio máximo & " + " & ".join(cambio) + r" & -- \\"]
(SALIDA / "tabla_robustez.tex").write_text("\n".join([
    r"\begin{tabular}{l cc cc cc c}", r"\toprule",
    r" & \multicolumn{2}{c}{LightGBM} & \multicolumn{2}{c}{CatBoost} & \multicolumn{2}{c}{XGBoost} & Línea \\",
    r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
    r"Método de ajuste & solo & macro & solo & macro & solo & macro & base \\",
    r"\midrule", *filas, r"\bottomrule", r"\end{tabular}"]) + "\n")
print("Tablas de hiperparámetros y robustez generadas")
