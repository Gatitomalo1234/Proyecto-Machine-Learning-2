"""Genera las tablas del informe final directamente desde las salidas del proyecto.

Así las cifras del informe coinciden por construcción con las de los notebooks. Escribe en
reports/informe_final/tablas/: tabla_resultados.tex (validación, test y pruebas formales de los
7 modelos) y tabla_importador.tex (error en pesos para una compra de USD 100.000).
Uso: python reports/informe_final/tablas_latex.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[1]
sys.path.insert(0, str(RAIZ))
from src.validacion import errores_en_pesos, prueba_diebold_mariano   # noqa: E402

TABLAS, PROC, SALIDA = RAIZ / "reports" / "tablas", RAIZ / "data" / "processed", AQUI / "tablas"
BASE = "Línea base (lineal)"
ORDEN = [BASE] + [f"{a} · {c}" for a in ["LightGBM", "CatBoost", "XGBoost"] for c in ["solo_trm", "trm_macro"]]


def num(x: float, d: int = 2, signo: bool = False) -> str:
    """Escribe un número con coma decimal y punto de miles (formato del informe).

    Parámetros
    ----------
    x : float
        Número que se escribe.
    d : int, por defecto 2
        Decimales.
    signo : bool, por defecto False
        Antepone "+" a los positivos.

    Devuelve
    --------
    str
        El número formateado; los negativos llevan ``$-$`` para que LaTeX escriba un signo menos.
    """
    s = f"{x:{'+' if signo else ''},.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return "$-$" + s[1:] if s.startswith("-") else s


def tabla_resultados() -> str:
    """Construye la tabla de resultados: validación, test y Diebold-Mariano frente a la línea base.

    Devuelve
    --------
    str
        Entorno ``tabular`` de LaTeX.
    """
    val = pd.read_csv(TABLAS / "tab05_validacion.csv", index_col=0)
    test = pd.read_csv(TABLAS / "tab07_test.csv", index_col=0)
    t11 = pd.read_csv(TABLAS / "tab11_boosting_vs_base.csv", index_col=0)
    pt = pd.read_csv(PROC / "predicciones_test.csv")
    e_base = errores_en_pesos(pt["y"], pt[BASE], pt["trm"])
    mejor_val, mejor_test = val["rmse_media"].idxmin(), test["rmse_pesos"].idxmin()
    filas = []
    for i, m in enumerate(ORDEN):
        alg, conj = ("Línea base (lineal)", "solo TRM") if m == BASE else (m.split(" · ")[0], {"solo_trm": "solo TRM", "trm_macro": "TRM + macro"}[m.split(" · ")[1]])
        if m != BASE and m.endswith("trm_macro"):
            alg = ""
        p_val = "--" if m == BASE else num(t11.loc[m, "p_valor"])
        p_test = "--" if m == BASE else num(prueba_diebold_mariano(errores_en_pesos(pt["y"], pt[m], pt["trm"]), e_base)["p_valor"])
        rv = num(val.loc[m, "rmse_media"]); rv = rf"\textbf{{{rv}}}" if m == mejor_val else rv
        rt = num(test.loc[m, "rmse_pesos"]); rt = rf"\textbf{{{rt}}}" if m == mejor_test else rt
        if i in (1, 3, 5):
            filas.append(r"\addlinespace[2pt]")
        filas.append(f"{alg} & {conj} & {rv} & {num(val.loc[m, 'r2_media'], 3)} & {p_val} & {rt} & {num(test.loc[m, 'mae_pesos'])} & "
                     f"{num(test.loc[m, 'r2_retornos'], 3)} & {p_test} \\\\")
    return "\n".join([r"\begin{tabular}{llccc|cccc}", r"\toprule",
                      r" & & \multicolumn{3}{c|}{Validación (2006--2022, 5 cortes)} & \multicolumn{4}{c}{Test (2022--2026, una sola vez)} \\",
                      r"\cmidrule(lr){3-5}\cmidrule(lr){6-9}",
                      r"Algoritmo & Variables & RMSE & $R^2$ & $p$ DM & RMSE & MAE & $R^2$ & $p$ DM \\", r"\midrule",
                      *filas, r"\bottomrule", r"\end{tabular}"]) + "\n"


def tabla_importador() -> str:
    """Construye la tabla del error en pesos para un importador que compra USD 100.000.

    Devuelve
    --------
    str
        Entorno ``tabular`` de LaTeX.
    """
    t = pd.read_csv(TABLAS / "tab15f_error_para_un_importador.csv", index_col=0)
    filas = []
    for nombre, fila in t.iterrows():
        m = lambda c: num(fila[c] / 1e6, 1)
        filas.append(f"{nombre} & {num(fila.iloc[0])} & {m(t.columns[3])} & {num(fila.iloc[1])} & {m(t.columns[4])} & "
                     f"{num(fila.iloc[2])} & {m(t.columns[5])} \\\\")
    return "\n".join([r"\begin{tabular}{lcccccc}", r"\toprule",
                      r" & \multicolumn{2}{c}{Error típico (MAE)} & \multicolumn{2}{c}{9 de cada 10 días, menor a} & \multicolumn{2}{c}{Peor día} \\",
                      r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
                      r"Pronóstico & \$/USD & COP millones & \$/USD & COP millones & \$/USD & COP millones \\", r"\midrule",
                      *filas, r"\bottomrule", r"\end{tabular}"]) + "\n"


def main() -> None:
    """Escribe las tablas del informe en reports/informe_final/tablas/."""
    SALIDA.mkdir(exist_ok=True)
    (SALIDA / "tabla_resultados.tex").write_text(tabla_resultados(), encoding="utf-8")
    (SALIDA / "tabla_importador.tex").write_text(tabla_importador(), encoding="utf-8")
    print("Tablas generadas en", SALIDA)


if __name__ == "__main__":
    main()
