"""Verifica que todas las funciones del proyecto estén documentadas con la misma convención.

Convención: docstrings en formato NumPy, en español, y tipos en la firma. Este script revisa los
módulos de src/, scripts/, reports/informe/ y app/, y las funciones definidas dentro de los notebooks.
Termina con código 1 y la lista de faltantes si encuentra alguno; con código 0 si todo cumple.

Reglas que comprueba:
- Cada archivo .py tiene docstring de módulo.
- Cada función tiene docstring, y su primera línea es un resumen que termina en punto.
- Cada parámetro aparece en la sección "Parámetros".
- Si la función devuelve un valor (``return <valor>``), tiene sección "Devuelve".
- Si la función tiene ``raise`` o ``assert``, tiene sección "Lanza".
- En los .py, cada parámetro y el retorno tienen tipo (en los notebooks no se exige).

Uso (desde la carpeta Entrega 2/):
    python scripts/verificar_documentacion.py
"""
from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

RAIZ = Path(__file__).resolve().parents[1]
CARPETAS_PY = ["src", "scripts", "reports/informe", "reports/informe_final", "app"]   # módulos que se revisan
SECCION = re.compile(r"^\s*(Parámetros|Devuelve|Lanza|Notas)\s*\n\s*-{3,}\s*$", re.MULTILINE)
FuncionAST = ast.FunctionDef | ast.AsyncFunctionDef


def secciones(doc: str) -> dict[str, str]:
    """Separa un docstring en sus secciones con formato NumPy.

    Parámetros
    ----------
    doc : str
        Docstring completo de una función.

    Devuelve
    --------
    dict[str, str]
        Nombre de la sección ("Parámetros", "Devuelve", "Lanza", "Notas") → texto que contiene.
        Solo incluye las secciones presentes.
    """
    marcas = list(SECCION.finditer(doc))
    return {m.group(1): doc[m.end(): marcas[i + 1].start() if i + 1 < len(marcas) else len(doc)]
            for i, m in enumerate(marcas)}


def nodos_propios(funcion: FuncionAST) -> Iterator[ast.AST]:
    """Recorre el cuerpo de una función sin entrar en las funciones, lambdas o clases anidadas.

    Así un ``return`` o un ``raise`` de una función interna no se le atribuye a la externa.

    Parámetros
    ----------
    funcion : ast.FunctionDef o ast.AsyncFunctionDef
        Función cuyo cuerpo se recorre.

    Devuelve
    --------
    Iterator[ast.AST]
        Cada nodo del cuerpo propio de la función.
    """
    pendientes = list(funcion.body)
    while pendientes:
        nodo = pendientes.pop()
        yield nodo
        if not isinstance(nodo, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            pendientes.extend(ast.iter_child_nodes(nodo))


def revisar_funcion(funcion: FuncionAST, exigir_tipos: bool) -> list[str]:
    """Aplica las reglas de la convención a una función.

    Parámetros
    ----------
    funcion : ast.FunctionDef o ast.AsyncFunctionDef
        Función que se revisa.
    exigir_tipos : bool
        Si es True, también exige tipo en cada parámetro y en el retorno (archivos .py).

    Devuelve
    --------
    list[str]
        Un mensaje por cada regla incumplida; lista vacía si la función cumple.
    """
    problemas = []
    a = funcion.args
    parametros = [p for p in a.posonlyargs + a.args + a.kwonlyargs + [x for x in (a.vararg, a.kwarg) if x]
                  if p.arg not in ("self", "cls")]
    if exigir_tipos:
        sin_tipo = [p.arg for p in parametros if p.annotation is None]
        if sin_tipo:
            problemas.append(f"parámetros sin tipo: {', '.join(sin_tipo)}")
        if funcion.returns is None:
            problemas.append("retorno sin tipo")

    doc = ast.get_docstring(funcion)
    if not doc:
        return problemas + ["sin docstring"]
    if not doc.splitlines()[0].strip().endswith("."):
        problemas.append("la primera línea no es un resumen terminado en punto")
    partes = secciones(doc)
    documentados = set(re.findall(r"^\s*([\w*]+(?:\s*,\s*[\w*]+)*)\s*:", partes.get("Parámetros", ""), re.MULTILINE))
    documentados = {n.strip().lstrip("*") for grupo in documentados for n in grupo.split(",")}
    faltan = [p.arg for p in parametros if p.arg not in documentados]
    if faltan:
        problemas.append(f"parámetros sin documentar en 'Parámetros': {', '.join(faltan)}")
    propios = list(nodos_propios(funcion))
    if any(isinstance(n, ast.Return) and n.value is not None for n in propios) and "Devuelve" not in partes:
        problemas.append("devuelve un valor y no tiene sección 'Devuelve'")
    if any(isinstance(n, (ast.Raise, ast.Assert)) for n in propios) and "Lanza" not in partes:
        problemas.append("tiene raise/assert y no tiene sección 'Lanza'")
    return problemas


def revisar_codigo(fuente: str, exigir_tipos: bool, exigir_modulo: bool) -> list[tuple[str, int, str]]:
    """Revisa todas las funciones de un bloque de código y, si aplica, el docstring del módulo.

    Parámetros
    ----------
    fuente : str
        Código Python (un archivo .py o una celda de notebook).
    exigir_tipos : bool
        Se pasa a `revisar_funcion`: True para archivos .py, False para notebooks.
    exigir_modulo : bool
        Si es True, exige docstring de módulo (archivos .py).

    Devuelve
    --------
    list[tuple[str, int, str]]
        (nombre de la función o "<módulo>", línea, problema), una tupla por problema.
    """
    arbol = ast.parse(fuente)
    hallados = []
    if exigir_modulo and not ast.get_docstring(arbol):
        hallados.append(("<módulo>", 1, "sin docstring de módulo"))
    for nodo in ast.walk(arbol):
        if isinstance(nodo, (ast.FunctionDef, ast.AsyncFunctionDef)):
            hallados += [(nodo.name, nodo.lineno, p) for p in revisar_funcion(nodo, exigir_tipos)]
    return hallados


def revisar_proyecto(raiz: Path = RAIZ) -> tuple[dict[str, list[tuple[str, int, str]]], int]:
    """Revisa los .py de `CARPETAS_PY` y las funciones definidas en los notebooks.

    Parámetros
    ----------
    raiz : Path, por defecto la carpeta Entrega 2/
        Carpeta raíz del proyecto.

    Devuelve
    --------
    tuple[dict[str, list[tuple[str, int, str]]], int]
        Problemas por archivo (en los notebooks, la ubicación es ``archivo#celdaNN``) y número de
        funciones revisadas.

    Notas
    -----
    En los notebooks no se exigen tipos: las funciones auxiliares de una celda se documentan con
    docstring, pero tiparlas no ayuda a leer el análisis.
    """
    problemas, n_funciones = {}, 0
    for carpeta in CARPETAS_PY:
        for ruta in sorted((raiz / carpeta).glob("*.py")):
            fuente = ruta.read_text(encoding="utf-8")
            n_funciones += sum(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) for n in ast.walk(ast.parse(fuente)))
            hallados = revisar_codigo(fuente, exigir_tipos=True, exigir_modulo=True)
            if hallados:
                problemas[str(ruta.relative_to(raiz))] = hallados
    for ruta in sorted((raiz / "notebooks").glob("*.ipynb")):
        celdas = [c for c in json.loads(ruta.read_text(encoding="utf-8"))["cells"] if c["cell_type"] == "code"]
        for i, celda in enumerate(celdas):
            fuente = "".join(celda["source"])
            try:
                arbol = ast.parse(fuente)
            except SyntaxError:          # celdas con comandos de IPython (%, !): no definen funciones
                continue
            n_funciones += sum(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) for n in ast.walk(arbol))
            hallados = revisar_codigo(fuente, exigir_tipos=False, exigir_modulo=False)
            if hallados:
                problemas[f"{ruta.relative_to(raiz)}#celda{i:02d}"] = hallados
    return problemas, n_funciones


def main() -> int:
    """Imprime el reporte de la revisión y devuelve el código de salida.

    Devuelve
    --------
    int
        0 si todas las funciones cumplen la convención; 1 si hay al menos un problema.
    """
    problemas, n_funciones = revisar_proyecto()
    funciones_con_problema = {(archivo, nombre) for archivo, lista in problemas.items()
                              for nombre, _, _ in lista if nombre != "<módulo>"}
    for archivo, lista in problemas.items():
        print(f"\n{archivo}")
        for nombre, linea, problema in lista:
            print(f"  línea {linea:>4}  {nombre}: {problema}")
    total = sum(len(lista) for lista in problemas.values())
    print(f"\nFunciones revisadas: {n_funciones} | con problemas: {len(funciones_con_problema)} | "
          f"problemas: {total}")
    print("Todo documentado según la convención." if total == 0 else "Faltan documentar los puntos de arriba.")
    return 0 if total == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
