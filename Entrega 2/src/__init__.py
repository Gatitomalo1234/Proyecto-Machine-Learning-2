"""Código reutilizable del proyecto de predicción de la TRM (Entrega 2).

Los notebooks y los scripts importan de aquí; ningún cálculo importante vive solo en un notebook.
Módulos, en el orden del flujo:

- ``config``: semilla, periodo, rutas, variables, modelos y parámetros de validación.
- ``descarga``: descarga de las fuentes oficiales y lectura de los archivos crudos (notebook 01).
- ``calidad``: periodo de estudio, cambios extremos y tabla de calidad (notebook 01, Tabla 1).
- ``alineacion``: unión de las series sin fuga de información (notebook 02).
- ``variables``: objetivo, variables predictoras y prueba de perturbación (notebook 02).
- ``validacion``: partición cronológica, cortes walk-forward y métricas (notebooks 02, 05 y 06).
- ``eda``: pruebas y resúmenes del análisis exploratorio (notebook 03).
- ``modelos``: modelos, espacios de búsqueda y validación anidada (notebook 05).
- ``graficos``: estilo y guardado de figuras y tablas numeradas (todos los notebooks).

Convención de documentación: docstrings en formato NumPy, en español, y tipos en la firma; se
comprueba con ``python scripts/verificar_documentacion.py``.
"""
