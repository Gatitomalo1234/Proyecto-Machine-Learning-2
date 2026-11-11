# Predicción de la TRM con gradient boosting: el aporte de variables macroeconómicas

Proyecto Integrador de **Machine Learning II**, Universidad Externado de Colombia — **Segunda entrega** (Fase 2: datos, modelamiento y validación).
Integrantes: Nicolás Cárdenas Díaz y María Paula Muñoz Malaver.
Artículo base: Eslami Ghayour y Yousefi (2023), *Performance Analysis of Gradient Boosting Models for Forex Market Prediction* (SSRN).

**Pregunta de investigación (Entrega 1).** ¿La incorporación de variables macroeconómicas locales (petróleo, diferencial de tasas Banrep–Fed, fortaleza del dólar) mejora la predicción del precio diario de la TRM frente a un modelo que solo usa su comportamiento pasado, y cuál de los tres algoritmos de gradient boosting aprovecha mejor esa información?

**Respuesta.** No la mejora. Con pruebas formales (Diebold-Mariano y bootstrap por bloques en validación), ninguna versión con macro predice mejor que la misma sin ellas, y ningún boosting se distingue de una regresión lineal sobre la TRM. Ningún algoritmo aprovecha las macro: LightGBM es el que menos se perjudica y XGBoost el que más. La única señal es la continuación del movimiento de hoy (la TRM es el promedio de un día), y es casi lineal. El diferencial de tasas hace daño por un cambio de régimen: en el desarrollo, los valores altos solo coincidieron con la devaluación de 2008–2009. **Modelo final: CatBoost con solo la TRM**, en empate estadístico con la regresión lineal (ficha en `models/ficha_modelo_final.md`). Con los datos del propio artículo, su método tampoco supera a copiar el precio de hoy (notebook 08b).

---

## Diseño (contrato con la Entrega 1)

| Elemento | Decisión |
|---|---|
| Objetivo | Retorno logarítmico de la TRM del día siguiente, ln(TRM_{t+1}/TRM_t); el nivel se reconstruye como TRM_t·e^ŷ para medir el error en pesos |
| Variables | Las de la Entrega 1, en retornos: TRM rezagada 1, 5 y 10 días; Brent; DXY; diferencial TPM − DFF (en nivel) |
| Línea base (única) | Regresión lineal sobre los rezagos 1, 5 y 10 de la TRM (`StandardScaler` + `LinearRegression`) |
| Modelos avanzados | LightGBM, CatBoost y XGBoost, cada uno con **solo TRM** y con **TRM + macro** (6 configuraciones) |
| Métricas | RMSE en pesos (principal), MAE en pesos y R² sobre retornos |
| Validación | Partición cronológica 80/20; **validación anidada** en el desarrollo: walk-forward externo de 5 cortes (gap 5) y `TimeSeriesSplit(3)` + `RandomizedSearchCV` (15 combinaciones) interno; test usado una sola vez |
| Periodo | 2003-12-01 a 2026-09-29 (fecha de descarga, congelada) |
| Semilla | 42 en todo |

Los cambios frente a la Entrega 1 y su motivo están en `reports/tablas/tab05b_cambios_entrega1.csv`.

---

## Estructura

```
Entrega 2/
├── README.md, requirements.txt (versiones exactas), .vscode/
├── config_congelada.json      Configuración elegida solo con desarrollo, con MD5 de su contenido
├── src/                       Código reutilizable
│   ├── config.py              Semilla, periodo, rutas, variables, modelos y parámetros de validación
│   ├── descarga.py            Descarga desde las fuentes oficiales y lectura de los archivos crudos
│   ├── calidad.py             Tabla de calidad por serie
│   ├── alineacion.py          Alineación sin fuga (merge_asof hacia atrás) y pruebas anti-fuga
│   ├── variables.py           Objetivo, 6 variables y prueba de perturbación
│   ├── eda.py                 Funciones del análisis exploratorio
│   ├── validacion.py          Partición, cortes walk-forward y métricas
│   ├── modelos.py             Espacios de búsqueda, pipelines, validación anidada, modelos finales y Optuna
│   └── graficos.py            Estilo y guardado de figuras y tablas
├── scripts/descargar_datos.py Descarga y congela los datos crudos (se ejecuta una vez)
├── scripts/verificar_documentacion.py  Comprueba que todas las funciones estén documentadas (ver "Documentación del código")
├── notebooks/                 Ver "Orden de ejecución"
├── data/raw/                  Datos crudos congelados + metadatos.json (fuente, URL, fecha, MD5)
├── data/raw/articulo/         Datos del artículo base (5 pares de divisas del repositorio del autor) + metadatos.json
├── data/raw/monitoreo/        TRM descargadas el 10/11/2026 para evaluar el modelo en días nuevos (notebook 10) + metadatos.json
├── data/processed/            series_limpias, dataset, particion, predicciones de validación y de test
├── app/                       Aplicación Streamlit (app.py, estilo.py, README.md): ver app/README.md; sus dependencias, en ../requirements.txt
├── tests/                     14 pruebas automáticas (pytest) del núcleo de la app y de la app
├── models/                    modelos_finales.joblib (los 7 modelos congelados; el notebook 05 los regenera idénticos);
│                              modelo_final_despliegue.joblib (modelo final + línea base con 2003–2026, para la app);
│                              ficha_modelo_final.md y modelo_final.json (configuración definitiva)
├── reports/figuras/           fig01–fig16 y figE1–figE5 (fig07–08: ventanas; fig09–10: modelo final y diferencial; fig11: artículo frente a la TRM; fig12–14: interpretación; fig15–16: rango adaptativo y monitoreo)
├── reports/tablas/            74 tablas numeradas (tab00–tab16 y tabE1–tabE10; tab10*: ventanas; tab11–12*, tab14*: modelo final; tab13*: datos del artículo; tab15*: interpretación; tab16*: limitaciones y proyección)
├── reports/informe/           Informe de la entrega (LaTeX + PDF) y los scripts que arman sus tablas y figuras
└── reports/guia_sustentacion.md  Guion y preguntas probables para la sustentación oral
```

---

## Instalación

Python 3.12. En la carpeta `Entrega 2/`:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` fija las versiones exactas con las que se obtuvieron los resultados. XGBoost debe ser la 3.2.0: la versión 3.4 entrega resultados distintos para este modelo.

Para ejecutar los notebooks desde la terminal con este entorno:

```bash
python -m ipykernel install --user --name trm      # una sola vez
python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=trm notebooks/01_datos_y_calidad.ipynb
```

## Orden de ejecución

| Paso | Archivo | Qué hace | Rúbrica |
|---|---|---|---|
| 0 | `python scripts/descargar_datos.py` | Descarga las 5 series por API oficial y las congela. Si la copia existe, no descarga y verifica su MD5 (`--forzar` para rehacerla) | 7.1 |
| 1 | `01_datos_y_calidad.ipynb` | Verifica la copia congelada, cruza la TRM del Banrep con la de la Superfinanciera, tabla de calidad y Figura 1 | 7.1 |
| 2 | `02_alineacion_y_variables.ipynb` | Alineación sin fuga, objetivo y 6 variables, prueba de perturbación y partición 80/20 | 7.1, 7.2 |
| 3 | `03_eda.ipynb` | EDA solo con desarrollo: 10 preguntas (estacionariedad, colas, volatilidad, memoria, relaciones, estabilidad, tasas, redundancia, no linealidad, regímenes) e hipótesis H1–H6 escritas antes de entrenar | 7.1, 7.4 |
| 4 | `04_reproduccion_articulo.ipynb` | El método del artículo tal cual (niveles, MinMax, 80/20) dentro del desarrollo: techo de extrapolación (Figura 3) | 7.3 |
| 5 | `05_modelos_validacion.ipynb` | Figura 2 (pipeline), línea base y 6 configuraciones con validación anidada, aporte macro, hipótesis, auditoría, congelamiento y prueba de robustez con Optuna | 7.2, 7.3, 7.4 |
| 6 | `06_test_e_interpretacion.ipynb` | Verifica lo congelado, abre el test una sola vez (7 configuraciones + método del artículo), estabilidad, SHAP, errores por periodo, hipótesis e implicaciones | 7.4 |
| 7 | `07_ventanas_rezago.ipynb` | Experimento pedido tras la Entrega 2: ¿más días de rezago mejoran? Compara rezagos 1, 5 y 10 con los últimos 20 días y con ventanas de tendencia y volatilidad, con la misma validación anidada y solo con el desarrollo (no abre el test). Resultado: no mejora; la memoria de la TRM es de un día y se debilita con la volatilidad | 8.1, 8.2, 8.4 |
| 8b | `08b_articulo_datos_originales.ipynb` | ¿Por qué al artículo "le funciona"? Descarga y congela sus datos (5 pares), revisa su calidad (7 picos erróneos), aplica su método tal cual (recupera sus R² y, con la TRM, la Tabla 4) frente a copiar el precio de hoy, y la corrección a retornos. Resultado: su R² no supera a copiar el precio; sus monedas no salen del rango de entrenamiento; en retornos, la TRM es la más predecible | 8.1 |
| 8 | `08_modelo_final.ipynb` | Respuesta a la pregunta con pruebas formales (Diebold-Mariano y bootstrap en validación), modelo final, por qué no funcionó con la TRM (señal y diferencial: régimen de 2008–2009, dependencia parcial y modelo sin diferencial), contraste con el artículo, evaluación en el test (solo reporte), ficha del modelo y modelo reentrenado para la app | 8.1 |
| 9 | `09_interpretacion.ipynb` | Interpretación del modelo final (SHAP y dependencia parcial): el movimiento de hoy pesa el 82 % y su efecto tiene forma de S con techo (+35 / −24 pb); cuatro días explicados con una regla fija; tipos de error (dirección casi al azar, el 10 % de días grandes produce la mitad del error, el rango del 80 % se queda corto con volatilidad alta); qué significa para un importador (error típico COP 2,7 millones en USD 100.000) y alcance de las conclusiones | 8.2 |
| 10 | `10_limitaciones_y_proyeccion.ipynb` | Limitaciones con su evidencia (datos, simulación, sobreajuste, estabilidad, generalización, método); M1: rango del 80 % que se ensancha con la volatilidad de 40 TRM (ventana elegida con la validación; cobertura en el test del 71 % al 80 %), llevado a la app; M2: monitoreo con las TRM nuevas posteriores al 26/09/2026 (congeladas en `data/raw/monitoreo/`); mejoras propuestas y proyección a Deep Learning y MLOps | 8.4 |

Cada notebook lee de disco lo que dejó el anterior, así que puede ejecutarse por separado. **El 08b va antes del 08**, porque la parte 3 del 08 usa sus tablas. El proyecto completo corre en unos 5 minutos.

## Aplicación (despliegue)

`app/` contiene la aplicación en Streamlit que predice la próxima TRM con el modelo final, en vivo desde datos.gov.co, con explicación (SHAP) y un semáforo de coherencia entre los datos de uso y los de entrenamiento. Toda su lógica está en `src/prediccion.py`. Instrucciones, pruebas y publicación en Streamlit Community Cloud: [`app/README.md`](app/README.md).

```bash
streamlit run app/app.py
python -m pytest
```

## Documentación del código

Todas las funciones (las 50 del proyecto: en `src/`, en los scripts y las definidas dentro de los notebooks) siguen la misma convención: docstring en formato NumPy, en español, con un resumen de una línea, el porqué cuando hay una decisión de método (fuga, reproducibilidad) y las secciones **Parámetros**, **Devuelve**, **Lanza** y **Notas** cuando aplican. En los archivos `.py` cada parámetro y cada retorno llevan tipo. Para consultar una función: `help(src.alineacion.alinear)`. `src/__init__.py` resume qué hace cada módulo y en qué notebook se usa.

La convención se comprueba automáticamente:

```bash
python scripts/verificar_documentacion.py
```

El script revisa que cada módulo y cada función tengan docstring, que se documente cada parámetro, lo que se devuelve y lo que se lanza, y que haya tipos. Termina con código 1 y la lista de faltantes si algo no cumple.

## Informe

El informe está en `reports/informe/informe_entrega2.pdf`. Sus tablas y figuras no se copian a mano: se generan a partir de las salidas de los notebooks, así que las cifras coinciden por construcción. Para regenerarlo después de correr los notebooks:

```bash
cd reports/informe
python tablas_latex.py        # Tablas 1, 4, 5 y 6 desde reports/tablas/*.csv y config_congelada.json
python recortar_titulos.py    # Figuras 2 y 3 (copias sin el título interno)
python figura_flujo.py        # Figura 1 (flujo del proyecto)
pdflatex informe_entrega2.tex && pdflatex informe_entrega2.tex
```

## Reproducibilidad y ausencia de fuga

- **Semilla 42** en la partición, los modelos, las búsquedas y SHAP; los modelos usan un solo hilo para que el resultado sea idéntico.
- **Datos congelados:** los archivos crudos se guardan tal como llegan, con huella MD5. Las fuentes corrigen datos históricos; por eso se trabaja sobre la copia.
- **Verificado:** borrando todo lo generado y ejecutando el proyecto desde cero, las tablas, los datos procesados y `config_congelada.json` salen idénticos byte a byte; los notebooks 07, 08 y 08b dan salidas idénticas en dos corridas, incluidas las huellas del modelo para la app (`models/modelo_final.json`).
- **Sin fuga:** cada variable usa solo información disponible antes del día que se predice (Tabla 2). Una prueba de perturbación en 4 fechas muestra que alterar el futuro no cambia ninguna variable. Las transformaciones van en `Pipeline` y se ajustan con el entrenamiento de cada corte. El EDA y la selección usan solo el desarrollo; el test se abre una vez, en el notebook 06, con los modelos congelados, cuya identidad se verifica antes (MD5 de la configuración y huella de las predicciones).
- **Robustez del ajuste:** cambiar la búsqueda aleatoria por optimización bayesiana (Optuna) o subir de 15 a 50 intentos mueve el RMSE menos de 0,22 pesos.

## Fuentes de datos

| Variable | Fuente |
|---|---|
| TRM (COP/USD) | Banco de la República, servicio web SDMX (flujo `DF_TRM_DAILY_HIST`); vigencias de la Superintendencia Financiera en datos.gov.co (conjunto `32sa-8pi3`) |
| Tasa de política monetaria | Banco de la República, servicio web SDMX (flujo `DF_CBR_DAILY_HIST`) |
| Petróleo Brent | FRED, serie `DCOILBRENTEU` |
| Tasa de fondos federales | FRED, serie `DFF` |
| Índice del dólar (DXY) | Yahoo Finance, `DX-Y.NYB` |
| Datos del artículo base (notebook 08b) | Repositorio público del autor, [`arasheslamii/1-Performance-comparison-of-Gradient-Boosting-algorithms-for-Forex-market-Prediction-A-Study-on-Lig`](https://github.com/arasheslamii/1-Performance-comparison-of-Gradient-Boosting-algorithms-for-Forex-market-Prediction-A-Study-on-Lig), fijado al commit `506912c` (GU, EU, AU, NU y UC.csv; EU-C.csv es idéntico a EU.csv y no se usa). Eslami Ghayour y Yousefi (2023) |

## Uso de IA generativa

El proyecto se desarrolló con apoyo de un asistente de IA (Claude Code, de Anthropic) para la estructuración del código, la documentación y la revisión metodológica. Las decisiones, la ejecución y la interpretación de los resultados son responsabilidad de los autores.
