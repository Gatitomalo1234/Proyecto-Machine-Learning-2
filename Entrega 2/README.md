# Predicción de la TRM con gradient boosting: el aporte de variables macroeconómicas

Proyecto Integrador de **Machine Learning II**, Universidad Externado de Colombia — **Segunda entrega** (Fase 2: datos, modelamiento y validación).
Integrantes: Nicolás Cárdenas Díaz y María Paula Muñoz Malaver.
Artículo base: Eslami Ghayour y Yousefi (2023), *Performance Analysis of Gradient Boosting Models for Forex Market Prediction* (SSRN).

**Pregunta de investigación (Entrega 1).** ¿La incorporación de variables macroeconómicas locales (petróleo, diferencial de tasas Banrep–Fed, fortaleza del dólar) mejora la predicción del precio diario de la TRM frente a un modelo que solo usa su comportamiento pasado, y cuál de los tres algoritmos de gradient boosting aprovecha mejor esa información?

**Respuesta preliminar.** No la mejora. En validación las variables macro no aportan de forma confiable en ningún algoritmo, y en el test empeoran a los tres. La señal predecible de la TRM diaria está en su movimiento del día anterior, y una regresión lineal la aprovecha tan bien como el mejor boosting. LightGBM es el que menos se perjudica con las macro. El diferencial de tasas induce un sobreajuste al régimen: el test visita valores que en el desarrollo solo aparecían en periodos de devaluación. No es fuga de información; es un cambio de distribución.

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
├── notebooks/                 Ver "Orden de ejecución"
├── data/raw/                  Datos crudos congelados + metadatos.json (fuente, URL, fecha, MD5)
├── data/processed/            series_limpias, dataset, particion, predicciones de validación y de test
├── models/                    modelos_finales.joblib (los 7 modelos congelados; el notebook 05 los regenera idénticos)
├── reports/figuras/           fig01–fig06 y figE1–figE5
├── reports/tablas/            38 tablas numeradas (tab00–tab09 y tabE1–tabE10)
├── reports/informe/           Informe de la entrega (LaTeX + PDF) y los scripts que arman sus tablas y figuras
└── reports/guia_sustentacion.md  Guion y preguntas probables para la sustentación oral
```

---

## Instalación

Python 3.11. En la carpeta `Entrega 2/`:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
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

Cada notebook lee de disco lo que dejó el anterior, así que puede ejecutarse por separado. El proyecto completo corre en unos 2 minutos.

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
- **Verificado:** borrando todo lo generado y ejecutando el proyecto desde cero, las 38 tablas, los datos procesados y `config_congelada.json` salen idénticos byte a byte.
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

## Uso de IA generativa

El proyecto se desarrolló con apoyo de un asistente de IA (Claude Code, de Anthropic) para la estructuración del código, la documentación y la revisión metodológica. Las decisiones, la ejecución y la interpretación de los resultados son responsabilidad de los autores.
