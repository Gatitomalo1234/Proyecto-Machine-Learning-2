# Aplicación: predicción de la próxima TRM

Despliegue básico del modelo final del proyecto (rúbrica 8.3). Toma las últimas TRM publicadas por la Superintendencia Financiera, calcula las variables del modelo con **las mismas fórmulas del entrenamiento**, predice la próxima TRM con el modelo final y con la línea base, explica la predicción y verifica que los datos de uso sean coherentes con los de entrenamiento.

## Cómo ejecutarla

Desde la carpeta `Entrega 2/`, con el entorno del proyecto instalado (`pip install -r requirements.txt`):

```bash
streamlit run app/app.py
```

Se abre en `http://localhost:8501`. Necesita internet para la TRM en vivo; sin internet funciona igual con la copia congelada de los datos y lo indica.

Pruebas automáticas (núcleo y aplicación):

```bash
python -m pytest            # 14 pruebas; -m "not red" omite la que consulta datos.gov.co
```

## Qué muestra

| Pestaña | Contenido |
|---|---|
| **Mañana** | TRM vigente hoy (en vivo), próxima TRM predicha por el modelo final (CatBoost con solo la TRM) y por la regresión lineal, movimiento esperado y rango del 80 %, que se ensancha con la volatilidad de las últimas 40 TRM (notebook 10; con menos de 41 TRM se usa un rango fijo) |
| **Por qué** | Cuánto aporta a la predicción el movimiento de hoy, el de hace 5 y el de hace 10 TRM (SHAP en CatBoost; aporte exacto en la lineal), en lenguaje sencillo |
| **Confianza** | Semáforo de coherencia entre entrenamiento y uso, error histórico en el test (2022–2026) y empate con la línea base |
| **Simular** | Predicción para una fecha pasada (con la TRM que se publicó después) o con TRM escritas a mano |
| **Modelo** | Ficha del modelo final (`models/ficha_modelo_final.md`) |

**"Próxima TRM"** es la siguiente que publique la Superintendencia Financiera: después de un viernes, la del lunes (o la del martes si el lunes es festivo).

## De dónde salen los datos y los modelos

- **TRM en vivo:** datos.gov.co, conjunto `32sa-8pi3` (Superintendencia Financiera), la misma fuente y el mismo formato de la copia congelada `data/raw/trm_vigencias_datosgov.csv`. Se consulta como máximo una vez por hora.
- **Modelos:** `models/modelo_final_despliegue.joblib`, el modelo final y la línea base reentrenados con 2003–2026 y los hiperparámetros congelados (notebook 08). Al cargarlos se comparan sus predicciones en 25 días de control con las guardadas en `models/modelo_final.json`, con una tolerancia de 1e-9 (otro sistema operativo, como el Linux de Streamlit Cloud, puede cambiar el último decimal sin que el modelo cambie).
- **Lógica:** toda en `src/prediccion.py`; la aplicación solo la llama.

## Coherencia entre entrenamiento y uso (semáforo)

| Chequeo | Verde | Qué significa si no |
|---|---|---|
| Datos recientes | La última TRM tiene 4 días o menos | La fuente no se ha actualizado o se usa la copia congelada |
| Historia completa | Hay al menos 11 TRM seguidas, sin saltos mayores que en el entrenamiento (6 días) | No se pueden calcular bien los rezagos |
| Dentro de lo visto | Los tres movimientos están en el 99 % central del entrenamiento | La predicción es una extrapolación |
| Régimen de volatilidad | La volatilidad de las últimas 20 TRM no está en el tercio más alto | Con volatilidad alta, el movimiento de hoy se repite menos mañana (0,14 frente a 0,23) |
| Modelo correcto | Las predicciones de control coinciden (tolerancia 1e-9) | El archivo del modelo no es el documentado |

**La garantía de fondo:** las pruebas comprueban que las variables que calcula la aplicación son idénticas, al bit, a las de `dataset.csv` en los 5.401 días, y que sus predicciones coinciden con las del notebook 06.

## Publicarla en Streamlit Community Cloud

1. En [share.streamlit.io](https://share.streamlit.io), **Create app** → desde el repositorio `Gatitomalo1234/Proyecto-Machine-Learning-2`, rama `main`.
2. **Main file path:** `Entrega 2/app/app.py`.
3. **Advanced settings → Python version:** 3.12 (las versiones de `app/requirements.txt` son las del entrenamiento; scikit-learn y catboost deben coincidir para cargar el modelo).
4. Streamlit instala `app/requirements.txt` (está junto a `app.py`). El tema oscuro está en `.streamlit/config.toml` (copia en la raíz del repositorio para la nube y en `Entrega 2/` para uso local); el estilo principal va en el CSS de `app/estilo.py`, así que no depende de él.
