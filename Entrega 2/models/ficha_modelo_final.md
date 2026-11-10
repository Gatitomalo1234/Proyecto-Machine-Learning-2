# Ficha del modelo final — Predicción diaria de la TRM

**Modelo final:** CatBoost · solo_trm. **Referencia:** Línea base (lineal). Generada por `notebooks/08_modelo_final.ipynb`.

## Para qué sirve
Predice el **movimiento de la TRM del día siguiente** y, con él, la TRM de mañana en pesos. Responde la pregunta del proyecto: las variables macro no mejoran la predicción diaria y ningún boosting se distingue de una regresión lineal.

## Qué predice y con qué
- **Objetivo:** y = ln(TRMₜ₊₁ / TRMₜ), el retorno logarítmico del día siguiente. La TRM de mañana se reconstruye como TRMₜ · e^ŷ.
- **Variables (3):** retorno logarítmico de la TRM de hoy, de hace 5 días y de hace 10 días: r_trm_lk = ln(TRMₜ₋ₖ₊₁ / TRMₜ₋ₖ).
- **Por qué retornos y no el precio:** los árboles no predicen fuera del rango que vieron (la TRM se salió en el 59 % de los días de evaluación del método del artículo); el retorno se mueve en el mismo rango en todas las épocas. Para movimientos diarios, el retorno logarítmico y el simple son prácticamente iguales (correlación 0,99995); el logarítmico se suma entre días y se convierte exacto en precio.

## Configuración
- **Algoritmo:** CatBoost, en un Pipeline de un paso, sin escalado.
- **Hiperparámetros:** subsample = 0,8; rsm = 0,8; min_data_in_leaf = 50; learning_rate = 0,03; l2_leaf_reg = 3,0; iterations = 200; depth = 2.
- **Fijos:** semilla 42, un hilo, muestreo Bernoulli.
- **Cómo se eligió:** menor RMSE medio en validación entre las 6 configuraciones de boosting; se mantiene si no es significativamente peor que la base (Diebold-Mariano al 5 %).

## Desempeño
| | Validación (5 cortes) | Test (1.080 días) |
|---|---|---|
| RMSE (pesos) | 20,71 | 35,64 |
| MAE (pesos) | 14,77 | 27,09 |
| R² sobre retornos | 0,025 | 0,037 |
| Diferencia de RMSE frente a la base | -0,157 | +0,013 |
| IC 95 % de la diferencia (bootstrap por bloques) | [-0,489 a +0,125] | [-0,121 a +0,227] |
| p-valor Diebold-Mariano | 0,319 | 0,898 |

**Lectura:** empate estadístico con la regresión lineal. Lo único predecible a un día es la continuación del movimiento de hoy (movimiento de mañana ≈ 0,17 × movimiento de hoy).

## Archivos y trazabilidad
- Congelado y evaluado en el test: `models/modelos_finales.joblib` (huella `bb2e506e5150f07a3dd7d6e7d32346b4`), configuración `config_congelada.json` (MD5 `580c4f306e227c6c677ede6291b6eef2`).
- Para la aplicación: `models/modelo_final_despliegue.joblib` (modelo final y línea base), entrenados con 5.401 días (2003-12-17 a 2026-09-26) y los mismos hiperparámetros; huellas `0393ceb696b9451db152bfb6dcffb13b` y `441c87fbedb64dd190888713409f6778`.
- Datos: `data/processed/dataset.csv` (MD5 `38929ed82d75910fba7bb526ab3d16f2`).

## Limitaciones
- Empate estadístico con la regresión lineal: la ventaja es de centavos.
- Un solo periodo de test (2022–2026), ya abierto en la Entrega 2.
- La memoria de un día se debilita con volatilidad alta; los movimientos grandes no se anticipan.
- Solo predice el día siguiente; no sirve para horizontes más largos.
