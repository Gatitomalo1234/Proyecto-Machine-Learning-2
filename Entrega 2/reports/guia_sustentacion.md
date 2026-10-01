# Guía de sustentación — Segunda entrega

Para Nicolás y María Paula. La rúbrica pide "dominio de todo el equipo": cualquiera de los dos debe poder contestar cualquier pregunta. Las cifras salen del informe (Tablas 1–6, Figuras 1–3).

---

## 1. La historia en un minuto

> Comparamos LightGBM, CatBoost y XGBoost, los tres modelos del artículo, con una regresión lineal que solo usa los movimientos recientes de la TRM. Cada boosting se entrenó con solo la TRM y con la TRM más el Brent, el DXY y el diferencial Banrep–Fed. **Resultado: las variables macro no mejoran la predicción diaria de la TRM, y los boosting empatan con la regresión lineal.** Lo único predecible es que parte del movimiento de hoy continúa mañana, porque la TRM es el promedio de las operaciones de un día, y esa relación es lineal. Primero reprodujimos el artículo tal cual y falla: los árboles no pueden predecir por encima del máximo que vieron. Por eso predecimos el retorno.

## 2. Guion por sección (unos 10 minutos)

| Sección | Qué decir | Apoyo | Cifras clave |
|---|---|---|---|
| Pregunta | La de la Entrega 1, textual: ¿las macro mejoran la predicción diaria y qué boosting las aprovecha mejor? | Introducción | — |
| 7.1 Datos | 5 series oficiales, dic 2003 – sep 2026, congeladas con MD5. Sin duplicados; los faltantes del Brent son festivos; los extremos son reales y se conservan. Sin simulación. | Tabla 1 | 5.401 días = 4.320 desarrollo + 1 purga + 1.080 test |
| 7.1 Transformaciones | Objetivo: retorno ln(TRM mañana / TRM hoy). Variables de la Entrega 1 como cambios diarios; el diferencial, en nivel. El EDA lo justifica. | Tabla 2 | Autocorrelación 0,17; DXY +0,09; Brent −0,06; R² lineal ≈ 3 % |
| 7.2 Sin fuga | Cada fila usa solo lo conocido antes del día. Ejemplo del miércoles. Pruebas: assert, perturbación, partición cronológica con purga. Escalado y búsqueda dentro del Pipeline. Test abierto una vez. | Figura 1 | 80/20; 1 día purgado; gap de 5 |
| 7.3 Artículo tal cual | Nivel + MinMax + 80/20 + parámetros por defecto: los árboles se quedan bajo el máximo del entrenamiento. | Figura 2 | Desarrollo: techo 3.435, RMSE 317–333 vs. 28 de la lineal. Test: techo 4.100 vs. TRM de 5.061 |
| 7.3 Adaptación | Qué cambiamos frente al artículo y por qué; lo local: petróleo y diferencial con la Fed. | Tabla 3 | — |
| 7.3 Modelos y ajuste | Línea base + 3 boosting × {solo TRM, TRM + macro}. 15 combinaciones al azar, CV temporal de 3 cortes. | Tabla 4 | 5 de 6 con profundidad 2; tasa ≤ 0,05; 100–400 árboles |
| 7.3 Comparación | Empate. Las macro no ayudan; LightGBM es el que menos pierde. | Tabla 5 | Validación: CatBoost 20,71 vs. base 20,83. Test: base 35,63 vs. CatBoost 35,64. Macro en test: +0,11 / +0,54 / +0,67 |
| 7.4 Validación y métricas | Anidada: 5 cortes externos, búsqueda solo dentro de cada entrenamiento. RMSE principal; MAE; R² sobre retornos. El desbalance no aplica. | Tabla 5 | Desviación entre cortes ≈ 7 pesos |
| 7.4 Interpretación | SHAP: el movimiento de ayer pesa 82 %. Con macro, el diferencial (31 %) aprende "alto = devaluación", y en el test fue alto durante una revaluación: cambio de régimen, no fuga. | Figura 3 | Umbral ≈ 7,8 pp |
| 7.4 Estabilidad y errores | Optuna no cambia nada (≤ 0,22). El orden de los modelos sí cambia. El error sigue a la volatilidad. | Tabla 6 | RMSE de la base: 46 / 35 / 28 por periodo |
| Cierre | Para un importador: TRM de hoy + 17 % de su movimiento, error típico de 27 pesos. Preliminar. Pendiente para la entrega final: el régimen del diferencial. | — | MAE en test: 27,10 |

**Reparto sugerido:** Nicolás presenta 7.1–7.2 y María Paula 7.3–7.4, y luego intercambian en las preguntas. Ensayen al revés al menos una vez.

## 3. Preguntas probables con respuesta corta

**Sobre el planteamiento**

1. **¿Por qué predecir el retorno y no el precio, como el artículo?** Porque un árbol predice promedios de lo que vio y no puede superar el máximo del entrenamiento (Figura 2). El retorno se mueve en el mismo rango en todas las épocas. Además, el precio no es estacionario. El precio se reconstruye (TRM hoy · e^ŷ) para medir el error en pesos.
2. **¿Por qué el R² sobre retornos y no sobre el precio?** Porque en nivel copiar la TRM de hoy ya da 0,99: no distingue un buen modelo de uno trivial. Sobre retornos mide cuánto del movimiento diario se anticipa.
3. **¿Por qué una regresión lineal como línea base?** Es lo que prometimos en la Entrega 1: un modelo simple del nivel de Machine Learning I. Usa las mismas tres variables que los boosting "solo TRM", así que la comparación es justa.
4. **¿Es un modelo de series de tiempo (ARIMA, autorregresivo)?** No. Es una regresión lineal con variables construidas con datos pasados. No modelamos la estructura de la serie; la tratamos como un problema de regresión supervisada con partición temporal.
5. **¿Por qué solo Brent y no WTI?** El crudo colombiano se referencia al Brent, los dos son casi idénticos y el WTI fue negativo en abril de 2020.

**Sobre los datos y la fuga**

6. **¿Qué es la fuga de datos y cómo la evitaron?** Es usar en el entrenamiento información que no se tendría al momento de predecir. La evitamos con la alineación hacia atrás (solo datos con fecha anterior), la partición cronológica con un día purgado, el Pipeline (el escalado aprende solo del entrenamiento), el EDA y la selección solo con desarrollo, y el test abierto una sola vez con la configuración congelada.
7. **Ejemplo concreto de alineación.** La TRM del miércoles se calcula con las operaciones del martes. El modelo ve el Brent y el DXY del cierre del martes y la tasa de la Fed del lunes, porque la Fed la publica al día siguiente.
8. **¿Por qué se elimina un día entre desarrollo y test?** El objetivo de la última fila del desarrollo es la TRM del día siguiente, que ya pertenece al test.
9. **¿Rellenar festivos no es imputar con fuga?** No: se usa el último dato disponible antes del día, nunca uno posterior. Solo mira al pasado.
10. **¿Qué es la prueba de perturbación?** Alteramos todos los datos posteriores a una fecha y comprobamos que ninguna variable anterior cambia. Si cambiara alguna, habría fuga.

**Sobre la validación y el ajuste**

11. **¿Qué es la validación anidada y por qué la usaron?** Un bucle externo de 5 cortes cronológicos mide el desempeño, y dentro de cada entrenamiento un bucle interno (3 cortes) elige los hiperparámetros. Así nunca se elige y se mide con los mismos datos, lo que daría un error optimista, y obtenemos 5 mediciones con su variabilidad.
12. **¿Por qué búsqueda al azar con 15 intentos?** Es un presupuesto modesto, porque cada combinación se entrena en 3 cortes dentro de cada uno de los 5 cortes externos. Para saber si alcanzaba lo comprobamos: con Optuna (bayesiana) y con 50 intentos, el error cambia menos de 0,22 pesos (Tabla 6).
13. **¿Optuna se usó para los modelos finales?** No. Fue solo una prueba de robustez; los modelos finales salen de la búsqueda al azar.
14. **¿Por qué esos rangos de hiperparámetros?** La señal es débil (R² ≈ 3 %) y pocos días extremos dominan el error, así que favorecemos árboles poco profundos, tasas bajas y hojas con muchas observaciones. Los modelos eligieron justamente eso.
15. **¿Por qué el error del test (≈ 35,6) es mayor que el de validación (≈ 20,8)?** Porque 2022–2026 fue más volátil. La distancia entre modelos sigue siendo de centavos.

**Sobre los resultados**

16. **Entonces, ¿cuál es el mejor modelo?** En validación, CatBoost con solo TRM, que es el que congelamos. En el test, la línea base, con CatBoost a 0,01 pesos. Es un empate: 0,13 pesos de ventaja frente a 7 pesos de variación entre periodos.
17. **¿Un resultado negativo vale?** Sí. La Entrega 1 ya lo anticipaba: si el modelo con macro no supera la línea base, las macro no aportan. Está medido con validación anidada y un test limpio.
18. **¿Por qué las macro no ayudan si el petróleo importa para Colombia?** Su relación con la TRM del día siguiente es muy débil (correlaciones de 0,06 a 0,09). Además, el diferencial de tasas cambió de régimen: en el desarrollo los valores altos coincidían con devaluaciones y en el test no.
19. **¿El problema del diferencial es fuga?** No. Es un cambio de distribución: el modelo no podía saber que el régimen cambiaría. Es la principal limitación y queda para la entrega final.
20. **¿Qué es SHAP y qué dice?** Reparte cada predicción entre las variables. Dice que el movimiento de ayer explica el 82 % de la importancia y empuja la predicción en su misma dirección.
21. **¿Para qué le sirve esto a alguien?** Un importador puede usar la TRM de hoy más el 17 % de su movimiento, con un error típico de 27 pesos. A un día, el boosting no justifica su complejidad.
22. **¿Qué es reproducible exactamente?** Semilla 42, un hilo de cómputo y datos congelados con MD5. Corriendo todo desde cero, las tablas salen idénticas byte a byte en unos 2 minutos.

**Sobre la IA**

23. **¿Cómo usaron la IA?** Como asistente para estructurar el código, revisar la metodología frente a la rúbrica y redactar borradores. Cada decisión la discutimos y la podemos defender, y verificamos cada cifra contra las salidas del código.

## 4. Antes de entrar

- [ ] Cada uno explica sin leer: la Figura 2 (techo), la Tabla 5 (empate) y la Figura 3 (SHAP).
- [ ] Cada uno da el ejemplo del miércoles sin ayuda.
- [ ] Tener abiertos el PDF, el README y el notebook 06 por si piden mostrar el código.
- [ ] Repasar las cifras: 5.401 / 20,71 vs. 20,83 / 35,63 vs. 35,64 / 82 % / 0,22 / 27 pesos.
