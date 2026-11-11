# Proyecto Integrador — Machine Learning II

Universidad Externado de Colombia.
Integrantes: Nicolás Cárdenas Díaz y María Paula Muñoz Malaver.

**Pregunta.** ¿La incorporación de variables macroeconómicas locales (petróleo, diferencial de tasas Banrep–Fed, fortaleza del dólar) mejora la predicción del precio diario de la TRM frente a un modelo que solo usa su comportamiento pasado, y cuál de los tres algoritmos de gradient boosting aprovecha mejor esa información?

**Artículo base.** Eslami Ghayour, A., y Yousefi, A. (2023). *Performance Analysis of Gradient Boosting Models for Forex Market Prediction*. SSRN, https://ssrn.com/abstract=4450688

| Carpeta | Fase | Contenido |
|---|---|---|
| [`Entrega 1/`](Entrega%201/entregable) | Comprensión del artículo y planteamiento | Informe entregado (LaTeX y PDF) |
| [`Entrega 2/`](Entrega%202) | Datos, modelamiento y validación | Código reproducible, datos congelados, notebooks, informe y guía de sustentación. Instrucciones en su [README](Entrega%202/README.md) |

## Entrega final (Fase 3)

La entrega final se construyó sobre la carpeta `Entrega 2/` (mismo código, datos y notebooks, ampliados con los notebooks 07 a 10).

| Qué | Dónde |
|---|---|
| Informe final (PDF) | [`Entrega 2/reports/informe_final/informe_final.pdf`](Entrega%202/reports/informe_final/informe_final.pdf) |
| Aplicación en línea | https://modelo-proyectoml2-trm.streamlit.app/ (código en [`Entrega 2/app/`](Entrega%202/app)) |
| Presentación (PDF) y guion | [`Entrega 2/reports/presentacion/`](Entrega%202/reports/presentacion) |
| Código y cómo reproducir | [`Entrega 2/README.md`](Entrega%202/README.md) |

**Respuesta corta.** Las variables macro no mejoran la predicción diaria de la TRM (en el test la empeoran, p de 0,01 a 0,04); ningún gradient boosting le gana a una regresión lineal; lo único predecible es que mañana repite una pequeña parte del movimiento de hoy. El modelo final es CatBoost con solo la TRM, en empate estadístico con la regresión lineal.
