"""Alineación temporal sin fuga de información (rúbrica 7.2).

La TRM es la columna vertebral: una fila por TRM publicada, con fecha t = inicio de su vigencia.
Regla anti-fuga: en la fila t solo entra información disponible ANTES del inicio del día t.
La TRM_{t+1} (el objetivo) se forma con las operaciones del día t, así que cualquier dato
conocido antes de que empiece t es información legítima y ninguno posterior lo es.
"""
import pandas as pd

# Días que tarda en publicarse el dato del día d (se suma a su fecha para saber cuándo se conoce)
REZAGO_PUBLICACION = {
    "brent": 0,  # cierre del día d, conocido al final de d
    "dxy": 0,    # cierre del día d, conocido al final de d
    "tpm": 0,    # el Banrep anuncia la tasa antes de que entre en vigencia
    "dff": 1,    # la Fed de Nueva York publica la DFF del día d la mañana de d+1
}


def alinear(trm, macros):
    """Une a cada TRM publicada el último valor de cada variable macro conocido antes de su fecha.

    trm:    DataFrame(fecha, valor) con una fila por TRM publicada.
    macros: {nombre: DataFrame(fecha, valor)} sin faltantes.

    Usa pd.merge_asof(direction="backward", allow_exact_matches=False) sobre la fecha en que el
    dato queda disponible (fecha + rezago de publicación). Nunca interpola: si un día no hay dato
    (festivo), se usa el último valor conocido, que es lo que vería un analista ese día.
    Agrega columnas de auditoría: fecha_<var> (fecha real del dato usado) y antig_<var> (días).
    """
    base = trm.rename(columns={"valor": "trm"}).sort_values("fecha").reset_index(drop=True)
    for nombre, df in macros.items():
        d = df.dropna(subset=["valor"]).sort_values("fecha").copy()
        d["disponible"] = d["fecha"] + pd.Timedelta(days=REZAGO_PUBLICACION[nombre])
        d = d.rename(columns={"valor": nombre, "fecha": f"fecha_{nombre}"})
        base = pd.merge_asof(base, d[["disponible", nombre, f"fecha_{nombre}"]],
                             left_on="fecha", right_on="disponible",
                             direction="backward", allow_exact_matches=False).drop(columns="disponible")
        base[f"antig_{nombre}"] = (base["fecha"] - base[f"fecha_{nombre}"]).dt.days
    return base


def verificar_sin_fuga(tabla, macros):
    """Pruebas automáticas: detienen la ejecución si alguna variable usa un dato no disponible."""
    assert tabla["fecha"].is_monotonic_increasing and not tabla["fecha"].duplicated().any()
    for nombre in macros:
        ok = tabla[f"fecha_{nombre}"].notna()
        disponible = tabla.loc[ok, f"fecha_{nombre}"] + pd.Timedelta(days=REZAGO_PUBLICACION[nombre])
        assert (disponible < tabla.loc[ok, "fecha"]).all(), f"{nombre}: usa un dato no disponible antes de t"
    return True


def tabla_relleno(tabla, macros, umbral_dias=5):
    """Cuánto se apoya cada variable en el último valor conocido (festivos, cierres de mercado)."""
    filas = []
    for nombre in macros:
        a = tabla[f"antig_{nombre}"].dropna()
        repetido = tabla[f"fecha_{nombre}"].eq(tabla[f"fecha_{nombre}"].shift())
        filas.append({
            "variable": nombre,
            "rezago_publicacion_dias": REZAGO_PUBLICACION[nombre],
            "filas_con_dato_repetido": int(repetido.sum()),
            "pct_repetido": 100 * repetido.mean(),
            "antiguedad_mediana_dias": a.median(),
            "antiguedad_p95_dias": a.quantile(0.95),
            "antiguedad_max_dias": a.max(),
            f"filas_con_dato_de_mas_de_{umbral_dias}_dias": int((a > umbral_dias).sum()),
        })
    return pd.DataFrame(filas).set_index("variable")
