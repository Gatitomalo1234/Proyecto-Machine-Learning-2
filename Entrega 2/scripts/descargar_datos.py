"""Descarga y congela los datos crudos del proyecto. Se ejecuta UNA vez.

Uso (desde la carpeta Entrega 2/):
    python scripts/descargar_datos.py            # descarga solo lo que falte
    python scripts/descargar_datos.py --forzar   # vuelve a descargar todo (cambia la copia congelada)

Deja en data/raw/ los archivos tal como llegan de cada fuente y un metadatos.json con
la fuente, la URL, el momento de descarga, la huella MD5, el número de filas y el rango de fechas.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import DATA_RAW              # noqa: E402
from src.descarga import FUENTES, descargar, guardar, md5   # noqa: E402

RUTA_META = DATA_RAW / "metadatos.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--forzar", action="store_true", help="volver a descargar aunque exista la copia congelada")
    args = parser.parse_args()

    meta = json.loads(RUTA_META.read_text(encoding="utf-8")) if RUTA_META.exists() else {}

    for nombre, info in FUENTES.items():
        ruta = DATA_RAW / info["archivo"]
        if ruta.exists() and nombre in meta and not args.forzar:
            if md5(ruta) != meta[nombre]["md5"]:
                raise RuntimeError(f"{ruta.name} no coincide con su MD5 registrado: la copia congelada fue modificada")
            print(f"{nombre:14s} ya congelada ({ruta.name}), no se descarga")
            continue
        print(f"{nombre:14s} descargando…")
        meta[nombre] = guardar(nombre, descargar(nombre))
        m = meta[nombre]
        print(f"{'':14s} {m['filas']:,} filas, {m['fecha_min']} a {m['fecha_max']}, MD5 {m['md5']}")

    RUTA_META.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nMetadatos en {RUTA_META.relative_to(DATA_RAW.parents[1])}")


if __name__ == "__main__":
    main()
