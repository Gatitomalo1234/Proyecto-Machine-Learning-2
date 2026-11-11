"""Prueba de humo de la aplicación Streamlit: arranca y recorre sus modos sin errores."""
from __future__ import annotations

import datetime
from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_app_arranca_y_recorre_los_modos() -> None:
    """La app carga las 4 pestañas, permite escribir TRM y simular una fecha de crisis (que dispara alertas)."""
    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app" / "app.py"), default_timeout=180).run()
    assert not at.exception and len(at.tabs) == 4
    at.radio[0].set_value("Escribir las últimas TRM").run()
    assert not at.exception
    at.radio[0].set_value("Una fecha pasada").run()
    at.date_input[0].set_value(datetime.date(2008, 10, 10)).run()
    assert not at.exception
    assert any(estado in " ".join(m.value for m in at.markdown) for estado in ("ALERTA", "AVISO"))
