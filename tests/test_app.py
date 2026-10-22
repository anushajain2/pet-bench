from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest

ROOT=Path(__file__).resolve().parents[1]
pytestmark=pytest.mark.skipif(not (ROOT/'reports/run_manifest.json').exists(),reason='Run pet-scout all first')


def test_dashboard_initial_render_and_interaction():
    app=AppTest.from_file(str(ROOT/'original_app.py'),default_timeout=30).run()
    assert not app.exception
    assert 'plastic recycling' in app.title[0].value
    app.selectbox[0].select('random').run()
    assert not app.exception
    app.toggle[0].set_value(False).run()
    assert not app.exception
