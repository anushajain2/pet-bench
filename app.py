"""PETBench dashboard; the first PET Scout study remains in the sidebar."""
from pathlib import Path
from pet_scout.bench_ui import render

render(Path(__file__).resolve().parent)
