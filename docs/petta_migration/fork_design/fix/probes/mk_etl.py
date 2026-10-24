"""Regenerate the DrugAge ETL build into scratch (nothing is written under the work tree)."""
import importlib.util, sys, tempfile, zipfile, os
from pathlib import Path
os.environ["PYTHON_DOTENV_DISABLED"] = "1"
REPO = Path(__file__).resolve().parents[5]
spec = importlib.util.spec_from_file_location("drugage_etl_fx", REPO / "drugage_etl.py")
etl = importlib.util.module_from_spec(spec); sys.modules["drugage_etl_fx"] = etl; spec.loader.exec_module(etl)
FIX = Path(os.environ["FIX_OUT"]) if os.environ.get("FIX_OUT") else Path(tempfile.mkdtemp(prefix="pln_fix_"))
OUT = FIX / "build_out" / "drugage_etl.metta"
OUT.parent.mkdir(parents=True, exist_ok=True)
with tempfile.TemporaryDirectory() as tmp:
    csv_path = Path(tmp) / "drugage.csv"
    with zipfile.ZipFile(REPO / "data" / "drugage" / "dataset.zip") as z:
        csv_path.write_bytes(z.read("drugage.csv"))
    etl.generate(csv_path, OUT)
print(OUT, OUT.stat().st_size)
