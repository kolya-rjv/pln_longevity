"""Run MeTTa on PeTTa through PeTTa's own Python interface (python/petta), one fresh process per call (state is
process-global).  argv: spec.json out.json.  spec = {"files": [...], "strings": [...], "exprs": [...]}.
Results (one list of strings per expression) go to out.json, never to stdout (PeTTa prints at fd level)."""
import os, sys, json
os.environ["PYTHON_DOTENV_DISABLED"] = "1"
sys.dont_write_bytecode = True
PETTA = os.environ["PLN_G0_PETTA_HOME"]
sys.path.insert(0, PETTA + "/python")
spec = json.load(open(sys.argv[1]))
from petta import PeTTa
p = PeTTa(verbose=False, petta_path=PETTA)
loaded = []
for f in spec.get("files", []):
    loaded.append([str(x) for x in p.load_metta_file(f)])
for s in spec.get("strings", []):
    loaded.append([str(x) for x in p.process_metta_string(s)])
res = []
for e in spec["exprs"]:
    try:
        res.append({"ok": True, "out": [str(x) for x in p.process_metta_string(e if e.startswith("!") else "!" + e)]})
    except BaseException as exc:
        res.append({"ok": False, "error": "%s: %s" % (type(exc).__name__, str(exc)[:300])})
json.dump({"loaded": loaded, "results": res}, open(sys.argv[2], "w"))
