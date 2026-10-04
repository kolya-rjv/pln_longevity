import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "pln_chat"))   # repo/pln_chat
import api
from core.pln_runner import patient_stack, run_query, linage2_patient_kb
job = json.loads(sys.argv[1])
here = Path(__file__).resolve().parent
if job["stack"] == "linage2":
    kb = list(linage2_patient_kb())
else:
    kb = list(api._runtime_kb_paths())
    if job["stack"] == "patient":
        kb = patient_stack(kb)
swaps = job.get("swap", {})
kb = [ (here / swaps[p.name]) if p.name in swaps else p for p in kb ]
t = time.time()
r = run_query(job["query"], kb_files=kb, extra_atoms=job.get("extra") or None)
print("RESULT " + json.dumps({"status": r.status, "atoms": [x.atom for x in r.results], "secs": round(time.time()-t, 2)}))
