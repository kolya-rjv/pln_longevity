import sys, json, subprocess, concurrent.futures as cf, time
jobs = json.load(open(sys.argv[1])); par = int(sys.argv[2]) if len(sys.argv) > 2 else 3
def go(j):
    t = time.time()
    try:
        d = subprocess.run([sys.executable, "run1.py", json.dumps(j)], capture_output=True, text=True, timeout=j.get("timeout", 400))
        line = [l for l in d.stdout.splitlines() if l.startswith("RESULT ")]
        out = json.loads(line[0][7:]) if line else {"status": "abort", "atoms": [], "err": d.stderr[-300:]}
        out["rc"] = d.returncode
    except subprocess.TimeoutExpired:
        out = {"status": "timeout", "atoms": [], "rc": None}
    out["name"] = j["name"]; out["wall"] = round(time.time()-t, 1)
    return out
with cf.ThreadPoolExecutor(par) as ex:
    res = list(ex.map(go, jobs))
json.dump(res, open(sys.argv[1].replace(".json", "_out.json"), "w"), indent=1)
for r in res:
    print(r["name"], "rc", r["rc"], r["status"], r.get("secs"), "|", (" ".join(r["atoms"]))[:400], r.get("err", "")[-200:])
