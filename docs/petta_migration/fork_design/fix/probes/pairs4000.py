from fx import *
extra = "".join(f"(Effect S{i} T{i} Pos (stv 0.5 0.8))\n" for i in range(4000))
print("chars", len(extra))
for label, q in (("one source asked", ["!(infer &self S7 T7)"]), ("every source (unbound)", ["!(infer &self $a $b)"])):
    for mode in ("legacy", "pln"):
        use(mode, PLN_QUERY_TIMEOUT_SECONDS=60)
        r, dt = run(MIN, extra, q)
        print(label, mode, r.status, len(r.results), f"{dt:.2f}s", (r.error or "")[:80])
