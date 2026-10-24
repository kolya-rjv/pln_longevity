# fix/: the fix phase of the pln chainer (2026-10-24)

* `NOTES.md`: the 18 findings of the two reviews (engineering and semantics lenses), each reproduced and fixed or explained, the null-convention decision, before/after measurements, the mutation check and the verification runs.
* `COST_MODEL.md`: what a `PLN.Derive` run costs, the model fitted to 19 graphs, and what the chainer refuses up front.
* `probes/`: the scripts that produced the numbers. `fx.py` (harness), `repro.py <scenario>` (one JSON line per finding: zero, rank, exact, skip, back, badcfg, dabad, readans, order, cost_dag, cost_digraph5, cost_chain200,
  cost_nolookup, cost_feedback), `calib_lib.py` + `calib2.py` (the calibration), `timing.py` (request cost, legacy against pln), `shapes_prod.py` (the 145 query shapes through the production path), `mk_etl.py` (the DrugAge build, regenerated outside the repo).
  Set `FIX_OUT=<dir>` to keep their outputs; nothing is written under the repo. They need PeTTa (`bash scripts/setup_petta.sh`).
* `results/calib2.txt`: the lines of the calibration run.
