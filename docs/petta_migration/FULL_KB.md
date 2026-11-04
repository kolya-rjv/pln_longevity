# Full application KB on PeTTa

Measured 2026-11-04 on x86_64 Linux, Python 3.10, SWI-Prolog 10.0.2,
PeTTa `fea83178af3722666d2948d074591d10a3134f56`.

With `PLN_RUNTIME=petta`, every production execution discovers all application
`.metta` sources in the repository root, `build/`, and
`pln_chat/ontology/metta_files/`. It does not use the Hyperon byte cap or choose
files based on the query. Explicit additional sources and request-specific atoms
join the same space. Unreadable sources fail the execution rather than silently
leaving a partial KB. Hyperon retains its existing scoped loading policy.

Tests, examples and vendored runtime libraries are not application KB sources.
`pln_chainer.metta` is the alternative implementation of `infer`/`explain`;
`core.pln_chainer` installs it only for `PLN_CHAINER=pln`. Loading it beside the
legacy implementations would add unintended answers. No application evidence is
excluded. Identical facts/declarations repeated in the committed sample,
generated corpus, or an injected slice are loaded once; distinct facts remain.

Mixed LinAge2/generic programs run against one full KB on PeTTa. The full patient
facts are available there too. Validation uses the full execution inventory.

## Full local dataset

The reproducible measurement in [full_kb_results.json](full_kb_results.json):

| Measurement | Result |
|---|---:|
| Application files (including all six local generated datasets) | 38 |
| Source top-level forms | 85,481 |
| Forms after deduplicating copied facts/declarations | 82,243 |
| KB text | 4,339,924 bytes |
| PeTTa load time | 13.79 seconds |
| Query time for the four probes together | 0.20 seconds |
| Child peak RSS | 1,240 MB |
| DrugAge Experiment records returned by a match/count | 3,423 |
| CellAge CellSenescenceRecord records returned by a match/count | 927 |

The same execution also answered a curated causal inference and a generated
GenAge gene lookup. Regression tests independently count Experiment ids from
the source files, check patient functions, and execute a mixed LinAge2/gene
program through `POST /metta/run` with the full KB. No LLM calls are used.

The previous 512 MB child-memory default cannot hold this generated corpus. The
default is now 2048 MB; `PETTA_MEMORY_LIMIT_MB` can still override it. Limits are
per child, and each execution still reloads the KB. These timings include no
persistent loaded-KB cache. More evidence can produce more inference routes;
this check is not a proof that all old ranking outputs remain unchanged.

## Million-fact probe

[full_kb_million_results.json](full_kb_million_results.json) records a separate
test with the full KB plus 1,000,000 synthetic ground facts (28,228,814 bytes
total), a 3072 MB memory-growth limit, a 180-second deadline and the default
2048 MB Prolog stack limit.

It **failed during loading**, after 22.78 seconds, with `Stack limit (2.0Gb)
exceeded` in PeTTa's `top_forms`/`grab_until_balanced` parser. Query execution
was not reached. Loading all current application data succeeds; million-atom
capacity under the default loader/stack configuration is not established.

## Reproduce

```bash
python scripts/check_petta_full_kb.py --output /tmp/full-kb.json
python scripts/check_petta_full_kb.py --synthetic-atoms 1000000 \
  --memory-limit-mb 3072 --deadline 180 --output /tmp/million-kb.json
PLN_RUNTIME=petta PLN_CHAINER=legacy python -m pytest tests/test_full_kb.py -q
```

The script reports failures with a nonzero exit status. Generated `build/`
files are local ETL outputs; the JSON report lists the exact files measured.
