# NHANES microdata — how to obtain it

**No NHANES file is bundled in this repository, and no NHANES-derived number is
committed to it.** This directory contains only `MANIFEST.tsv` (which files to fetch,
and what this repo claims is inside them) and this README. The ETLs read microdata you
download yourself, and write their output to `build/`, which is gitignored.

That is a deliberate rule, not an oversight. A number in a `.metta` file is consumed by
inference and becomes a claim the system will act on, so this repo never ships a
statistic it could not compute from a source it actually read. The environment this
integration was written in cannot reach `wwwn.cdc.gov` or `ftp.cdc.gov` (the egress
policy returns 403 on CONNECT), so every NHANES reference value is left for you to
generate locally.

## 1. Fetch the files

NHANES is public, free, and requires no registration or licence. The public data files
follow one URL pattern:

```
https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{first_year}/DataFiles/{BASENAME}.XPT
```

where `{first_year}` is the first year of the two-year cycle (`1999` for the 1999-2000
cycle, `2001` for 2001-2002) and `{BASENAME}` is the file's short name. `MANIFEST.tsv`
lists every file this repo's ETLs know about, with its full URL already assembled. The
same table is printed, with the evidence behind each entry, by:

```
python3 nhanes_reference_etl.py --show-registry
python3 nhanes_reference_etl.py --manifest          # regenerates MANIFEST.tsv
```

Download them into this directory (`data/nhanes/`), which is where the ETLs look by
default, keeping CDC's own file names:

```
cd data/nhanes
for F in DEMO LAB11 LAB10 LAB10AM; do
  curl -fLO "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/1999/DataFiles/$F.XPT"
done
for F in DEMO_B L11_B L10_B L10AM_B; do
  curl -fLO "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2001/DataFiles/$F.XPT"
done
```

**Do not commit them.** They are large, they are already public at a stable URL, and
redistributing them adds nothing. Note that `.gitignore` does not currently cover
`*.XPT` — verify with `git status` before committing, or add a rule:

```
echo 'data/nhanes/*.XPT' >> .gitignore
```

## 2. Run the ETL

```
python3 nhanes_reference_etl.py --output build/nhanes_reference.metta
```

Files passed with `--demo` / `--lab` override the search of this directory, so the data
can live anywhere:

```
python3 nhanes_reference_etl.py \
    --demo /data/nhanes/DEMO.XPT  --demo /data/nhanes/DEMO_B.XPT \
    --lab  /data/nhanes/LAB11.XPT --lab  /data/nhanes/L11_B.XPT \
    --output build/nhanes_reference.metta
```

Files are matched to the registry by basename, so the order of the flags does not
matter. A file the run needs but cannot find is a hard error that prints its download
URL and the exact local path expected.

## 3. `MANIFEST.tsv` — and why it has a confidence column

One row per (cycle, file, analyte), with the NHANES variable, unit, survey weight,
measurement scale, assay-lot id, a **confidence**, and notes.

The confidence column is the most important one. Every file name, variable name and
assay description in this integration is a factual claim about CDC's published data
that **could not be checked against the source** from the environment this was written
in. Rather than present those claims as certain, each carries its evidence and a
confidence level, and the machinery around them is built so that a wrong guess fails
loudly instead of producing a quietly wrong number:

| confidence | meaning |
|---|---|
| `high` | a naming convention the repo can see evidence for (e.g. the `_B` cycle suffix) |
| `medium` | a specific variable/file claim with a stated basis, unverified against CDC |
| `low` | a claim with a weaker basis, or one about an assay or a subsample weight |

If a name turns out to be wrong, you do **not** need to edit code:

```
python3 nhanes_reference_etl.py --inspect data/nhanes/LAB11.XPT   # what is really in it
echo '{"CRP": {"variable": "LBXHSCRP", "unit": "mg/L"}}' > fix.json
python3 nhanes_reference_etl.py --registry fix.json --output build/nhanes_reference.metta
```

The ETL never falls back to a similarly named variable. CDC explicitly warns that
`LBXSGL` is not a substitute for `LBXGLU`, nor `LBXSCH` for `LBXTC` — they are different
assays on different samples — so a missing variable is an error that lists every column
the file *does* contain, never a search for something that looks close.

## 4. What is and is not emitted

Only analytes whose marker symbol is **declared** in this repo's `.metta` files produce
atoms. At the time of writing that is four:

| symbol | declared in |
|---|---|
| `CRP` | `mechanistic_bridges.metta` |
| `FastingGlucose` | `mechanistic_bridges.metta` |
| `HbA1c` | `mechanistic_bridges.metta` |
| `PlasmaCystatinC` | `grim_age_core.metta` |

The ETL re-checks those declarations by scanning the repo on every run, so a symbol
removed from the knowledge base stops producing atoms rather than leaving a dangling
reference behind.

`MANIFEST.tsv` also lists triglycerides, total and HDL cholesterol, serum creatinine and
serum insulin, marked **`symbol undeclared, not emitted`**. They are listed so the
omission is visible and auditable rather than looking like an oversight. To enable one,
the hand-written MeTTa layer must declare the symbol with its type and scale first;
then flip `emit=True` in the registry.

## 5. Two things about these files that are easy to get wrong

**The survey weight is per analyte, and sometimes not in `DEMO`.** NHANES oversamples,
so an unweighted mean is biased for the US population — but the correct weight depends
on which subsample the analyte was measured in. The MEC-examined analytes use
`WTMEC4YR` from `DEMO`; fasting glucose uses `WTSAF4YR` read **from the fasting lab file
itself**; surplus-sera cystatin C uses `WTSCY4YR`. And a four-year weight is only valid
for the four-year pooled span: a single-cycle run needs the two-year weight, which is
why the ETL looks the weight up by (subsample, pooled cycle set) and refuses to run when
that pair is not in its table.

**Cycles may only be pooled across a common assay.** The trap is CRP: it is mg/dL by
latex nephelometry through 2010, hsCRP in mg/L from 2015, and there is **no CRP or
hsCRP variable at all in 2011-2014** — so a gap in the series is expected rather than a
lost file, and a "pool everything" run would silently average two different quantities.
Every (analyte, cycle) carries an assay-lot id, and pooling two cycles whose lots
disagree is refused, not warned about.

## 6. Citing it

NHANES asks that publications cite the survey. See
<https://www.cdc.gov/nchs/nhanes/> for the current citation text, the survey
documentation, and the analytic guidelines — in particular the guidance on variance
estimation with the strata and PSU design variables, which this repo does **not**
implement (it emits no standard errors; see `nhanes_common.py`).
