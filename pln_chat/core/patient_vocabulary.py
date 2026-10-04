"""The words a model may use when it reads a patient: the knowledge base's own.

The My Patient tab can have a language model read free text (core.patient_extract). The
model never writes MeTTa and never chooses a symbol of its own: every identifier it may
return is an enum generated here from the files the rest of the system already trusts —

    smoking status   the `(: X SmokingStatus)` declarations in patient_profile.metta
    LinAge2 inputs   the `(ModelInput <Symbol> LinAge2) ;; <NHANES code> — <description>`
                     lines of linage2_core.metta (59 inputs)
    lab groups       one entry per alias group of the reader's name index
                     (core.patient_text._ALIAS_INDEX): the names that mean the same input or
                     inputs, so "urea" and "urea nitrogen (BUN)" stay apart and "lymphocytes"
                     stays one name for the percentage and the count — the reader, not the
                     model, decides which an entry means, from the unit
    conditions       the 23 NHANES questionnaire items behind fs1Score

`drift()` lists every way these sources disagree with each other or with the reader and
the patient builder; tests/test_patient_vocabulary.py requires it to be empty, so editing a
.metta file without the code (or the reverse) fails a test instead of a patient.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from config import ONTOLOGY_DIR

LINAGE2_MODEL_FILE = ONTOLOGY_DIR / "data" / "linage2" / "linage2_model.json"
_STATUS_RE = re.compile(r"^\(:\s+(\w+)\s+SmokingStatus\)", re.M)
_INPUT_RE = re.compile(r"\(ModelInput (\w+) LinAge2\)[^\n]*?;;\s*(\w+)\s*—\s*([^\n]*)")
#: the questionnaire items that are not diagnoses (self-rated health, its trend, visits)
HEALTH_ITEMS = ("HUQ010", "HUQ020", "HUQ050")
COTININE = "LBXCOT"


@dataclass(frozen=True)
class LabGroup:
    """Names the reader treats alike: any of `aliases` followed by a value means one of
    `codes`, and the reader picks which from the unit (or refuses)."""
    key: str                                   # what the model chooses
    codes: tuple[str, ...]                     # NHANES inputs, in the reader's order
    labels: tuple[str, ...]
    aliases: tuple[str, ...]
    units: tuple[str, ...]                     # accepted units, as a person writes them
    model_units: tuple[str, ...]               # the unit each code takes in LinAge2
    fasting: bool = False                      # the names say the draw was fasting


@dataclass(frozen=True)
class Vocabulary:
    smoking_statuses: tuple[str, ...]
    #: NHANES code -> (KB symbol, description), as linage2_core.metta declares them
    model_inputs: dict
    #: group key -> LabGroup, one per alias group of the reader
    labs: dict
    #: NHANES item -> plain label, the 23 conditions LinAge2's comorbidity score reads
    conditions: dict


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _lab_groups() -> dict:
    """The reader's alias index, grouped: aliases that resolve to the same list of unit
    tables (and say fasting alike) are one group, keyed by a readable name."""
    from core.patient_text import LABS, _ALIAS_INDEX, _FASTING_ALIASES, display_unit

    order = {id(spec): i for i, spec in enumerate(LABS)}
    grouped: dict[tuple, list[str]] = {}
    specs_of: dict[tuple, list] = {}
    for alias, specs in _ALIAS_INDEX:
        key = (tuple(id(s_) for s_ in specs), alias in _FASTING_ALIASES)
        grouped.setdefault(key, []).append(alias)
        specs_of[key] = specs
    groups = {}
    for key in sorted(grouped, key=lambda k: (min(order[i] for i in k[0]), k[1])):
        specs, fasting = specs_of[key], key[1]
        aliases = tuple(a for s_ in specs for a in s_.aliases if a in grouped[key])
        aliases = tuple(dict.fromkeys(aliases))
        if fasting:
            name = "fasting glucose"
        elif len(specs) == 1:
            name = specs[0].label.lower()
        else:
            name = specs[0].aliases[0]
        groups[name] = LabGroup(
            key=name, codes=tuple(s_.code for s_ in specs), labels=tuple(s_.label for s_ in specs),
            aliases=aliases,
            units=tuple(dict.fromkeys(display_unit(u) for s_ in specs for u in s_.units)),
            model_units=tuple(s_.unit for s_ in specs), fasting=fasting)
    return groups


@lru_cache(maxsize=1)
def vocabulary() -> Vocabulary:
    from core.patient_text import _ITEM_LABEL

    statuses = tuple(_STATUS_RE.findall(_read(ONTOLOGY_DIR / "patient_profile.metta")))
    inputs = {code: (symbol, desc.strip())
              for symbol, code, desc in _INPUT_RE.findall(_read(ONTOLOGY_DIR / "linage2_core.metta"))}
    model = json.loads(_read(LINAGE2_MODEL_FILE))
    labs = _lab_groups()
    items = [q for q in model["questionnaire_items"] if q not in HEALTH_ITEMS]
    conditions = {q: _ITEM_LABEL.get(q, q) for q in items}
    return Vocabulary(statuses, inputs, labs, conditions)


#: A unit inside a description's parentheses: "(mmol/L)", "(beats/min, 60-sec pulse)"
_PAREN = re.compile(r"\(([^()]*)\)")


def _described_units(description: str, known: set) -> list[str]:
    """The normalised units a description names, in order, among `known`."""
    from core.patient_text import normalise_unit

    out = []
    for group in _PAREN.findall(description):
        for part in re.split(r"[,;:=]", group):
            u = normalise_unit(part)
            if u in known:
                out.append(u)
    return out


def drift() -> list[str]:
    """Every disagreement between the KB, the LinAge2 model file, the reader and the
    builder that would let the model's vocabulary and the knowledge base part ways."""
    from core import linage2_model as lm
    from core.patient_builder import SMOKING
    from core.patient_text import (
        _ALIAS_INDEX,
        _FS1_ITEMS,
        _HEALTH,
        _ITEM_LABEL,
        _TREND,
        LABS,
        SPECS,
        _cotinine_level,
        _visits_category,
        display_unit,
        normalise_unit,
    )

    v = vocabulary()
    model = json.loads(_read(LINAGE2_MODEL_FILE))
    out: list[str] = []
    if set(v.smoking_statuses) != set(SMOKING):
        out.append(f"patient_profile.metta declares smoking statuses {sorted(v.smoking_statuses)}, "
                   f"the patient builder accepts {sorted(SMOKING)}")
    features = {f["code"] for f in model["features"]}
    if set(v.model_inputs) != features:
        out.append(f"linage2_core.metta ModelInput codes and the model file's features differ: "
                   f"{sorted(set(v.model_inputs) ^ features)}")
    derived = model["derived"]
    sources = {s for d in derived.values() for s in d["from"]}
    typed = set(SPECS) | {COTININE}
    for code in v.model_inputs:
        if code not in typed and code not in derived:
            out.append(f"KB input {code} can be neither typed nor derived")
    for code in model["lab_inputs"]:
        if code not in typed:
            out.append(f"model lab input {code} has no unit table in the reader")
    for code in sources - set(model["questionnaire_items"]):
        if code not in typed:
            out.append(f"derived-input source {code} has no unit table in the reader")
    for code in set(SPECS) - set(model["lab_inputs"]) - sources:
        if code not in v.model_inputs:
            out.append(f"the reader reads {code}, which no model input uses")
    if set(_FS1_ITEMS) != set(v.conditions):
        out.append(f"the reader's 23 conditions and the model file's questionnaire differ: "
                   f"{sorted(set(_FS1_ITEMS) ^ set(v.conditions))}")
    if set(_ITEM_LABEL) != set(v.conditions):
        out.append("every condition needs a plain label in the reader")
    if set(lm.FS1_ITEMS) | {"MCQ160B"} != set(v.conditions):
        out.append(f"the 23 conditions and linage2_model's fs1 items (+ MCQ160B, read but not "
                   f"counted) differ: {sorted((set(lm.FS1_ITEMS) | {'MCQ160B'}) ^ set(v.conditions))}")
    if tuple(HEALTH_ITEMS) != tuple(lm.FS2_ITEMS) + tuple(lm.FS3_ITEMS):
        out.append(f"HEALTH_ITEMS {HEALTH_ITEMS} are not fs2 + fs3 {lm.FS2_ITEMS + lm.FS3_ITEMS}")

    # every answer the reader can give is one LinAge2 accepts
    for item, given in (("HUQ010", set(_HEALTH.values())), ("HUQ020", set(_TREND.values())),
                        ("HUQ050", {_visits_category(n) for n in range(0, 400)})):
        if not given <= lm._ANSWER_CODES[item]:
            out.append(f"the reader can answer {item} with {sorted(given - lm._ANSWER_CODES[item])}, "
                       f"which LinAge2 does not accept")
    for item in v.conditions:
        if not {1, 2} <= lm._ANSWER_CODES.get(item, frozenset()):
            out.append(f"LinAge2 does not accept a yes/no answer for {item}")

    # cotinine: four levels, the reader's bins on their boundaries, the KB and the model file
    levels = {k for k in model["cotinine_levels"] if k != "note"}
    if levels != {"0", "1", "2", "3"}:
        out.append(f"the model file's cotinine levels are {sorted(levels)}, not 0-3")
    bounds = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\s*ng/mL", model["cotinine_levels"]["0"]
                                           + " " + model["cotinine_levels"]["3"])]
    if bounds != [10.0, 200.0] or _cotinine_level(9.99) != 0 or _cotinine_level(10) != 1 \
            or _cotinine_level(100) != 2 or _cotinine_level(199.9) != 2 or _cotinine_level(200) != 3:
        out.append("the reader's cotinine bins and the model file's levels differ")
    for where, desc in (("linage2_core.metta", v.model_inputs.get(COTININE, ("", ""))[1]),
                        ("the model file", model["descriptions"].get(COTININE, ""))):
        if set(re.findall(r"\b(\d)\s*\(", desc)) != {"0", "1", "2", "3"}:
            out.append(f"{where} does not describe cotinine as the levels 0-3: {desc!r}")

    # every unit a description names is the unit the reader converts into
    known = {normalise_unit(u) for s_ in LABS for u in list(s_.units) + [s_.unit]}
    for where, descriptions in (("linage2_core.metta", {c: d for c, (_, d) in v.model_inputs.items()}),
                                ("the model file", model["descriptions"])):
        for code, desc in descriptions.items():
            if code not in SPECS:
                continue
            named = _described_units(desc, known)
            canonical = normalise_unit(SPECS[code].unit)
            if named and named[0] != canonical:
                out.append(f"{where} gives {code} in {display_unit(named[0])}; the reader converts "
                           f"into {SPECS[code].unit}")

    # one lab entry per alias group, every alias and every unit in exactly one
    seen_alias: dict[str, str] = {}
    for key, group in v.labs.items():
        for alias in group.aliases:
            if alias in seen_alias:
                out.append(f"alias {alias!r} is in two lab groups: {seen_alias[alias]!r} and {key!r}")
            seen_alias[alias] = key
    index = {alias: specs for alias, specs in _ALIAS_INDEX}
    for alias, specs in index.items():
        key = seen_alias.get(alias)
        if key is None:
            out.append(f"the reader's alias {alias!r} is in no lab group")
            continue
        group = v.labs[key]
        if tuple(s_.code for s_ in specs) != group.codes:
            out.append(f"alias {alias!r} means {[s_.code for s_ in specs]} to the reader but "
                       f"{list(group.codes)} in the vocabulary")
        missing = {display_unit(u) for s_ in specs for u in s_.units} - set(group.units)
        if missing:
            out.append(f"lab group {key!r} lacks the units {sorted(missing)}")
    if len({g.key for g in v.labs.values()}) != len(v.labs) or COTININE in v.labs:
        out.append("lab group keys must be unique, and cotinine is its own kind, not a lab")
    return out
