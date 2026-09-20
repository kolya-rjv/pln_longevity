"""Human-evidence lookups over `human_evidence.metta`.

"What does the evidence say about metformin in HUMANS?" was answered, in the
2026-09-18 evaluation, with a paragraph of English: the loaded DrugAge stack is
model-organism evidence, so the question cannot be answered. True — and composed
by a language model, because there was nothing in the knowledge base to point
at. `human_evidence.metta` is the record set that fixes that; this module
indexes it so the question can be asked without an LLM in the loop at all.

The record shape it reads — one atom per study, because hyperon 0.2.10 aborts
the process once the shared runtime space passes roughly 1050 top-level
expressions and a one-predicate-per-field record set costs more than twice as
much (the measurement is in `human_evidence.metta` §1)::

    (HumanEvidence Justice2019_DQ_PulmonaryFunction DasatinibPlusQuercetin PulmonaryFunction
       (design OpenLabelPilot) (n 14) (result ReportedNull)
       (tier SingleHumanTrial) (pmid "30616998")
       (measured "pulmonary function tests …")
       (found "unchanged — …")
       (caveat "A null in an uncontrolled n=14 pilot …"))
    (SupportedByPublication Justice2019_DQ_PulmonaryFunction JusticeEtAl2019_SenolyticsIPF)

A field the study does not have is the symbol ``NotStated`` in the record, and
this module turns it into ``None`` rather than into a zero or an empty string.

Three absences, kept apart
--------------------------
The index exists to preserve distinctions that prose destroys:

* ``result="ReportedNull"`` — somebody measured it in people and found nothing.
* ``tier=None`` with ``result="NotYetReported"`` — the trial is planned and has
  not reported, so there is no tier to give. TAME is the example, and a tier
  invented for it would be the most consequential fabrication available in this
  domain.
* no record at all — reported as an explicit ``note``, never as an empty list
  with no explanation.

A text scan, not a MeTTa query, exactly as `ontology/hallmarks.py` is: a few
dozen atoms, no inference needed, and keeping it out of hyperon keeps it off the
worker pool (which holds the GIL for the whole of `MeTTa.run()`).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from ontology.inventory import iter_top_level, split_args

_RE_STRING = re.compile(r'^"(.*)"$', re.DOTALL)

#: Which keyword sub-expression of a `(HumanEvidence …)` atom fills which
#: dataclass attribute. `n` is parsed separately because it is a number.
_KEYWORDS = {
    "design": "design",
    "result": "result",
    "tier": "tier",
    "pmid": "pmid",
    "measured": "measured",
    "found": "finding",
    "caveat": "caveat",
}

#: The symbol a record uses for a field the study does not have.
NOT_STATED = "NotStated"


@dataclass
class Publication:
    """A cited paper. `pmid` and `doi` are what make a record checkable."""
    symbol: str
    title: Optional[str] = None
    year: Optional[int] = None
    journal: Optional[str] = None
    doi: Optional[str] = None
    pmid: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "title": self.title,
            "year": self.year,
            "journal": self.journal,
            "doi": self.doi,
            "pmid": self.pmid,
        }


@dataclass
class HumanStudy:
    """One curated human study of one intervention on one outcome."""
    record_id: str
    intervention: Optional[str] = None
    outcome: Optional[str] = None
    design: Optional[str] = None
    n: Optional[int] = None
    measured: Optional[str] = None
    finding: Optional[str] = None
    result: Optional[str] = None
    #: None for a planned trial that has not reported — never defaulted to a
    #: tier, because "no result yet" is not a weak result.
    tier: Optional[str] = None
    caveat: Optional[str] = None
    #: The PMID written in the record itself. It must agree with
    #: `publication.pmid`, and a test asserts that it does.
    pmid: Optional[str] = None
    publication: Optional[Publication] = None
    source_file: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            "record_id": self.record_id,
            "intervention": self.intervention,
            "outcome": self.outcome,
            "design": self.design,
            "n": self.n,
            "measured": self.measured,
            "finding": self.finding,
            "result": self.result,
            "evidence_tier": self.tier,
            "caveat": self.caveat,
            "pmid": self.pmid,
            "publication": self.publication.as_dict() if self.publication else None,
            "source_file": self.source_file,
        }


@dataclass
class CrossReference:
    """Human evidence this KB holds somewhere else, pointed at rather than copied."""
    intervention: str
    tier: str
    where: str
    source_file: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            "intervention": self.intervention,
            "evidence_tier": self.tier,
            "where": self.where,
            "source_file": self.source_file,
            "provenance": "cross_reference",
        }


@dataclass
class HumanEvidenceIndex:
    studies: dict[str, HumanStudy] = field(default_factory=dict)
    cross_references: list[CrossReference] = field(default_factory=list)
    publications: dict[str, Publication] = field(default_factory=dict)

    def records(self) -> list[HumanStudy]:
        return [s for s in self.studies.values() if s.intervention]

    def for_intervention(self, intervention: str) -> list[HumanStudy]:
        key = intervention.lower()
        return [s for s in self.records() if (s.intervention or "").lower() == key]

    def cross_references_for(self, intervention: str) -> list[CrossReference]:
        key = intervention.lower()
        return [x for x in self.cross_references if x.intervention.lower() == key]

    def interventions(self) -> list[str]:
        names = {s.intervention for s in self.records() if s.intervention}
        names |= {x.intervention for x in self.cross_references}
        return sorted(names)

    def covered_outcomes(self) -> list[str]:
        return sorted({s.outcome for s in self.records() if s.outcome})


def _unquote(value: str) -> str:
    m = _RE_STRING.match(value.strip())
    return m.group(1) if m else value.strip()


def build_human_evidence_index(paths: Iterable[Path]) -> HumanEvidenceIndex:
    index = HumanEvidenceIndex()
    #: record id -> publication symbol, joined once both files are scanned.
    cited: dict[str, str] = {}

    for path in paths:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for expr in iter_top_level(text):
            if not (expr.startswith("(") and expr.endswith(")")):
                continue
            parts = split_args(expr[1:-1].strip())
            if not parts:
                continue
            head, args = parts[0], parts[1:]

            if head == ":" and len(args) >= 2 and args[1] == "Publication":
                index.publications.setdefault(args[0], Publication(args[0]))
                continue

            if head == "HumanEvidence" and len(args) >= 3:
                rec = HumanStudy(args[0], intervention=args[1], outcome=args[2],
                                 source_file=path.name)
                for chunk in args[3:]:
                    if not (chunk.startswith("(") and chunk.endswith(")")):
                        continue
                    kv = split_args(chunk[1:-1].strip())
                    if len(kv) < 2:
                        continue
                    key, value = kv[0], _unquote(" ".join(kv[1:]))
                    if value == NOT_STATED:
                        continue          # absent stays absent — never a zero
                    if key == "n":
                        try:
                            rec.n = int(float(value))
                        except ValueError:
                            pass
                    elif key in _KEYWORDS:
                        setattr(rec, _KEYWORDS[key], value)
                index.studies[rec.record_id] = rec
                continue
            if head == "HumanEvidenceElsewhere" and len(args) == 3:
                index.cross_references.append(CrossReference(
                    args[0], args[1], _unquote(args[2]), source_file=path.name,
                ))
                continue

            # Publication metadata, wherever it is declared. A record may cite a
            # paper another file owns (human_evidence.metta reuses
            # hallmark_targeting.metta's Bannister record rather than
            # redeclaring it), so these are collected from every scanned file.
            if head == "SupportedByPublication" and len(args) == 2:
                cited[args[0]] = args[1]
                continue
            if head in ("PublicationTitle", "PublicationYear", "JournalName",
                        "DOI", "PubMedID") and len(args) >= 2:
                pub = index.publications.setdefault(args[0], Publication(args[0]))
                value = _unquote(" ".join(args[1:]))
                if head == "PublicationTitle":
                    pub.title = value
                elif head == "PublicationYear":
                    try:
                        pub.year = int(float(value))
                    except ValueError:
                        pass
                elif head == "JournalName":
                    pub.journal = value
                elif head == "DOI":
                    pub.doi = value
                elif head == "PubMedID":
                    pub.pmid = value

    # A record and the publication it cites are separate atoms, in separate
    # files in the Bannister case, so the join happens once everything is read.
    for record_id, pub_symbol in cited.items():
        study = index.studies.get(record_id)
        if study is not None:
            study.publication = index.publications.get(pub_symbol) or Publication(pub_symbol)
    return index


_CACHE: dict[tuple, HumanEvidenceIndex] = {}


def human_evidence_index(paths: Iterable[Path]) -> HumanEvidenceIndex:
    path_list = list(paths)
    key = tuple(
        (str(p), p.stat().st_mtime, p.stat().st_size) if p.exists() else (str(p), 0.0, 0)
        for p in path_list
    )
    cached = _CACHE.get(key)
    if cached is None:
        cached = build_human_evidence_index(path_list)
        _CACHE[key] = cached
    return cached
