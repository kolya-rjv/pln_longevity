"""Hallmark <-> intervention lookups over the curated evidence records.

"Which hallmarks does rapamycin target?" and "which interventions target
mitochondrial dysfunction?" both came back empty in the 2026-09-18 evaluation,
for two different reasons. The translator reached for `TargetsHallmark`, which
`logical_predicates.metta` declares and nothing populates. And there was no
endpoint to ask without an LLM at all.

The relation DOES exist, as review-level evidence records:

    (: LopezOtin2023_Fisetin_Mouse HallmarkInterventionEvidence)
    (EvidenceHallmark     LopezOtin2023_Fisetin_Mouse CellularSenescence)
    (EvidenceIntervention LopezOtin2023_Fisetin_Mouse Fisetin)
    (EvidenceSpeciesModel LopezOtin2023_Fisetin_Mouse Mouse)
    (EvidenceOutcomeText  LopezOtin2023_Fisetin_Mouse "increased health- and lifespan")
    (EvidenceReferenceNumber LopezOtin2023_Fisetin_Mouse 42)
    (SupportedByPublication  LopezOtin2023_Fisetin_Mouse LopezOtinEtAl2023_Hallmarks)

This module indexes those records (and the `HallmarkComponent` anchors) so the
question can be answered directly, with provenance, and so an intervention with
NO hallmark evidence is reported as exactly that rather than as silence.

`TargetsHallmark` is no longer empty
------------------------------------
`hallmark_targeting.metta` back-fills the declared predicate: one line per
review record, plus the two headline drugs the KB was missing entirely
(rapamycin, metformin). So the index reads BOTH shapes now, and keeps them
apart in the output:

* an **evidence record** carries the species model, the reported outcome text
  and the review reference number — the richer provenance, preserved;
* a **targeting link** carries only "this intervention is studied as acting on
  this hallmark", with the publication it was sourced to.

A targeting link is reported as a targeting link. Dressing one up as an
evidence record would invent a species model and an outcome sentence that no
source states — the same fabrication the KB's honesty contract forbids in
MeTTa, committed one layer up in Python instead.

A text scan, not a MeTTa query: the records are a few dozen atoms, the answer
needs no inference, and keeping it out of hyperon keeps it off the worker pool.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from ontology.inventory import iter_top_level, split_args

_RE_STRING = re.compile(r'^"(.*)"$', re.DOTALL)


@dataclass
class HallmarkEvidence:
    """One review-level intervention -> hallmark record."""
    record_id: str
    intervention: Optional[str] = None
    hallmark: Optional[str] = None
    species_model: Optional[str] = None
    outcome_text: Optional[str] = None
    reference_number: Optional[int] = None
    publication: Optional[str] = None
    source_file: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            "record_id": self.record_id,
            "intervention": self.intervention,
            "hallmark": self.hallmark,
            "species_model": self.species_model,
            "outcome_text": self.outcome_text,
            "reference_number": self.reference_number,
            "publication": self.publication,
            "source_file": self.source_file,
        }


@dataclass
class HallmarkTargeting:
    """One `(TargetsHallmark <intervention> <hallmark>)` fact.

    Deliberately thinner than `HallmarkEvidence`: that is the whole point. The
    fact asserts a target and nothing else, so the record has nowhere to put a
    species model or an outcome sentence, and cannot accidentally grow one.
    """
    intervention: str
    hallmark: str
    source_file: Optional[str] = None
    #: every publication the KB attaches to this intervention, if any.
    publications: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "intervention": self.intervention,
            "hallmark": self.hallmark,
            "source_file": self.source_file,
            "publications": list(self.publications),
            # Named, not implied: a caller must be able to tell this apart from
            # a review record without knowing which fields a record carries.
            "provenance": "targeting_fact",
        }


@dataclass
class HallmarkIndex:
    """Everything the hallmark questions need, already joined."""
    evidence: dict[str, HallmarkEvidence] = field(default_factory=dict)
    #: hallmark -> its anchor components (HallmarkComponent <thing> <hallmark>)
    components: dict[str, list[str]] = field(default_factory=dict)
    #: every hallmark symbol declared in hallmarks_core
    hallmarks: set[str] = field(default_factory=set)
    #: (TargetsHallmark <intervention> <hallmark>) facts, in file order
    targeting: list[HallmarkTargeting] = field(default_factory=list)
    #: subject -> publications, from (SupportedByPublication <subject> <pub>)
    publications: dict[str, list[str]] = field(default_factory=dict)

    def records(self) -> list[HallmarkEvidence]:
        return [
            e for e in self.evidence.values()
            if e.intervention and e.hallmark
        ]

    def targeting_for_hallmark(self, hallmark: str) -> list[HallmarkTargeting]:
        """Targeting facts for a hallmark, minus the ones a record already covers.

        A back-filled line and the record it was derived from are the same
        claim; returning both would double-count the review table.
        """
        key = hallmark.lower()
        covered = {
            (e.intervention or "").lower()
            for e in self.for_hallmark(hallmark)
        }
        return [
            t for t in self.targeting
            if t.hallmark.lower() == key and t.intervention.lower() not in covered
        ]

    def targeting_for_intervention(self, intervention: str) -> list[HallmarkTargeting]:
        key = intervention.lower()
        covered = {
            (e.hallmark or "").lower()
            for e in self.for_intervention(intervention)
        }
        return [
            t for t in self.targeting
            if t.intervention.lower() == key and t.hallmark.lower() not in covered
        ]

    def for_hallmark(self, hallmark: str) -> list[HallmarkEvidence]:
        key = hallmark.lower()
        return [e for e in self.records() if (e.hallmark or "").lower() == key]

    def for_intervention(self, intervention: str) -> list[HallmarkEvidence]:
        key = intervention.lower()
        return [e for e in self.records() if (e.intervention or "").lower() == key]

    def interventions(self) -> list[str]:
        """Every intervention with a hallmark link of EITHER kind."""
        names = {e.intervention for e in self.records() if e.intervention}
        names |= {t.intervention for t in self.targeting}
        return sorted(names)

    def covered_hallmarks(self) -> list[str]:
        names = {e.hallmark for e in self.records() if e.hallmark}
        names |= {t.hallmark for t in self.targeting}
        return sorted(names)

    def interventions_for(self, hallmark: str) -> list[str]:
        """Intervention names for a hallmark, from both shapes, deduplicated."""
        names = {e.intervention for e in self.for_hallmark(hallmark) if e.intervention}
        names |= {t.intervention for t in self.targeting_for_hallmark(hallmark)}
        return sorted(names)


def _unquote(value: str) -> str:
    m = _RE_STRING.match(value.strip())
    return m.group(1) if m else value.strip()


def build_hallmark_index(paths: Iterable[Path]) -> HallmarkIndex:
    index = HallmarkIndex()
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

            if head == ":" and len(args) >= 2 and args[1] == "HallmarkInterventionEvidence":
                rec = index.evidence.setdefault(args[0], HallmarkEvidence(args[0]))
                rec.source_file = path.name
                continue
            if head == ":" and len(args) >= 2 and args[1] == "HallmarkOfAging":
                index.hallmarks.add(args[0])
                continue
            if head == "HallmarkComponent" and len(args) == 2:
                index.components.setdefault(args[1], []).append(args[0])
                continue
            if head == "TargetsHallmark" and len(args) == 2:
                index.targeting.append(
                    HallmarkTargeting(args[0], args[1], source_file=path.name)
                )
                index.hallmarks.add(args[1])
                continue
            if head.startswith("Evidence") or head == "SupportedByPublication":
                if len(args) < 2:
                    continue
                rec = index.evidence.setdefault(args[0], HallmarkEvidence(args[0]))
                rec.source_file = rec.source_file or path.name
                value = " ".join(args[1:])
                if head == "EvidenceIntervention":
                    rec.intervention = value
                elif head == "EvidenceHallmark":
                    rec.hallmark = value
                    index.hallmarks.add(value)
                elif head == "EvidenceSpeciesModel":
                    rec.species_model = value
                elif head == "EvidenceOutcomeText":
                    rec.outcome_text = _unquote(value)
                elif head == "EvidenceReferenceNumber":
                    try:
                        rec.reference_number = int(float(value))
                    except ValueError:
                        pass
                elif head == "SupportedByPublication":
                    rec.publication = value
                    index.publications.setdefault(args[0], []).append(value)

    # A targeting fact and its sources are separate atoms, and either can be
    # read first, so the join happens once both files have been scanned.
    for link in index.targeting:
        link.publications = list(index.publications.get(link.intervention, []))
    return index


_CACHE: dict[tuple, HallmarkIndex] = {}


def hallmark_index(paths: Iterable[Path]) -> HallmarkIndex:
    path_list = list(paths)
    key = tuple(
        (str(p), p.stat().st_mtime, p.stat().st_size) if p.exists() else (str(p), 0.0, 0)
        for p in path_list
    )
    cached = _CACHE.get(key)
    if cached is None:
        cached = build_hallmark_index(path_list)
        _CACHE[key] = cached
    return cached
