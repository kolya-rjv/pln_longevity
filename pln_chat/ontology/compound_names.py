"""Shared compound-name resolution for every DrugAge-facing entry point.

Why this exists
---------------
`ontology.drugage_selector._norm` matches a requested compound against a DrugAge
row by lowercasing and dropping separators. That is enough for
`rapamycin` -> `Rapamycin`, and nothing else. A live evaluation of the HTTP API
(78 calls, 18 Sep 2026) found the practical consequence: `sirolimus`, `NMN`,
`NAD+`, `EGCG`, `N-acetylcysteine` and every spelling of `17-alpha-estradiol`
were silently omitted from `/drugage/rank`, so the SAME question succeeded or
failed on phrasing alone. All six exist in the DrugAge build under a different
string:

    sirolimus            -> Rapamycin                        (INN synonym)
    NMN                  -> Nicotinamide_mononucleotide      (abbreviation)
    NAD+                 -> Nicotinamide_adenine_dinucleotide(abbreviation + charge)
    EGCG                 -> Epigallocatechin_3_gallate       (abbreviation + locant)
    N-acetylcysteine     -> N_acetyl_L_cysteine              (stereo descriptor)
    17-alpha-estradiol   -> N_17alphaestradiol               (ETL numeric-prefix artifact)

Three DIFFERENT failure modes hide behind one symptom, so one normaliser cannot
fix them. This module separates them into an explicit, auditable ladder and —
critically — reports which rung matched, so a caller can tell an exact hit from
an accepted typo correction:

    exact        the requested string is a DrugAge symbol verbatim
    normalized   case / separator / unicode / Greek-letter differences only
    synonym      a CURATED identity from the table below (sirolimus == rapamycin)
    etl_artifact an artefact of the ETL's symbol sanitiser, undone mechanically
    fuzzy        a near-miss accepted above FUZZY_ACCEPT with a warning
    ambiguous    several candidates, none preferred -> NOT resolved, suggestions
    unmatched    nothing close enough -> NOT resolved, suggestions

Nothing here ever invents a match: `ambiguous` and `unmatched` resolve to
``None`` and carry suggestions instead, because ranking the wrong compound is
worse than omitting one (docs/etl_inference_wiring.md §5).

The curated synonym table is a CURATED DATA asset in the sense of
mechanistic_bridges.metta: every entry is a chemical identity (same substance,
different name), not a similarity judgement, and each carries a short note
saying why. Entries that would merge two distinct substances are deliberately
absent — bare `vitamin e` is not mapped to any one tocopherol row, and a name
that legitimately matches several DrugAge symbols is returned as `ambiguous`
with the candidates rather than silently collapsed to one.

Kept free of any FastAPI / Gradio / hyperon import so it is unit-testable
standalone and usable from the selector, the HTTP layer and the LLM prompt
builder alike.
"""
from __future__ import annotations

import difflib
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Iterable, Optional

# ── Tuning knobs (the one place the matcher's constants live) ────────────────

#: Minimum difflib ratio for an UNAMBIGUOUS near-miss to be auto-accepted as a
#: typo correction. 0.86 accepts the reported `rapamicin` -> `Rapamycin`
#: (ratio 0.889); anything lower starts pulling in genuinely different
#: substances, so a near-miss below this is suggested, never applied.
FUZZY_ACCEPT = 0.86

#: An accepted near-miss must ALSO beat its runner-up by this margin. Without
#: it, two equally-close names (a real risk in a 1,043-name vocabulary full of
#: `ABC23` / `ABC26` style codes) would be decided by sort order.
FUZZY_MARGIN = 0.05

#: Ratio above which a candidate is worth SUGGESTING even when it is not
#: accepted. Deliberately loose — a suggestion costs nothing and an omitted
#: compound with no hint is exactly the reported failure.
FUZZY_SUGGEST = 0.62

#: How many suggestions to return for an unresolved name.
MAX_SUGGESTIONS = 3


# ── Unicode / Greek normalisation ────────────────────────────────────────────
# A question typed by a human carries the characters a paper uses. DrugAge
# symbols are ASCII. Transliterate rather than strip, so `17-α-estradiol` and
# `17-alpha-estradiol` reach the same key.
_GREEK = {
    "α": "alpha", "Α": "alpha",
    "β": "beta",  "Β": "beta",
    "γ": "gamma", "Γ": "gamma",
    "δ": "delta", "Δ": "delta",
    "ε": "epsilon", "Ε": "epsilon",
    "ζ": "zeta",  "η": "eta",
    "θ": "theta", "κ": "kappa",
    "λ": "lambda", "μ": "mu", "µ": "mu",
    "ω": "omega", "Ω": "omega",
}

# Dashes, primes and other punctuation a paper or a user may paste.
_PUNCT = {
    "‐": "-", "‑": "-", "‒": "-", "–": "-",
    "—": "-", "―": "-", "−": "-",
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "·": "-",
}


def transliterate(name: str) -> str:
    """ASCII-fold a user-typed compound name (Greek letters spelled out)."""
    out: list[str] = []
    for ch in name:
        if ch in _GREEK:
            out.append(_GREEK[ch])
        elif ch in _PUNCT:
            out.append(_PUNCT[ch])
        else:
            out.append(ch)
    folded = unicodedata.normalize("NFKD", "".join(out))
    return "".join(c for c in folded if not unicodedata.combining(c))


def canonical_key(name: str) -> str:
    """The primary matching key: transliterated, lowercased, alphanumerics only.

    This is a strict superset of `drugage_selector._norm` — it additionally
    folds Greek letters and unicode dashes — so every name `_norm` matched
    today still matches, by construction.
    """
    return re.sub(r"[^a-z0-9]", "", transliterate(name).lower())


# ── ETL symbol artefacts ─────────────────────────────────────────────────────
# drugage_etl.py must emit MeTTa SYMBOLS, so a DrugAge name that starts with a
# digit is prefixed with `N_` (`17alpha-Estradiol` -> `N_17alphaestradiol`) and
# internal separators collapse to `_`. Undoing that prefix is mechanical, not a
# judgement call, so it gets its own rung rather than living in the curated
# table.
_ETL_NUMERIC_PREFIX = re.compile(r"^n(?=\d)")


def _etl_variants(key: str) -> set[str]:
    """Alternate keys that differ from `key` only by an ETL symbol artefact."""
    variants = {key}
    stripped = _ETL_NUMERIC_PREFIX.sub("", key)
    if stripped != key:
        variants.add(stripped)
    else:
        variants.add("n" + key)
    return variants


# ── Stereo / salt / hydrate descriptors ──────────────────────────────────────
# `N-acetylcysteine` and `N_acetyl_L_cysteine` are the same substance; the
# difference is an optional stereo descriptor. Salt and hydrate suffixes
# (`hydrochloride`, `sodium`, `monohydrate`, ...) name the same active moiety.
# Removing them yields a LOOSE key used only as a secondary lookup, never as the
# primary one, so `Cysteine` and `Cysteine_hydrochloride` stay distinguishable
# when both are asked for by their exact names.
_STEREO_TOKENS = ("dl", "l", "d", "rs", "r", "s", "cis", "trans", "e", "z")
_SALT_TOKENS = (
    "hydrochloride", "hcl", "sodium", "potassium", "calcium", "magnesium",
    "sulfate", "sulphate", "phosphate", "acetate", "citrate", "tartrate",
    "maleate", "mesylate", "besylate", "fumarate", "succinate", "chloride",
    "bromide", "monohydrate", "dihydrate", "hydrate", "anhydrous", "salt",
)


def loose_key(name: str) -> str:
    """Secondary key: canonical key minus stereo descriptors and salt suffixes.

    A stereo descriptor is stripped in LEADING or MEDIAL position only, never in
    FINAL position. In chemical nomenclature a stereo descriptor prefixes the
    thing it qualifies — `D-glucosamine`, `trans-resveratrol`,
    `N-acetyl-L-cysteine` — whereas a trailing single letter is a SERIES
    DESIGNATOR naming a distinct substance: `Urolithin D`, `Vitamin E`.

    Treating the two alike is what made `urolithin` resolve to `Urolithin_D` and
    `vitamin` to `Vitamin_E`, both at score 1.0 and both labelled "same active
    moiety". Worse, it hid the ambiguity: `Urolithin_D` lost its `d` and landed
    under `urolithin` while `Urolithin_A` kept its `a` and landed under
    `urolithina`, so the two never shared a bucket and the collision check below
    could not see them. Salt and hydrate suffixes DO trail the moiety they
    qualify (`metformin hydrochloride`), so those still strip anywhere.
    """
    text = transliterate(name).lower()
    tokens = [t for t in re.split(r"[^a-z0-9]+", text) if t]
    last = len(tokens) - 1
    kept = [
        t for i, t in enumerate(tokens)
        if t not in _SALT_TOKENS
        and not (t in _STEREO_TOKENS and i != last)
    ]
    return "".join(kept) or canonical_key(name)


# ── Curated synonym table ────────────────────────────────────────────────────
# CURATED DATA. Each key is a name a caller plausibly types; each value is the
# name DrugAge uses. Every row is a chemical IDENTITY — the same substance under
# a different name (INN vs brand, abbreviation vs full name, common vs
# systematic) — never "these two are similar". Keys are matched through
# canonical_key(), so case, separators and Greek spelling do not matter here.
#
# Deliberately ABSENT: any name that maps to more than one DrugAge substance
# (e.g. bare `estradiol`, `vitamin e`), and any alias that is already a DrugAge
# symbol (those resolve one rung earlier, so an entry would be dead weight).
# Ambiguous names are reported as `ambiguous` with candidates, never collapsed.
_SYNONYM_SOURCE: dict[str, tuple[str, str]] = {
    # mTOR
    "sirolimus": ("Rapamycin", "INN for rapamycin (Rapamune)"),
    "rapamune": ("Rapamycin", "brand name for rapamycin"),
    # NAD+ pathway
    "nmn": ("Nicotinamide_mononucleotide", "standard abbreviation"),
    "beta-nmn": ("Nicotinamide_mononucleotide", "beta anomer, the supplemented form"),
    "nr": ("Nicotinamide_riboside", "standard abbreviation"),
    "niagen": ("Nicotinamide_riboside", "brand name for nicotinamide riboside"),
    "nad": ("Nicotinamide_adenine_dinucleotide", "standard abbreviation"),
    "nad+": ("Nicotinamide_adenine_dinucleotide", "abbreviation, oxidised form"),
    "nadh": ("Nicotinamide_adenine_dinucleotide", "reduced form of the same dinucleotide"),
    "niacinamide": ("Nicotinamide", "USAN synonym for nicotinamide"),
    "vitamin b3": ("Nicotinamide", "nicotinamide is the amide form of vitamin B3"),
    # Polyphenols
    "egcg": ("Epigallocatechin_3_gallate", "standard abbreviation"),
    "epigallocatechin gallate": ("Epigallocatechin_3_gallate", "same compound, locant omitted"),
    "epigallocatechin-3-o-gallate": ("Epigallocatechin_3_gallate", "systematic spelling"),
    # Thiols
    "nac": ("N_acetyl_L_cysteine", "standard abbreviation"),
    "n-acetylcysteine": ("N_acetyl_L_cysteine", "stereo descriptor omitted"),
    "acetylcysteine": ("N_acetyl_L_cysteine", "INN, N- and stereo descriptor omitted"),
    # Steroids
    "17-alpha-estradiol": ("N_17alphaestradiol", "ETL symbol for 17alpha-estradiol"),
    "17alpha-estradiol": ("N_17alphaestradiol", "ETL symbol for 17alpha-estradiol"),
    "17a-estradiol": ("N_17alphaestradiol", "'a' spelling of the alpha locant"),
    "alpha-estradiol": ("N_17alphaestradiol", "common short form of 17alpha-estradiol"),
    "17-beta-estradiol": ("Beta_estradiol", "17beta-estradiol, the DrugAge Beta_estradiol row"),
    "17beta-estradiol": ("Beta_estradiol", "17beta-estradiol, the DrugAge Beta_estradiol row"),
    # Krebs-cycle / supplements
    "akg": ("Alpha_ketoglutarate", "standard abbreviation"),
    "alpha-kg": ("Alpha_ketoglutarate", "standard abbreviation"),
    "2-oxoglutarate": ("Alpha_ketoglutarate", "IUPAC name for alpha-ketoglutarate"),
    "ca-akg": ("Alpha_ketoglutarate", "calcium salt of the same acid"),
    "coq10": ("Coenzyme_Q10", "standard abbreviation"),
    "ubiquinone": ("Coenzyme_Q10", "chemical name for coenzyme Q10"),
    "ala": ("Alpha_lipoic_acid", "standard abbreviation for alpha-lipoic acid"),
    "lipoic acid": ("Alpha_lipoic_acid", "common name, locant omitted"),
    "thioctic acid": ("Alpha_lipoic_acid", "INN for alpha-lipoic acid"),
    # Pharmaceuticals
    "glucophage": ("Metformin", "brand name for metformin"),
    "metformin hydrochloride": ("Metformin", "salt of the same active moiety"),
    "precose": ("Acarbose", "brand name for acarbose"),
    "glucobay": ("Acarbose", "brand name for acarbose"),
    "sprycel": ("Dasatinib", "brand name for dasatinib"),
    "asa": ("Aspirin", "standard abbreviation for acetylsalicylic acid"),
    "acetylsalicylic acid": ("Aspirin", "chemical name; DrugAge writes Aspirin"),
    "lithium": ("Lithium_Chloride", "DrugAge records lithium as the chloride salt"),
    # Senotherapeutics / mitochondria
    "ss-31": ("Elamipretide", "development code for elamipretide"),
    "mitoquinone": ("MitoQ", "chemical name for the mitochondria-targeted ubiquinone"),
    # Botanicals
    "green tea": ("Green_tea_extract", "DrugAge records the extract"),
    "curcuma": ("Curcumin", "the active principle DrugAge records"),
    "resveratrol trans": ("Resveratrol", "trans isomer, the supplemented form"),
    "trans-resveratrol": ("Resveratrol", "trans isomer, the supplemented form"),
    "spermidine trihydrochloride": ("Spermidine", "salt of the same polyamine"),
}

#: canonical_key(alias) -> (drugage_name, note)
SYNONYMS: dict[str, tuple[str, str]] = {
    canonical_key(alias): value for alias, value in _SYNONYM_SOURCE.items()
}


# ── Resolution result ────────────────────────────────────────────────────────

@dataclass
class Resolution:
    """The outcome of resolving ONE requested compound name.

    `matched` is None exactly when the name was not resolved; `suggestions`
    then carries the nearest DrugAge names so a caller (or a user) can retry.
    """
    query: str
    matched: Optional[str] = None
    method: str = "unmatched"      # see the module docstring for the ladder
    score: float = 0.0             # similarity for `fuzzy`, else 1.0 / 0.0
    note: Optional[str] = None     # why a synonym/artefact rung fired
    suggestions: list[str] = field(default_factory=list)

    @property
    def resolved(self) -> bool:
        return self.matched is not None

    @property
    def warning(self) -> Optional[str]:
        """A caller-facing note when the match was not literal."""
        if self.method == "synonym":
            return f"'{self.query}' resolved to DrugAge compound '{self.matched}' ({self.note})."
        if self.method == "etl_artifact":
            return (
                f"'{self.query}' resolved to DrugAge symbol '{self.matched}' "
                f"({self.note})."
            )
        if self.method == "fuzzy":
            return (
                f"'{self.query}' did not match any DrugAge compound exactly; "
                f"used the closest name '{self.matched}' (similarity "
                f"{self.score:.2f}). Pass the exact name to avoid the guess."
            )
        if self.method == "ambiguous":
            options = ", ".join(self.suggestions)
            return (
                f"'{self.query}' is ambiguous in DrugAge — it could mean any of: "
                f"{options}. Ask for one of those names."
            )
        if self.method == "unmatched":
            if self.suggestions:
                return (
                    f"'{self.query}' matched no DrugAge compound. Closest names: "
                    f"{', '.join(self.suggestions)}."
                )
            return f"'{self.query}' matched no DrugAge compound."
        return None

    def as_dict(self) -> dict:
        return {
            "query": self.query,
            "matched": self.matched,
            "method": self.method,
            "score": round(self.score, 4),
            "note": self.note,
            "suggestions": list(self.suggestions),
        }


# ── The resolver ─────────────────────────────────────────────────────────────

class CompoundResolver:
    """Resolve caller-typed compound names against a fixed DrugAge vocabulary.

    Build it once per vocabulary (the distinct `UsesIntervention` names in the
    loaded DrugAge rows) and reuse it: the index is built eagerly and the
    lookups are dict hits plus, only for a miss, one `difflib` pass.
    """

    def __init__(self, vocabulary: Iterable[str]) -> None:
        self.names: list[str] = sorted({n for n in vocabulary if n})
        # Primary index: canonical key -> the DrugAge names sharing it. A list,
        # not a single name, so a genuine collision is reported as ambiguous
        # rather than resolved by dict-insertion order.
        self._by_key: dict[str, list[str]] = {}
        # Secondary index on the loose (stereo/salt-stripped) key.
        self._by_loose: dict[str, list[str]] = {}
        for name in self.names:
            self._by_key.setdefault(canonical_key(name), []).append(name)
            self._by_loose.setdefault(loose_key(name), []).append(name)

    # -- internals ------------------------------------------------------------

    def _suggest(self, key: str, limit: int = MAX_SUGGESTIONS) -> list[str]:
        """Nearest DrugAge names for an unresolved key (never auto-applied)."""
        close = difflib.get_close_matches(
            key, list(self._by_key), n=limit, cutoff=FUZZY_SUGGEST
        )
        out: list[str] = []
        for k in close:
            out.extend(self._by_key[k])
        # A substring hit is often more useful than an edit-distance one
        # (`glucosamine` -> `D_glucosamine`), so top the list up with those.
        if len(out) < limit and len(key) >= 4:
            for k, names in self._by_key.items():
                if key in k or k in key:
                    for n in names:
                        if n not in out:
                            out.append(n)
                if len(out) >= limit:
                    break
        return out[:limit]

    #: Longest series designator treated as one: `a`, `d`, `k2`, `q10`, `b12`.
    FAMILY_SUFFIX_MAX = 3

    def _family_candidates(self, key: str) -> list[str]:
        """Vocabulary names that are `key` plus a short series designator."""
        out: list[str] = []
        if len(key) < 4:
            return out
        for candidate_key, names in self._by_key.items():
            suffix = candidate_key[len(key):]
            if not candidate_key.startswith(key) or not suffix:
                continue
            if len(suffix) <= self.FAMILY_SUFFIX_MAX and suffix.isalnum():
                out.extend(names)
        return out

    def _best_fuzzy(self, key: str) -> tuple[Optional[str], float]:
        """Accept a near-miss only when it is both close AND clearly ahead."""
        matches = difflib.get_close_matches(
            key, list(self._by_key), n=3, cutoff=FUZZY_SUGGEST
        )
        if not matches:
            return None, 0.0
        scored = [
            (difflib.SequenceMatcher(None, key, m).ratio(), m) for m in matches
        ]
        scored.sort(reverse=True)
        top_score, top_key = scored[0]
        if top_score < FUZZY_ACCEPT:
            return None, 0.0
        if len(scored) > 1 and (top_score - scored[1][0]) < FUZZY_MARGIN:
            return None, 0.0          # two equally-close names: do not guess
        names = self._by_key[top_key]
        if len(names) != 1:
            return None, 0.0
        return names[0], top_score

    # -- public ---------------------------------------------------------------

    def resolve(self, query: str) -> Resolution:
        """Resolve ONE name through the ladder documented at module level."""
        raw = (query or "").strip()
        if not raw:
            return Resolution(query=query, method="unmatched")

        # 1. exact symbol
        if raw in self._by_key.get(canonical_key(raw), []) and raw in self.names:
            return Resolution(query=raw, matched=raw, method="exact", score=1.0)

        key = canonical_key(raw)

        # 2. case / separator / unicode differences only
        hits = self._by_key.get(key, [])
        if len(hits) == 1:
            return Resolution(query=raw, matched=hits[0], method="normalized", score=1.0)
        if len(hits) > 1:
            return Resolution(
                query=raw, method="ambiguous", suggestions=sorted(hits)[:MAX_SUGGESTIONS]
            )

        # 3. curated synonym
        syn = SYNONYMS.get(key)
        if syn is not None:
            target, note = syn
            target_hits = self._by_key.get(canonical_key(target), [])
            if len(target_hits) == 1:
                return Resolution(
                    query=raw, matched=target_hits[0], method="synonym",
                    score=1.0, note=note,
                )
            # The synonym table names a compound this vocabulary does not hold
            # (e.g. ranking against the 201-row sample). Say so rather than
            # pretending the alias itself was unknown.
            return Resolution(
                query=raw, method="unmatched",
                note=f"'{raw}' is a known synonym for '{target}', which is not in "
                     f"the loaded DrugAge rows.",
                suggestions=self._suggest(canonical_key(target)),
            )

        # 4. ETL symbol artefact (leading N_ before a digit)
        for variant in _etl_variants(key) - {key}:
            hits = self._by_key.get(variant, [])
            if len(hits) == 1:
                return Resolution(
                    query=raw, matched=hits[0], method="etl_artifact", score=1.0,
                    note="the ETL prefixes a symbol starting with a digit with 'N_'",
                )

        # 5. stereo descriptor / salt form
        loose = loose_key(raw)
        hits = self._by_loose.get(loose, [])
        if len(hits) == 1:
            return Resolution(
                query=raw, matched=hits[0], method="synonym", score=1.0,
                note="stereo descriptor or salt form differs; same active moiety",
            )
        if len(hits) > 1:
            return Resolution(
                query=raw, method="ambiguous", suggestions=sorted(hits)[:MAX_SUGGESTIONS]
            )

        # 6. family stem — `urolithin` when the build holds Urolithin_A and
        #    Urolithin_D. The query is a complete name PLUS a series designator
        #    short enough to be one (`a`, `d`, `q10`, `b12`), for two or more
        #    substances. That is a question, not a compound, so decline it with
        #    the candidates rather than letting the fuzzy rung pick a winner.
        family = self._family_candidates(key)
        if len(family) > 1:
            return Resolution(
                query=raw, method="ambiguous",
                suggestions=sorted(family)[:MAX_SUGGESTIONS],
                note=f"'{raw}' names a family of compounds in DrugAge, not one "
                     f"compound. Ask for one of the suggestions by name.",
            )

        # 7. accepted typo correction
        best, score = self._best_fuzzy(key)
        if best is not None:
            return Resolution(query=raw, matched=best, method="fuzzy", score=score)

        # 8. give up, with directions
        return Resolution(query=raw, method="unmatched", suggestions=self._suggest(key))

    def resolve_all(self, queries: Iterable[str]) -> list[Resolution]:
        return [self.resolve(q) for q in queries]


def alias_hint_lines(limit: int = 24) -> list[str]:
    """Compact `alias -> DrugAge name` lines for injection into the LLM prompt.

    The translator is the other half of the reported failure: it mapped
    "urolithin A" to `Urolithin_A` but passed "NMN" through unresolved. Showing
    it the same table the endpoint uses keeps the two halves in lockstep instead
    of each guessing separately.
    """
    seen: set[str] = set()
    lines: list[str] = []
    for alias, (target, _note) in _SYNONYM_SOURCE.items():
        if target in seen:
            continue
        seen.add(target)
        lines.append(f"{alias} -> {target}")
        if len(lines) >= limit:
            break
    return lines
