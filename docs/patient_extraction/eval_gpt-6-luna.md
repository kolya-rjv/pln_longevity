# Model reader — live evaluation (gpt-6-luna)

Run 2026-10-05: 1463 texts (398 corpus entries with expected outcomes, 1065 round-4 reproductions), wall 428 s.

## Verdict

**PASS**: no expected-refused text became usable without a click, and every text was read by the model (no rules-only fall-back).

## Corpus (expected outcomes)

| | rules alone | rules + model |
|---|---|---|
| expected refused, still refused | 180/180 (100%) | 180/180 (100%) |
| expected usable, read as expected | 218/218 (100%) | 218/218 (100%) |
| expected usable, blocked by the model | — | 0/218 (0%) |
| expected usable, usable but different | — | 0 |

## What the model did

- Texts with a statement the rules did not (fully) understand: 226; rewritten: 46/226 (20%) (46 rewrites in all).
- Texts with a suggestion: 656; with a blocking one: 1.
- Round-4 texts the rules read as usable that the model now blocks: 9/326 (3%); usable with a different smoking, age or sex: 0.
- Items discarded by the checks: 142 of 4612 (smoking 65, condition 23, no_other_conditions 19, healthcare_visits 8, self_rated_health 7, age 7, lab 5, grimage 3).

Why items were discarded (top 15):

- 34 × the quote is about someone else
- 24 × the quote has no smoking word
- 12 × the quote names exceptions
- 11 × in what the model says is about someone else
- 7 × the quote is not in the text
- 7 × the quote does not say there are no
- 6 × the quote is not about health
- 6 × no single age phrase in the quote
- 4 × the wording reads no, the model says yes
- 4 × not one count
- 3 × a range of visits
- 3 × the wording reads yes, the model says no
- 3 × the number continues
- 2 × smokeless tobacco says nothing about smoking
- 2 × the quote occurs more than once

## Latency and tokens

- Model call: p50 2.5 s, p95 4.0 s, max 6.7 s (1290 calls).
- Tokens: prompt 7,562,908 (cached 7,505,190), completion 210,657 (reasoning 102,372); per call 5,862 in, 163 out.

## Round 4: texts the model now blocks (sample)

- `smoker, quit during covid` — 'smoker, quit during covid': the rules read current smoker, the model reads former smoker; choose a wording below, or rewrite it
- `patches` — 'patches': nicotine replacement raises cotinine, which LinAge2 reads, but is not smoking to the knowledge base; give 'cotinine N ng/mL' from a test, or remove the nicotine replacement to be read without it
- `\bsmok` — '\bsmok': the model cannot tell what this says about your smoking (The text is incomplete and does not establish smoking status.); write 'current smoker', 'former smoker' or 'never smoked'
- `former smoker, quit 2010 by vaping` — 'former smoker, quit 2010': vaping raises cotinine, which LinAge2 reads, but is not smoking to the knowledge base; give 'cotinine N ng/mL' from a test, or remove the vaping to be read without it
- `smoker, 1990-2015` — 'smoker, 1990-2015': the model cannot tell whether you smoke now, used to, or never did; write 'current smoker', 'former smoker' or 'never smoked'
- `tobacco: yes` — 'tobacco: yes': the model cannot tell what this says about your smoking (Tobacco use is stated, but it does not specify whether the person smokes.); write 'current smoker', 'former smoker' or 'never smoked'
- `t dm` — 't dm': the model cannot tell whether this is one of the conditions LinAge2 counts (The abbreviation is ambiguous.); write it as 'diagnoses: …' or 'no …' with the condition's name
- `former smoker, quit 2015 by switching to vaping` — 'former smoker, quit 2015 by switching to vaping': vaping raises cotinine, which LinAge2 reads, but is not smoking to the knowledge base; give 'cotinine N ng/mL' from a test, or remove the vaping to be read without it
- `non\tsmoker` — 'non\tsmoker': the model cannot tell what this says about your smoking (The text contains an unclear separator in “non\tsmoker.”); write 'current smoker', 'former smoker' or 'never smoked'

