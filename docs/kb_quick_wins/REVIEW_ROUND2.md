# Review round 2 — the strict medication reader and the review-fix commit

Round 1 (`REVIEW_ROUND1.md`) led to a redesign of the medication reader (commit `5b10b91`). Round 2 reviewed the redesign and that commit: eight lenses (wrong reads, structure and lines, regression against the pre-medication reader, hygiene and performance, downstream and the non-reader fixes, mutation testing of the tests, honesty of every sentence, the model-assisted Read), each finding re-run by an independent skeptic. About 2.3 million reader inputs were tried (the regression lens alone ran 1.29 million differentially against the pre-medication reader).

**Result: the seven reader and commit lenses reported 67 findings; 66 were confirmed by a skeptic when this was written (1 high, 26 medium, 39 low) in the families below**, all closed in the commit that follows this document except where a row says otherwise. Every confirmed input is pinned in `tests/fixtures/medication_never_current.json` (753 inputs from both rounds) and in `tests/test_patient_medications.py`.

| family | disposition |
|---|---|
| symbols / non-Latin / zero-width characters read as nothing | fixed: a statement or line with a character the grammar cannot account for is not read (`_foreign`); only ASCII letters, digits, spaces and `.,;:!?()%/+&="-–—°µ` |
| heading / list / retraction / date context sampled only one line away | fixed: `context_for` scans the nearest heading above through list items (6 lines), checks the next two lines for a retraction (a first-two-words rule; lab lines never retract), a date line above, markdown and rule lines skipped, unreadable headings count as 'other'; every branch honours the heading, not only the dosed entry |
| relatives: possessives, plurals, missing words, line above | fixed: possessive and plural forms, a larger list, the reader's own someone-else rule consulted, a relative on the lines above makes a subject-less statement theirs, and a stop under a relative heading is not read either |
| head/rest handed back from the normalised text (hyphens lost) | fixed: the grammar tokenises on the text as typed (hyphens are tokens), so what goes back to the rest of the reader is as typed |
| doses read as years, 'a day' / 'each day', 'as prescribed', fullwidth drug names | fixed: a dose is not a year, 'per/each day', 'as prescribed' exempt from the `prescribed` block, NFKC before the mention check |
| hedges, corrections, future, replaced / inactive / historical, d/c | fixed: added to the blocking words (`d/c`, `dc'd` by pattern) |
| uncapped NFKC / normalisation, `float()` on 'inf' | fixed: nothing longer than the caps is normalised; numbers are digits only |
| honesty of text: tab wording, bare LinAgeAccel, rule 16, CHD-number forms, linage2-block status at z 1.0, stale docs | fixed: wording made true; the LinAge2 hint needs a real LinAgeDelta; absolute-risk / risk-ci / risk-confidence get the notes; the block clock's status uses the atom's value; docs corrected |
| model route: a medication line next to a rewritten line flips; blocking message promises a button | partly fixed: a line the rules cannot read is a retraction only if it STARTS like one (the cross-line dependence that flipped readings after a model rewrite was the 'was' in 'my albumin was 4.1'); the message no longer promises a button that does not exist. Left: a clicked 'wording' button can still delete a retraction (low; the build note shows what was read) |
| test-quality: untested behaviours | see the commit that follows the review's mutation testing |

## Confirmed findings (verifier-confirmed, by severity)

| severity | finding | lens |
|---|---|---|
| high | A mark or non-ASCII word between 'and' / the head and the condition or lab is dropped, so a negated or qualified condition is read as a fabricated Yes (before: refused) and a struck-out or 'target' la | r2:regression |
| medium | Uncapped NFKC on whole lines/statements: |  |
| medium | _tokens silently drops every character i |  |
| medium | A list under a non-medication heading is read as current from its second item on | r2:wrong-reads-tense |
| medium | Possessive relatives are not recognised, so a statement about someone else is read as the person's | r2:wrong-reads-tense |
| medium | A year on the line above or below does not make the medication not-now (only a year on the same line does) | r2:wrong-reads-tense |
| medium | A retraction on the next line is only seen if it is in the first three words and in _STOP_START | r2:wrong-reads-tense |
| medium | A list under a non-medication heading is read as current from its second item on | r2:wrong-reads-tense |
| medium | The heading guard is not applied to the |  |
| medium | Possessive relatives are not recognised, so a statement about someone else is read as the person's | r2:wrong-reads-tense |
| medium | A retraction on the next line is only seen if it is in the first three words and in _STOP_START | r2:wrong-reads-tense |
| medium | Retraction/status words after the drug statement that are not in _BLOCK leave the drug current | r2:wrong-reads-tense |
| medium | Heading lines without a colon are 'other' only if they are 4 words or fewer, have no digit and contain a _BLOCK word | r2:wrong-reads-tense |
| medium | The tokenizer silently drops symbols and non-ASCII words, so 'unchecked', 'struck out' and 'not' marks are ignored | r2:wrong-reads-tense |
| medium | Another person is not recognised when they are not in the closed _OTHERS list, or when the line above is not directly about them | r2:wrong-reads-tense |
| medium | Heading context is only the single line |  |
| medium | Non-medication headings are recognised too narrowly: no-colon 'Past medications', >4-word headings, **bold:** headings, and a dated heading 'Medications (2019):' (counted as a CURRENT heading) all let | r2:structure-lines |
| medium | 'Someone else on the line above' is a short closed list, ignores possessives and pronoun lines, and looks one line up only: another person's medications are read as the patient's | r2:structure-lines |
| medium | line_blocked vocabulary lacks ordinary not-now words, so a same-line qualifier (split off by the statement splitter) leaves the drug current: d/c'd, replaced by, changed to, gave up, took, ran out, pr | r2:structure-lines |
| medium | A medication under a non-medication heading is still read as CURRENT for every verb form; the docstring, REVIEW row 5 and the test header say the heading is checked and the tail closed | r2:honesty-of-text |
| medium | NFKC runs, uncapped, over whole neighbouring lines and over every statement: one 20,000-char line of combining marks takes 1.6 s CPU (was 0.013 s) and stalls other threads | r2:hygiene-performance |
| medium | A non-medication heading protects only the FIRST line under it: the second item of 'Past medications:' / 'Allergies:' / a relative's list is read as a current medication | r2:hygiene-performance |
| medium | A line with no words (a '-----' rule, '***', bullet-only, or a zero-width-space/BOM line) between a heading / relative / stop line and the statement cuts the context: _heading_above, _other_above and  | r2:hygiene-performance |
| medium | Headings that are not exactly 'Word:' are not recognised as non-medication headings (no colon, Markdown-decorated, possessive, year): the list under them is read as current | r2:hygiene-performance |
| medium | Negation, stop, relative and hedge words that are not [a-z] (and markdown checkbox/strikethrough marks) are invisible to the line check, so a drug the person does not take is read as a current medicat | r2:regression |
| medium | head/rest are handed back from the normalised, hyphen-split text, so 'pre-diabetes' after the drug is no longer read and the patient is refused with a message quoting words the user never typed | r2:regression |
| medium | Original-code wrong reads found through |  |
| low | The drug-mention pre-check runs on un-no |  |
| low | The heading guard is not applied to the |  |
| low | A year on the line above or below does not make the medication not-now (only a year on the same line does) | r2:wrong-reads-tense |
| low | Hedges, corrections and future time adverbials around the drug statement are not blocked | r2:wrong-reads-tense |
| low | A 'not taking' statement about someone else is read as the person's and can block the build with a false contradiction | r2:wrong-reads-tense |
| low | Zero-width, soft-hyphen and combining characters inside block words (and non-English negations) defeat line_blocked | r2:wrong-reads-tense |
| low | Ordinary current-medication phrasings are not read (missed readings) | r2:wrong-reads-tense |
| low | A line below that takes the drug back is recognised only if one of the first 3 words is in a short list: 'Switched to...', 'Replaced by...', "d/c'd", 'Reason for stopping', 'Last dose', 'Gave up', a d | r2:structure-lines |
| low | 'Not taking' is read with no context at |  |
| low | A metformin dose that looks like a year |  |
| low | The new bare-LinAgeAccel sentence says every LinAge2 form returns nothing; the counterfactual and scenario forms return a 0.0 record | r2:honesty-of-text |
| low | Rule 16 and the LinAge2 hint imply decompose-grimage answers for a patient with no AgeAccelGrim; it returns nothing and no note says why | r2:honesty-of-text |
| low | test_every_blocking_word_blocks_the_line... iterates the set it tests, so it pins nothing: 219 of 240 words can be deleted from _BLOCK/_OTHERS with the whole file green | r2:honesty-of-text |
| low | docs/supplement_recommendations.md still describes 'a second equation' for supplement-for-patient; 5b10b91 replaced it with one let-forced equation | r2:honesty-of-text |
| low | 'as prescribed' is documented as read (docstring grammar, REVIEW row 38 'fixed') but 'prescribed' is a blocking word, so the full reader never reads it | r2:honesty-of-text |
| low | docs/linage2_integration.md says 'never |  |
| low | Numbers in the 5b10b91 commit message and in two docstrings disagree with the repo and with REVIEW_ROUND1.md | r2:honesty-of-text |
| low | 'Every form that prints the CHD number gets the notes' (commit message, REVIEW row 22) is false: absolute-risk, risk-ci and risk-confidence get none on the chat path | r2:honesty-of-text |
| low | Tab no-witness wording says the KB has curated edges for 'the DNA-methylation markers'; it has them for three of the nine DNAm markers | r2:honesty-of-text |
| low | A drug spelled with NFKC-equivalent letters (fullwidth, math-bold, circled, superscript) is invisible to mentioned_symbols, so the contradiction guard and the withdrawal loop do not fire: a stop/quest | r2:hygiene-performance |
| low | A dose of 2000 mg is taken for a year, so 'I take metformin 2000 mg daily' (the usual daily maximum) is never read | r2:hygiene-performance |
| low | LinAgeAccel from a `linage2` block still gets its status from the UNROUNDED z (the 6-digit fix of #21 covers only ordinary markers): preview and tab say Elevated while the atom and the engine say Norm | r2:downstream-and-fixes |
| low | A patient with a bare LinAgeAccel marker is still told it 'carries a LinAge2 clinical-clock result (LinAgeDelta and one LinAgeContribution per lab)'; has_linage2 stays true; every linage-* form return | r2:downstream-and-fixes |
| low | 'Every form that prints the CHD number gets the notes' (commit message, REVIEW row 22) is false: absolute-risk, risk-ci and risk-confidence get none on the chat path | r2:honesty-of-text |
| low | docs/supplement_recommendations.md still describes 'a second equation' for supplement-for-patient; 5b10b91 replaced it with one let-forced equation | r2:honesty-of-text |
| low | A medication line next to a 'my X was N' line or a 'doctor' line makes the model route withdraw EVERY rewrite (age, sex, labs), and the build is blocked | r2:model-route |
| low | A blocking 'rules wording' button deletes the unread statement that withdrew a medication; clicking it turns a withdrawn medication into a current one | r2:model-route |
| low | No test crosses the model route with a medication; two single-point guards can be removed and all 645 tests still pass | r2:model-route |
| low | Blocking disagreement message says 'choose a wording below' when a medication statement in the span leaves no wording | r2:model-route |
| low | The test that claims to pin every blocking word iterates the set it pins: 219 single-word deletions from _BLOCK/_OTHERS pass the whole suite and change readings | r2:test-quality |
| low | A blank line between a heading / relative / stop line and the medication line is untested: dropping the blank-line skip re-reads a past or someone else's drug as current | r2:test-quality |
| low | Year rule: 'since' anywhere, 19xx years, and the digit look-arounds are unpinned | r2:test-quality |
| low | Mutations that WIDEN the closed grammar |  |
| low | heading_kind rules are unpinned (short-heading length, digit guard, dash endings, bare 'Medications', empty headings) | r2:test-quality |
| low | MAX_LINE_CHARS has no test and MAX_STATEMENT_CHARS is tested only by an input scaled by the constant; the HOSTILE tests do not pin either cap | r2:test-quality |
| low | Original-code wrong read: a fullwidth-letter mention of the drug bypasses the withdrawal loop, because mentioned_symbols' precheck lowercases without NFKC | r2:test-quality |
| low | patient_text wiring is unpinned: header |  |
| low | Dose, timing, subject and tail vocabulary can be deleted with no failing test (about 50 words/phrases): 'twice a day', 'with meals', 'every morning', 'we take', 'am on', 'managed with', 'on the metfor | r2:test-quality |
| low | z rounding: z_round_7 passes; the test's input 1.0000004 cannot tell 6 from 7 significant digits, so the bug it guards (witness says Elevated, atom says 1) returns undetected | r2:test-quality |
