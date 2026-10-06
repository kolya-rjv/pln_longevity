# Model reader — live evaluation (gpt-6-luna)

Run 2026-10-06: 1483 texts (234 with an expected reading, 184 the old rules refused, 1065 round-4 reproductions), wall 393 s.

## Verdict

**PASS**: no text with an expected reading was read as a usable patient with a different value, and every text was read by the model.

## Texts with an expected reading

| | count |
|---|---|
| read as expected | 225/234 (96%) |
| asked about (not usable until reworded) | 9/234 (4%) |
| a value missing (the statement listed as not used) | 0/234 (0%) |
| **usable but different** | 0/234 (0%) |

- corpus:body: 12/13 (92%) as expected
- corpus:diagnoses: 80/82 (98%) as expected
- corpus:smoking: 133/139 (96%) as expected

Asked about (the person rewords; nothing wrong is built):

- `'58 year old male\nsmoke with friends on weekends'`: 'smoke with friends on weekends': cannot tell what this says about your smoking (The text does not specify whether this is tobacco smoking.); write 'current smoker', 'former smoker' or 'never smoked'
- `'58 year old male\nnever smoked, does not vape'`: 'does not vape': cannot tell whether you smoke now, used to, or never did; write 'current smoker', 'former smoker, quit 2010' or 'never smoked'
- `'58 year old male\nwas a smoker'`: 'was a smoker': cannot tell whether you smoke now, used to, or never did; write 'current smoker', 'former smoker, quit 2010' or 'never smoked'
- `'58 year old male\nsmoker, 1990-2015'`: 'smoker, 1990-2015': cannot tell what this says about your smoking (The date range ends in 2015, but “smoker” can indicate current smoking; it is unclear whether the person smokes now.); write 'current smoker', 'former smoker' or 'never smo
- `'58 year old male\nsmoker, quit for my kids'`: 'smoker, quit for my kids': cannot tell what this says about your smoking (It says both “smoker” and “quit,” so current versus former smoking is unclear.); write 'current smoker', 'former smoker' or 'never smoked'
- `'58 year old male\nfailed to quit until 2010'`: 'failed to quit until 2010': cannot tell what this says about your smoking (It is unclear whether they quit in 2010 or were still smoking after that.); write 'current smoker', 'former smoker' or 'never smoked'
- `'58 year old male\nfree of diabetes and hypertension'`: 'free of hypertension': a condition was read here but could not be checked against your text (the quote is not in the text), so it would be answered wrongly; write it as 'diagnoses: …' or 'no …' with the condition's name
- `'58 year old male\nnegative for diabetes, hypertension and stroke'`: 'negative for hypertension': a condition was read here but could not be checked against your text (the quote is not in the text), so it would be answered wrongly; write it as 'diagnoses: …' or 'no …' with the condition's name; 'negative for
- `'Male, 61. Height 1.75 m, weight 90 kg.'`: no age found (e.g. '58 year old' or 'age 58')

## Texts the old rules refused

The model asked about 75/184 (41%) of them and read 109 as usable. A usable reading here is a judgement to review, not a failure: the rules refused what they could not parse, not only what is ambiguous.

- `'58 year old male\nquit smoking after I failed to quit 5 times'` → smoking FormerSmoker (level 0)
- `'58 year old male\ncurrent smoker, quit for 6 months in 2019'` → smoking CurrentSmoker (level 3)
- `'58 year old male\ncurrent smoker, gave up for 3 months in 2020'` → smoking CurrentSmoker (level 3)
- `"58 year old male\nI'm a smoker, quit 2019"` → smoking FormerSmoker (level 0)
- `'58 year old male\nsmoker, quit for 6 months in 2019'` → smoking CurrentSmoker (level 3)
- `'58 year old male\ncurrent smoker, quit for a year in 2015'` → smoking CurrentSmoker (level 3)
- `'58 year old male\nheavy smoker, until 2010'` → smoking FormerSmoker (level 0)
- `'58 year old male\nlight smoker, until 2018'` → smoking FormerSmoker (level 0)
- `'58 year old male\nheavy smoker, quit when my wife got pregnant'` → smoking FormerSmoker (level 0)
- `'58 year old male\nsmoker 2015'` → smoking CurrentSmoker (level 3)
- `'58 year old male\nsmoke-free 2010'` → smoking FormerSmoker (level 0)
- `'58 year old male\nnever smoked 2020'` → smoking NeverSmoker (level 0)
- `'58 year old male\nsmoking - none since 2010'` → smoking FormerSmoker (level 0)
- `'58 year old male\nwas a smoker, still am'` → smoking CurrentSmoker (level 3)
- `'58 year old male\nused to be a smoker, still am'` → smoking CurrentSmoker (level 3)
- `'58 year old male\nsmoker (quit 2015, started again)'` → smoking CurrentSmoker (level 3)
- `'58 year old male\nsmoker - quit 2015 - back on it'` → smoking CurrentSmoker (level 3)
- `'58 year old male\nI no longer smoke a pack a day, down to 5'` → smoking CurrentSmoker (level 3)
- `'58 year old male\nI was a smoker but started again'` → smoking CurrentSmoker (level 3)
- `'58 year old male\nwas a smoker and still am'` → smoking CurrentSmoker (level 3)
- `'58 year old male\nwas unable to quit smoking until 2015'` → smoking FormerSmoker (level 0)
- `'58 year old male\nstruggled to stop smoking until 2012'` → smoking FormerSmoker (level 0)
- `'58 year old male\nnon-smoker, but smoke with friends on weekends'` → smoking CurrentSmoker (level 1)
- `'58 year old male\nI never quit smoking'` → smoking CurrentSmoker (level 3)
- `"58 year old male\ncouldn't quit smoking"` → smoking CurrentSmoker (level 3)
- `'58 year old male\nI need to quit smoking'` → smoking CurrentSmoker (level 3)
- `'58 year old male\non Chantix to quit smoking'` → smoking CurrentSmoker (level 3)
- `'58 year old male\nSmoker: Nil'` → smoking NeverSmoker (level 0)
- `'58 year old male\nSmoker: negative'` → smoking NeverSmoker (level 0)
- `'58 year old male\nSmoker: denies'` → smoking NeverSmoker (level 0)
- `'58 year old male\nSmoker: 0'` → smoking NeverSmoker (level 0)
- `'58 year old male\nSmoker: false'` → smoking NeverSmoker (level 0)
- `'58 year old male\n0 cigarettes a day'` → smoking NeverSmoker (level 0)
- `'58 year old male\nsmokes marijuana'` → smoking None (level None)
- `'58 year old male\nsmoker in my youth'` → smoking FormerSmoker (level 0)
- `'58 year old male\nsmoker from age 16 to 40'` → smoking FormerSmoker (level 0)
- `'58 year old male\nsmoker in college'` → smoking FormerSmoker (level 0)
- `'58 year old male\nsmoker, now quit'` → smoking FormerSmoker (level 0)
- `'58 year old male\nsmoker, quit.'` → smoking FormerSmoker (level 0)
- `'58 year old male\nsmoker, quit ten years ago'` → smoking FormerSmoker (level 0)
- `'58 year old male\nsmoker, recently quit'` → smoking FormerSmoker (level 0)
- `'58 year old male\nsmoker\nquit 2015'` → smoking FormerSmoker (level 0)
- `'58 year old male\nsmoker since 1990 till now'` → smoking CurrentSmoker (level 3)
- `'58 year old male\nquit cigarettes but smoke cigars'` → smoking CurrentSmoker (level 3)
- `'58 year old male\noccasionally smokes'` → smoking CurrentSmoker (level 1)
- `'58 year old male\nsmokes 2 cigarettes a day'` → smoking CurrentSmoker (level 3)
- `'58 year old male\nSmoker: weekends only'` → smoking CurrentSmoker (level 1)
- `'58 year old male\nlight smoker (20 cigarettes a day)'` → smoking CurrentSmoker (level 3)
- `'58 year old male\nsmokes cigars occasionally'` → smoking CurrentSmoker (level 1)
- `'58 year old male\nsmoker since my wife died'` → smoking CurrentSmoker (level 3)
- `'58 year old male\nsmokes with friends on weekends'` → smoking CurrentSmoker (level 1)
- `'58 year old male\ncurrent smoker (wife also smokes)'` → smoking CurrentSmoker (level 3)
- `'58 year old male\nI smoke with my friends on weekends'` → smoking CurrentSmoker (level 1)
- `'58 year old male\nSmoke-free for 10 years'` → smoking FormerSmoker (level 0)
- `'58 year old male\nSmoker: no, but used to'` → smoking FormerSmoker (level 0)
- `'58 year old male\nNon-smoker for 10 years'` → smoking FormerSmoker (level 0)
- `'58 year old male\nsmoker for the past 20 years'` → smoking CurrentSmoker (level 3)
- `'58 year old male\nlight smoker\n20 cigarettes a day'` → smoking CurrentSmoker (level 3)
- `'58 year old male\nsocial smoker, 1 pack a day'` → smoking CurrentSmoker (level 3)
- `'58 year old male\nheavy smoker, smokes socially'` → smoking CurrentSmoker (level 1)

## Round-4 reproductions

The texts the rules reader's fourth review round broke (no expected reading): read as usable 829/1065 (78%), asked about 236.

- `'tried to quit smoking'` — 'tried to quit smoking': cannot tell whether you smoke now, used to, or never did; write 'current smoker', 'former smoker, quit 2010' or 'never smoked'
- `'attempted to quit smoking'` — 'attempted to quit smoking': cannot tell what this says about your smoking (Does not establish whether he currently smokes, previously smoked, or never smoked.); write 'current smoker', 'former smoker
- `'should quit smoking'` — 'should quit smoking': cannot tell what this says about your smoking (Does not establish whether the person currently or ever smoked.); write 'current smoker', 'former smoker' or 'never smoked'
- `'my doctor says I should quit smoking'` — 'my doctor says I should quit smoking': cannot tell what this says about your smoking (Says they should quit, but does not state whether they currently smoke or smoked in the past.); write 'current sm
- `'quit smoking soon'` — 'quit smoking soon': cannot tell what this says about your smoking (It does not say whether they currently smoke or smoked in the past.); write 'current smoker', 'former smoker' or 'never smoked'
- `'using patches to quit smoking'` — 'using patches to quit smoking': nicotine replacement raises cotinine, which LinAge2 reads, but is not smoking to the knowledge base; give 'cotinine N ng/mL' from a test, or remove the nicotine replac
- `'58 year old male, tried to quit smoking many times'` — 'tried to quit smoking many times': cannot tell whether you smoke now, used to, or never did; write 'current smoker', 'former smoker, quit 2010' or 'never smoked'
- `'58 year old male, smoked a pack a day for 40 years'` — 'smoked a pack a day for 40 years': cannot tell whether you smoke now, used to, or never did; write 'current smoker', 'former smoker, quit 2010' or 'never smoked'
- `'58 year old male, I smoked 20 cigarettes a day'` — 'I smoked 20 cigarettes a day': cannot tell whether you smoke now, used to, or never did; write 'current smoker', 'former smoker, quit 2010' or 'never smoked'
- `'smoked 2 packs a day for 30 years'` — 'smoked 2 packs a day for 30 years': cannot tell whether you smoke now, used to, or never did; write 'current smoker', 'former smoker, quit 2010' or 'never smoked'
- `'I smoked a pack a day'` — 'I smoked a pack a day': cannot tell what this says about your smoking (Does not say whether he currently smokes, quit, or never smoked.); write 'current smoker', 'former smoker' or 'never smoked'
- `'had 20 cigarettes a day'` — 'had 20 cigarettes a day': cannot tell what this says about your smoking (Does not establish whether this is current or past smoking.); write 'current smoker', 'former smoker' or 'never smoked'
- `'quit 2012'` — 'quit 2012': cannot tell what this says about your smoking (It does not say what they quit.); write 'current smoker', 'former smoker' or 'never smoked'
- `'denies'` — 'denies': cannot tell whether this is one of the conditions LinAge2 counts (No condition or other referent is specified.); write it as 'diagnoses: …' or 'no …' with the condition's name
- `'Current smoker: false'` — 'Current smoker: false': cannot tell whether you smoke now, used to, or never did; write 'current smoker', 'former smoker, quit 2010' or 'never smoked'
- `'Smoker: -'` — 'Smoker: -': cannot tell what this says about your smoking (The dash does not clarify smoking status.); write 'current smoker', 'former smoker' or 'never smoked'
- `'Smoker: [ ]'` — 'Smoker: [ ]': cannot tell what this says about your smoking (An empty checkbox does not clarify smoking status.); write 'current smoker', 'former smoker' or 'never smoked'
- `'Smoker: absent'` — 'Smoker: absent': cannot tell what this says about your smoking (“Absent” does not clarify whether the person never smoked, formerly smoked, or smokes now.); write 'current smoker', 'former smoker' or
- `'Smoker: non'` — 'Smoker: non': cannot tell what this says about your smoking (The smoking status is incomplete or ambiguous.); write 'current smoker', 'former smoker' or 'never smoked'
- `'I smoke weed'` — 'I smoke weed': cannabis is not tobacco, and LinAge2 reads tobacco exposure (cotinine); say your tobacco smoking on its own ('never smoked', 'former smoker', 'current smoker')
- `'weed'` — 'weed': cannabis is not tobacco, and LinAge2 reads tobacco exposure (cotinine); say your tobacco smoking on its own ('never smoked', 'former smoker', 'current smoker')
- `'smokes weed'` — 'smokes weed': cannabis is not tobacco, and LinAge2 reads tobacco exposure (cotinine); say your tobacco smoking on its own ('never smoked', 'former smoker', 'current smoker')
- `'smokes marijuana daily'` — 'smokes marijuana daily': cannabis is not tobacco, and LinAge2 reads tobacco exposure (cotinine); say your tobacco smoking on its own ('never smoked', 'former smoker', 'current smoker')
- `'I smoke cannabis'` — 'I smoke cannabis': cannabis is not tobacco, and LinAge2 reads tobacco exposure (cotinine); say your tobacco smoking on its own ('never smoked', 'former smoker', 'current smoker')
- `'cannabis smoker'` — 'cannabis smoker': cannabis is not tobacco, and LinAge2 reads tobacco exposure (cotinine); say your tobacco smoking on its own ('never smoked', 'former smoker', 'current smoker')

## What the checks discarded

- Items discarded: 30 of 4706 (condition 12, smoking 11, healthcare_visits 4, age 2, sex 1).

- 13 × in what the model says is about someone else
- 12 × the quote is not in the text
- 4 × the number is not in the quote
- 1 × the quote does not say male

## Latency and tokens

- Model call: p50 2.2 s, p95 3.5 s, max 11.1 s (1308 calls).
- Tokens: prompt 8,849,796 (cached 8,791,504), completion 207,759 (reasoning 92,690); per call 6,765 in, 158 out.

