"""A corpus of plain-text phrasings for the My Patient reader, with the outcome each must get.

Three adversarial review rounds of core/patient_text.py found the same two failure
shapes again and again: a phrasing read as a CONFIDENT, WRONG patient (the worst case
— smoking alone is about 8.8 years on the clock), and a clear, common phrasing refused
(the tab stops being usable). Every reproduction from those rounds is pinned here, next
to the clear phrasings that must keep working, so a fix for one cannot quietly undo
another.

Codes for smoking: C1/C2/C3 current smoker at that cotinine level, F former, N never,
"-" no smoking status and nothing refused, X refused (the build is blocked with a
problem about smoking). Every text is prefixed with "58 year old male" unless it gives
its own age.

    pytest tests/test_patient_text_corpus.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
PLN_CHAT = REPO / "pln_chat"
if str(PLN_CHAT) not in sys.path:
    sys.path.insert(0, str(PLN_CHAT))

from core.patient_text import EXAMPLES, read_patient_text  # noqa: E402

_STATUS = {"C1": ("CurrentSmoker", 1), "C2": ("CurrentSmoker", 2), "C3": ("CurrentSmoker", 3),
           "F": ("FormerSmoker", 0), "N": ("NeverSmoker", 0), "-": (None, None)}


def _text(t: str) -> str:
    return t if "year old" in t.split("\n")[0] or t.startswith("age ") else "58 year old male\n" + t


SMOKING = [
    # ── read as said ────────────────────────────────────────────────────────────
    ("current smoker", "C3"), ("58 year old male, current smoker", "C3"),
    ("i smoke", "C3"), ("I smoke cigarettes", "C3"), ("cigarette smoker", "C3"), ("smokes daily", "C3"),
    ("smoker: yes", "C3"), ("smoking: current", "C3"), ("smoking status: current", "C3"),
    ("heavy smoker for 30 years", "C3"), ("current smoker since 1995", "C3"), ("smoker since 1990", "C3"),
    ("smoker since age 16", "C3"), ("current smoker, since 1990", "C3"), ("current smoker, since age 18", "C3"),
    ("smoking since 1990", "C3"), ("smoker, 1 pack per day", "C3"), ("current smoker, 1 pack a day", "C3"),
    ("current smoker, 20 cigarettes a day", "C3"), ("smokes 10 cigarettes a day", "C3"),
    ("20 cigarettes a day", "C3"), ("can't quit smoking", "C3"), ("trying to quit smoking", "C3"),
    ("smoker, trying to quit", "C3"), ("smoker, no plans to quit", "C3"),
    ("current smoker, no diabetes", "C3"), ("current smoker; no diabetes", "C3"),
    ("smoker, quit 3 times but relapsed", "C3"),
    ("light smoker", "C1"), ("social smoker", "C1"), ("smoke with friends on weekends", "C1"),
    ("moderate smoker", "C2"),
    ("never smoked", "N"), ("non-smoker", "N"), ("not a smoker", "N"), ("never a smoker", "N"),
    ("never been a smoker", "N"), ("I have never been a smoker", "N"), ("smoker: no", "N"),
    ("smokes: no", "N"), ("smoker? no", "N"), ("smoker - no", "N"), ("doesn't smoke", "N"),
    ("never smoked cigarettes", "N"), ("never smoked tobacco", "N"), ("never smoked or vaped", "N"),
    ("never smoked, never vaped", "N"), ("never smoked, does not vape", "N"),
    ("never smoked a day in my life", "N"), ("smoking status: never", "N"), ("smoking status - never", "N"),
    ("Tobacco use: never smoker", "N"), ("tobacco: never", "N"), ("tobacco: none", "N"),
    ("no tobacco use", "N"), ("no tobacco", "N"), ("never used tobacco", "N"),
    ("no history of smoking", "N"), ("denies smoking", "N"), ("denies tobacco use", "N"),
    ("no cigarettes", "N"),
    ("never vaped", "-"), ("doesn't vape", "-"), ("20 pack years", "-"),
    ("previous smoker", "F"), ("former smoker", "F"), ("ex-smoker", "F"), ("former heavy smoker", "F"),
    ("smoker until 2015", "F"), ("quit smoking 20 years ago", "F"), ("was a smoker", "F"),
    ("used to be a smoker", "F"), ("used to smoke", "F"), ("no longer smokes", "F"),
    ("don't smoke anymore", "F"), ("smoker (quit 2010)", "F"), ("smoker - quit 5 years ago", "F"),
    ("former smoker, quit 2010", "F"), ("former smoker, quit in 2010", "F"),
    ("former smoker, quit 10 years ago", "F"), ("ex-smoker, quit 2010", "F"),
    ("former smoker, stopped in 2015", "F"), ("former smoker - quit 2010", "F"),
    ("former smoker (quit 2010)", "F"), ("former smoker (quit in 2010)", "F"),
    ("ex-smoker (quit 15 years ago)", "F"), ("former smoker, quit smoking 2010", "F"),
    ("quit smoking 2010", "F"), ("stopped smoking 2015", "F"), ("smoking: quit 2010", "F"),
    ("former smoker, quit at 40", "F"), ("former smoker, 20 pack years, quit 2010", "F"),
    ("smoking status: former", "F"), ("smoker - no longer", "F"), ("smoker? no longer", "F"),
    ("smoker, quit 2015", "F"), ("smoker, quit in 2010", "F"), ("smoker, until 2015", "F"),
    ("smoker (1990-2015)", "F"), ("smoker, 1990-2015", "F"), ("smoker since 1990 until 2015", "F"),
    ("smoker, since 1990 until 2015", "F"), ("smoker, quit when my son was born", "F"),
    ("smoker, quit for my kids", "F"), ("smoker, until my first child was born", "F"),
    ("smoker, quit smoking in 2010", "F"),
    # ── other statements on the line are not about smoking ─────────────────────
    ("current smoker, stopped drinking in 2019", "C3"), ("smoker, gave up alcohol", "C3"),
    ("never smoked, quit drinking in 2015", "N"), ("non-smoker, retired since 2015", "N"),
    ("never smoked, born 1967", "N"), ("former smoker, married since 1990", "F"),
    ("can't stop snacking", "-"), ("failed to quit alcohol", "-"), ("unable to stop drinking", "-"),
    ("struggling to quit coffee", "-"), ("failed to quit until 2010", "-"),
    ("current smoker, but my wife doesn't", "C3"), ("never smoked, but my wife smokes", "N"),
    ("non-smoker, but exposed to secondhand smoke", "N"),
    ("45 year old female, never smoked\nmy husband smokes", "N"),
    # ── refused: the reader cannot tell, or the text contradicts itself ────────
    ("not a current smoker", "X"), ("not currently a smoker", "X"), ("heavy smoker? not sure", "X"),
    ("smoker - n/a", "X"), ("smoker? n/a", "X"), ("smoker: n/a", "X"),
    ("quit smoking after I failed to quit 5 times", "X"),
    ("current smoker, quit 2015", "X"), ("current smoker, quit for 6 months in 2019", "X"),
    ("current smoker, gave up for 3 months in 2020", "X"), ("current smoker - quit 2015", "X"),
    ("current smoker (quit 2015)", "X"), ("current smoker, stopped 2015", "X"),
    ("smoking status: current smoker, quit 2015", "X"), ("I smoke, quit 2015", "X"),
    ("I'm a smoker, quit 2019", "X"), ("smoker, quit for 6 months in 2019", "X"),
    ("current smoker, quit for a year in 2015", "X"), ("heavy smoker, until 2010", "X"),
    ("light smoker, until 2018", "X"), ("current smoker, until 2019", "X"),
    ("heavy smoker, quit when my wife got pregnant", "X"), ("heavy smoker 1980-2005", "X"),
    ("smoker 2015", "X"), ("smoke-free 2010", "X"), ("never smoked 2020", "X"),
    ("smoking - none since 2010", "X"), ("was a smoker, still am", "X"),
    ("used to be a smoker, still am", "X"), ("smoker (quit 2015, started again)", "X"),
    ("smoker - quit 2015 - back on it", "X"), ("I no longer smoke a pack a day, down to 5", "X"),
    ("no longer smoke as much", "X"), ("no longer smoking heavily", "X"),
    ("I was a smoker but started again", "X"), ("was a smoker and still am", "X"),
    ("was unable to quit smoking until 2015", "X"), ("struggled to stop smoking until 2012", "X"),
    ("non-smoker, but smoke with friends on weekends", "X"),
    ("never smoked, but started with my friends last year", "X"),
    ("vapes daily", "X"), ("smoked for 20 years", "X"), ("my wife and I smoke", "X"),
    ("current smoker\nsmoker, quit 2015", "X"),
    # ── round 4 (docs/patient_extraction/review_round4.json): each was read as a
    #    confident, wrong status; refused now, or read right (conf<n> = the finding) ──
    ("I never quit smoking", "X"), ("couldn't quit smoking", "X"), ("tried to quit smoking", "X"),   # conf0
    ("I need to quit smoking", "X"), ("on Chantix to quit smoking", "X"),
    ("using patches to quit smoking", "X"), ("quit smoking soon", "X"),
    ("smoked a pack a day for 40 years", "X"), ("I smoked 20 cigarettes a day", "X"),                   # conf1
    ("Smoker: Nil", "X"), ("Smoker: negative", "X"), ("Smoker: denies", "X"), ("Smoker: 0", "X"),      # conf2
    ("Smoker: false", "X"), ("Smoker: -", "X"),
    ("0 cigarettes a day", "X"),                                                                      # conf3
    ("I smoke weed", "X"), ("cannabis smoker", "X"), ("smokes marijuana", "X"),                       # conf4, 51
    ("smoker in my youth", "X"), ("prior smoker", "F"), ("formr smoker", "X"),                         # conf5, 18
    ("smoker from age 16 to 40", "X"), ("smoker in college", "X"), ("smoker previously", "F"),
    ("smoker, now quit", "X"), ("smoker, quit.", "X"), ("smoker, quit ten years ago", "X"),           # conf6, 19
    ("smoker, recently quit", "X"), ("current smoker, recently quit", "X"), ("I smoke, just quit", "X"),
    ("smoker\nquit 2015", "X"), ("former smoker\nrelapsed last year", "X"),                          # conf7
    ("smoker since 1990 till now", "X"),                                                              # conf8
    ("quit cigarettes but smoke cigars", "X"),                                                        # conf9
    ("non-smoker but vape", "X"), ("never smoked but chew tobacco", "X"),                             # conf10, 52
    ("former smoker on nicotine patches", "X"), ("never smoked; uses snus daily", "X"),
    ("never smoked but vapes", "X"), ("quit smoking in 2015 and switched to vaping", "X"),
    ("occasionally smokes", "X"), ("smokes 2 cigarettes a day", "X"), ("Smoker: weekends only", "X"),   # conf11
    ("light smoker (20 cigarettes a day)", "X"), ("smokes cigars occasionally", "X"),
    ("smoker since my wife died", "X"), ("heavy smoker like my dad", "X"),                            # conf12, 21, 45
    ("smokes with friends on weekends", "X"), ("current smoker since my father died", "X"),
    ("current smoker (wife also smokes)", "X"), ("I smoke with my friends on weekends", "X"),
    ("Never smoker: no", "X"), ("Smoker, current status unknown", "X"),                               # conf13
    ("Smoking status: Never assessed", "X"), ("Smokeless tobacco: never used", "-"),
    ("Smoking status: Former smoker\nSmokeless tobacco: Never used", "F"),
    ("quit smoking for a year in 2015", "X"), ("gave up smoking for Lent", "X"),                      # conf14
    ("stopped smoking for a while", "X"),
    ("Smoke-free for 10 years", "X"), ("Smoker: no, but used to", "X"), ("Non-smoker for 10 years", "X"),  # conf15, 23
    ("smoker for the past 20 years", "X"), ("smoker, started 30 years ago", "C3"),                    # conf20
    ("light smoker\n20 cigarettes a day", "X"), ("social smoker, 1 pack a day", "X"),                # conf24
    ("heavy smoker, smokes socially", "X"),
    ("smoked for 30 years", "X"),                                                                     # conf47
    ("smoker since I was 16 years old", "C3"), ("quit smoking at 45 years old", "F"),                 # conf48
    ("no history of smoking or diabetes", "N"),                                                       # conf49
    ("nonsmoker", "N"), ("exsmoker", "F"), ("chainsmoker", "C3"), ("I am a nonsmoker", "N"),          # conf54, 30
    ("non\u2013smoker", "N"), ("non\u00a0smoker", "N"), ("non- smoker", "X"), ("non.smoker", "X"),     # crit0
    ("can\u2019t quit smoking", "C3"), ("don\u2019t smoke", "N"), ("I\u2019ve never smoked", "N"),    # crit1
    ("smoker who was hospitalized", "X"),                                                             # crit3
    ("former smoker cotinine 300 ng/mL", "X"),                                                        # conf46
]


@pytest.mark.parametrize("text, expect", SMOKING, ids=[t for t, _ in SMOKING])
def test_smoking(text, expect):
    p = read_patient_text(_text(text))
    if expect == "X":
        assert not p.ok, (p.smoking, p.cotinine_level)
        assert any("smok" in x or "vap" in x for x in p.all_problems()), p.all_problems()
        return
    assert (p.smoking, p.cotinine_level) == _STATUS[expect], (p.all_problems(), p.not_understood)
    assert p.ok, p.all_problems()


DIAGNOSES = [
    # (text, ok, {item: answer})
    ("has diabetes, since 2015\nhypertension", True, {"DIQ010": 1, "BPQ020": 1}),
    ("current smoker, hypertension, since 2010\nno diabetes", True, {"BPQ020": 1, "DIQ010": 2}),
    ("asthma, but otherwise healthy", True, {"MCQ010": 1, "MCQ220": 2}),
    ("hypertension, otherwise in good health", True, {"BPQ020": 1, "MCQ220": 2}),
    ("asthma, otherwise fit and well", True, {"MCQ010": 1}),
    ("diagnoses: hypertension, high cholesterol\nno other conditions", True, {"BPQ020": 1, "MCQ220": 2}),
    ("had a stroke\nno other conditions", True, {"MCQ160F": 1}),
    ("had a stroke in 2019\nno other conditions", True, {"MCQ160F": 1}),
    ("mild hypertension\nno other conditions", True, {"BPQ020": 1}),
    ("high BP\nno other conditions", True, {"BPQ020": 1}),
    ("T2DM\nno other conditions", True, {"DIQ010": 1}),
    ("type 2 diabetes (2015)\nno other diagnoses", True, {"DIQ010": 1}),
    ("diagnoses: hypertension\nCKD stage 3", True, {"KIQ020": 1, "BPQ020": 1}),
    # HUQ070 is an overnight stay in the PAST 12 MONTHS: a dated older stay is refused, not
    # a Yes (round 4, conf41; this row expected HUQ070 = 1 until the reader was hardened)
    ("hypertension\nhospitalized in 2020\nno other conditions", False, {}),
    ("hypertension\nhospitalized in the past year\nno other conditions", True, {"HUQ070": 1}),
    ("diagnoses: osteoporosis, hypothyroidism\nbroken wrist in 2019\nno other conditions", True,
     {"OSQ010B": 1, "OSQ060": 1, "MCQ160I": 1}),
    ("58 year old male with diabetes 2015\nno other conditions", True, {"DIQ010": 1}),
    ("58 year old male with hypertension diagnosed 2015\nno other conditions", True, {"BPQ020": 1}),
    ("hypertension\nno conditions apart from hypertension", True, {"BPQ020": 1}),
    ("no known conditions except diabetes\ndiagnoses: diabetes", True, {"DIQ010": 1}),
    ("no diabetes, prediabetes", True, {"DIQ010": 3}),
    ("prediabetes, no diabetes", True, {"DIQ010": 3}),
    ("no diabetes\nprediabetes", True, {"DIQ010": 3}),
    ("diagnoses: prediabetes\nno diabetes", True, {"DIQ010": 3}),
    ("has prediabetes, not diabetic", True, {"DIQ010": 3}),
    ("history of prediabetes, no diabetes", True, {"DIQ010": 3}),
    ("prediabetes; no diabetes", True, {"DIQ010": 3}),
    ("diagnoses: hypertension\ntakes lisinopril", True, {"BPQ020": 1}),
    ("diagnoses: hypertension\nno medications", True, {"BPQ020": 1}),
    ("diagnoses: hypertension\nexercises 3 times a week", True, {"BPQ020": 1}),
    ("no known conditions\nallergic to penicillin", True, {"BPQ020": 2}),
    ("no known conditions\nvegetarian", True, {"BPQ020": 2}),
    ("diagnoses: hypertension\nhigh cholesterol", True, {"BPQ020": 1}),
    ("diagnoses: hypertension\nself-reported health: good", True, {"HUQ010": 3}),
    ("diagnoses: hypertension\nhealth vs last year: same", True, {"HUQ020": 3}),
    ("diagnoses: hypertension\nhealth status: good", True, {"HUQ010": 3}),
    ("diagnoses: hypertension\nmy health is good", True, {"HUQ010": 3}),
    ("diagnoses: hypertension\ndoctor's visits: 2", True, {"HUQ050": 2}),
    ("diagnoses: none\nno known conditions", True, {"BPQ020": 2}),
    ("diagnoses: hypertension\n25-OH vitamin D 30 ng/mL", True, {"BPQ020": 1}),
    ("no known conditions\nvitamin D 30 ng/mL (low)", True, {"BPQ020": 2}),
    ("diagnoses: hypertension\nno other known conditions", True, {"BPQ020": 1, "MCQ220": 2}),
    ("diagnoses: hypertension\nno other medical conditions", True, {"BPQ020": 1}),
    ("diagnoses: hypertension\nno other medical problems", True, {"BPQ020": 1}),
    ("diagnoses: hypertension\nno other illnesses", True, {"BPQ020": 1}),
    ("diagnoses: hypertension\nno other health issues", True, {"BPQ020": 1}),
    ("diagnoses: hypertension\nnothing else", True, {"BPQ020": 1}),
    ("diagnoses: hypertension\ngenerally healthy", True, {"BPQ020": 1}),
    ("diagnoses: hypertension\nwalks daily", True, {"BPQ020": 1}),
    ("diagnoses: hypertension\nno allergies", True, {"BPQ020": 1}),
    ("58 year old male, healthy\nno known conditions", True, {"BPQ020": 2}),
    ("no known conditions except high cholesterol", True, {"BPQ020": 2, "MCQ220": 2}),
    ("no known conditions except mild asthma", True, {"MCQ010": 1}),
    ("no known conditions except hypertension, asthma", True, {"BPQ020": 1, "MCQ010": 1}),
    ("no known conditions except hypertension, asthma and arthritis", True,
     {"BPQ020": 1, "MCQ010": 1, "MCQ160A": 1}),
    ("no conditions apart from hypertension, diabetes and asthma", True,
     {"BPQ020": 1, "DIQ010": 1, "MCQ010": 1}),
    ("I have no known conditions, current smoker", True, {"BPQ020": 2}),
    ("I have no diabetes, current smoker", True, {"DIQ010": 2}),
    ("history: no stroke, smoker", True, {"MCQ160F": 2}),
    ("medical history: hypertension, not diabetic, smoker", True, {"BPQ020": 1, "DIQ010": 2}),
    ("diagnoses: hypertension, no diabetes, current smoker", True, {"BPQ020": 1, "DIQ010": 2}),
    ("I have diabetes, never smoked, currently on insulin", True, {"DIQ010": 1}),
    ("I have asthma now, never smoked", True, {"MCQ010": 1}),
    ("62 year old male\ndiagnoses: hypertension, diabetes\nno other conditions", True,
     {"BPQ020": 1, "DIQ010": 1, "MCQ220": 2}),
    ("hypertension, no diabetes and prediabetes", True, {"BPQ020": 1, "DIQ010": 3}),
    # refused
    ("diagnoses: diabetes\nno diabetes", False, {}),
    ("no known conditions other than diabetes\nhypertension", False, {}),
    ("no known conditions\nno other conditions except diabetes", False, {}),
    ("no medical history\nno known conditions except diabetes", False, {}),
    ("diagnoses: hypertension\nno known conditions", False, {}),
    ("no known conditions\ndiagnoses: hypertension", False, {}),
    ("heart disease\nno other conditions", False, {}),
    ("hypertension on lisinopril", False, {}),
    # ── round 4: a negation heading a list covers it, or the list is refused (conf31, 55)
    ("no diabetes or hypertension", True, {"DIQ010": 2, "BPQ020": 2}),
    ("denies HTN, DM2, CAD", True, {"BPQ020": 2, "DIQ010": 2, "MCQ160C": 2}),
    ("free of diabetes and hypertension", True, {"DIQ010": 2, "BPQ020": 2}),
    ("never had a heart attack or stroke", True, {"MCQ160E": 2, "MCQ160F": 2}),
    ("no history of cancer, stroke, or heart attack", True, {"MCQ220": 2, "MCQ160F": 2, "MCQ160E": 2}),
    ("negative for diabetes, hypertension and stroke", True, {"DIQ010": 2, "BPQ020": 2, "MCQ160F": 2}),
    ("no diabetes or prediabetes", True, {"DIQ010": 2}),
    ("no T2DM, HTN or CVA", False, {}),
    ("diagnosed with asthma or COPD, not sure which", False, {}),
    #    sentences are statements (conf32)
    ("No diabetes. Hypertension.", True, {"DIQ010": 2, "BPQ020": 1}),
    ("Never had cancer. Had a stroke in 2019.", True, {"MCQ220": 2, "MCQ160F": 1}),
    ("No kidney disease. Type 2 diabetes since 2015.", True, {"KIQ020": 2, "DIQ010": 1}),
    #    a health line is read whole (conf33)
    ("health: good apart from diabetes\nno other conditions", False, {}),
    ("my health is good but I have diabetes", False, {}),
    #    the person's own condition next to someone else's (conf35, 50)
    ("I have COPD, my father had a stroke", True, {"MCQ160G": 1}),
    ("had asthma as a child\nno other conditions", False, {}),
    ("diagnoses: hypertension, family history of diabetes\nno other conditions", True,
     {"BPQ020": 1, "DIQ010": 2}),
    ("diagnoses: hypertension, diabetes, father had a stroke", True, {"BPQ020": 1, "DIQ010": 1}),
    ("I have diabetes like my mother", False, {}),
    #    a negation that qualifies treatment or time (conf36)
    ("never treated hypertension", False, {}), ("not well controlled diabetes", False, {}),
    ("never hospitalized for asthma", False, {}), ("no asthma since 2010", False, {}),
    ("no recent stroke", False, {}),
    #    a synonym the rules do not read is refused, never answered No (conf37, 56; crit4, 5)
    ("diagnoses: hypertension, MI 2015", False, {}), ("hypertensive\nno other conditions", False, {}),
    ("diagnoses: hypertensive, asthma", False, {}), ("diagnoses: hypertension, two strokes", False, {}),
    ("diagnoses: hypertension, tumours", False, {}), ("fractured hip and wrist", False, {}),
    #    a form's empty or 0 answer is not a Yes (conf38)
    ("diabetes: 0", False, {}), ("stroke: -", False, {}), ("asthma - 0", False, {}),
    #    a condition in a smoking 'when' clause (conf22, 39, 44)
    ("former smoker, quit after my heart attack in 2015\nno other conditions", False, {}),
    ("current smoker with T2DM", False, {}),
    ("medical history: smoker, quit 2015, hypertension, diabetes", False, {}),
    ("no known conditions except asthma, former smoker, quit 2010, hypertension", False, {}),
    ("former smoker, quit 2010. diabetes.\nno other conditions", True, {"DIQ010": 1}),
    #    a time-limited item outside its window (conf41)
    ("hospitalized 15 years ago\nno other conditions", False, {}),
    ("history of anemia\nno other conditions", False, {}),
    #    a lab value inside a diagnosis line is read as a lab (conf57)
    ("I have diabetes, HbA1c 7.1 %", True, {"DIQ010": 1}),
]


@pytest.mark.parametrize("text, ok, items", DIAGNOSES, ids=[t for t, _, _ in DIAGNOSES])
def test_diagnoses(text, ok, items):
    p = read_patient_text(_text(text))
    assert p.ok is ok, (p.all_problems(), p.not_understood, p.questionnaire_notes)
    for item, value in items.items():
        assert p.questionnaire.get(item) == value, (item, p.questionnaire, p.all_problems())


#: Round 4, the rest: (text, ok, {attribute or questionnaire item or lab code: value}).
OTHER = [
    # contradictory health answers (conf34)
    ("self-rated health: excellent\nself-rated health: poor", False, {}),
    ("doctor visits: 2\ndoctor visits: 15", False, {}),
    # visits are a yearly count; anything else is not read (conf40, crit10)
    ("doctor visits: 2 per month", True, {"HUQ050": None}),
    ("doctor visits: 3-5", True, {"HUQ050": None}),
    ("doctor visits: 10 in the last 5 years", True, {"HUQ050": None}),
    ("doctor visits: 4 per year", True, {"HUQ050": 3}),
    # a comparison with other people is not health vs a year ago (conf42)
    ("health: poor\nhealth is better than most people my age", True, {"HUQ010": 5, "HUQ020": None}),
    ("health is better than a year ago", True, {"HUQ020": 1}),
    # cotinine as typed is read, and an unread cotinine line blocks (conf46)
    ("cotinine 250 ng/mL.", True, {"cotinine_level": 3}), ("cotinine is 250 ng/mL", True, {"cotinine_level": 3}),
    ("cotinine 250 ng / mL", True, {"cotinine_level": 3}),
    ("cotinine 250 ng/mL (high)", False, {}), ("current smoker\ncotinine <10 ng/mL", False, {}),
    # an age phrase about something else is not the person's age (conf48, crit2)
    ("my biological age is 52 years old", True, {"age": 58}),
    ("retired at 55 years old", True, {"age": 58}),
    ("58 year old female, never smoked\nmenopause at 50 years old", True, {"age": 58, "sex": "Female"}),
    # a typed BMI that contradicts weight and height; two GrimAge accelerations (crit11)
    ("BMI 30\nweight 70 kg\nheight 180 cm", False, {}),
    ("GrimAge acceleration +3 years\nGrimAge acceleration -2 years", False, {}),
    # a lab value inside a diagnosis line (conf57)
    ("I have diabetes, HbA1c 7.1 %", True, {"LBXGH": 7.1}),
    ("no known conditions except prediabetes, fasting glucose 110 mg/dL", True, {"DIQ010": 3}),
]


@pytest.mark.parametrize("text, ok, checks", OTHER, ids=[t for t, _, _ in OTHER])
def test_round4_other(text, ok, checks):
    p = read_patient_text(_text(text))
    assert p.ok is ok, (p.all_problems(), p.not_understood)
    for key, value in checks.items():
        if hasattr(p, key):
            got = getattr(p, key)
        elif key in {r.code for r in p.readings}:
            got = next(r.value for r in p.readings if r.code == key)
        else:
            got = p.questionnaire.get(key)
        assert got == (pytest.approx(value) if isinstance(value, float) else value), (key, got)


def test_the_smoking_of_a_header_line_is_read():
    p = read_patient_text("58 year old male\nI have no known conditions, current smoker")
    assert (p.smoking, p.cotinine_level) == ("CurrentSmoker", 3)


def test_the_tab_examples_still_read_cleanly():
    for name, text in EXAMPLES.items():
        p = read_patient_text(text)
        assert p.ok and not p.not_understood, (name, p.all_problems(), p.not_understood)
