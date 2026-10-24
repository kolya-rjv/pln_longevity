; MixCompound: the five-row example of fork_design/designs/design1_mp_state.md section 10.
; One row per paper, so the paper-level fold is the row-level fold. Hand-built, not DrugAge data.
; Expected, typed by hand from Truth_Revision (weights c/(1-c)): Neg rows 9001-9004 -> 0.263050 / 0.949466 native,
;   x 0.9 chain discount = 0.854519; Pos row 9005 -> 0.6 / 0.45; net = 0.263050*0.854519 - 0.6*0.45 = -0.0452.

(InstanceOf DrugAgeRow_9001 Experiment)
(UsesIntervention DrugAgeRow_9001 MixCompound)
(UsesSpecies DrugAgeRow_9001 Mus_musculus)
(HasSex DrugAgeRow_9001 Male)
(ReportedIn DrugAgeRow_9001 PMID_1001)
(IsITPStudy DrugAgeRow_9001)
(AvgLifespanChangePercent DrugAgeRow_9001 20.0)
(AvgLifespanSignificance DrugAgeRow_9001 Significant)

(InstanceOf DrugAgeRow_9002 Experiment)
(UsesIntervention DrugAgeRow_9002 MixCompound)
(UsesSpecies DrugAgeRow_9002 Caenorhabditis_elegans)
(HasSex DrugAgeRow_9002 Unknown)
(ReportedIn DrugAgeRow_9002 PMID_1002)
(AvgLifespanChangePercent DrugAgeRow_9002 40.0)
(AvgLifespanSignificance DrugAgeRow_9002 Significant)

(InstanceOf DrugAgeRow_9003 Experiment)
(UsesIntervention DrugAgeRow_9003 MixCompound)
(UsesSpecies DrugAgeRow_9003 Mus_musculus)
(HasSex DrugAgeRow_9003 Female)
(ReportedIn DrugAgeRow_9003 PMID_1003)
(AvgLifespanChangePercent DrugAgeRow_9003 10.0)
(AvgLifespanSignificance DrugAgeRow_9003 NotSignificant)

(InstanceOf DrugAgeRow_9004 Experiment)
(UsesIntervention DrugAgeRow_9004 MixCompound)
(UsesSpecies DrugAgeRow_9004 Mus_musculus)
(HasSex DrugAgeRow_9004 Female)
(ReportedIn DrugAgeRow_9004 PMID_1004)
(IsITPStudy DrugAgeRow_9004)
(AvgLifespanChangePercent DrugAgeRow_9004 0.0)
(AvgLifespanSignificance DrugAgeRow_9004 NotSignificant)

(InstanceOf DrugAgeRow_9005 Experiment)
(UsesIntervention DrugAgeRow_9005 MixCompound)
(UsesSpecies DrugAgeRow_9005 Mus_musculus)
(HasSex DrugAgeRow_9005 Male)
(ReportedIn DrugAgeRow_9005 PMID_1005)
(AvgLifespanChangePercent DrugAgeRow_9005 -30.0)
(AvgLifespanSignificance DrugAgeRow_9005 Significant)
