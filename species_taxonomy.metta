;; ==========================================
;; SPECIES TAXONOMY (From DrugAge CSV)
;; ==========================================
;; Uses the user-defined Inheritance predicate (declared in
;; logical_predicates.metta) for taxonomic subsumption that PLN
;; deduction / induction rules can operate on.
;;
;; Convention:
;;   (Inheritance <species> <higher-taxon>)
;;
;; Higher taxa (Vertebrate, Invertebrate, Fungi, Protozoa) are
;; themselves placed under Taxon via Inheritance in system_types.metta.

;; ==== VERTEBRATES ====
;; Mammals
(Inheritance Mus_musculus Vertebrate)
(Inheritance Rattus_norvegicus Vertebrate)
(Inheritance Mesocricetus_auratus Vertebrate)
(Inheritance Canis_lupus_familiaris Vertebrate)

;; Fish
(Inheritance Nothobranchius_furzeri Vertebrate)
(Inheritance Nothobranchius_guentheri Vertebrate)

;; ==== INVERTEBRATES ====
;; Nematodes
(Inheritance Caenorhabditis_elegans Invertebrate)
(Inheritance Caenorhabditis_briggsae Invertebrate)
(Inheritance Caenorhabditis_tropicalis Invertebrate)

;; Insects / Arthropods
(Inheritance Drosophila_mojavensis Invertebrate)
(Inheritance Drosophila_melanogaster Invertebrate)
(Inheritance Drosophila_bipectinata Invertebrate)
(Inheritance Drosophila_virilis Invertebrate)
(Inheritance Drosophila_kikkawai Invertebrate)
(Inheritance Musca_domestica Invertebrate)
(Inheritance Zaprionus_paravittiger Invertebrate)
(Inheritance Aedes_aegypti Invertebrate)
(Inheritance Aedes_albopictus Invertebrate)
(Inheritance Anastrepha_ludens Invertebrate)
(Inheritance Apis_mellifera Invertebrate)
(Inheritance Tribolium_castaneum Invertebrate)
(Inheritance Acheta_domesticus Invertebrate)
(Inheritance Anopheles_stephensi Invertebrate)
(Inheritance Bombyx_mori Invertebrate)

;; Crustaceans, Rotifers, Annelids
(Inheritance Daphnia_pulex Invertebrate)
(Inheritance Ceriodaphnia_affinis Invertebrate)
(Inheritance Mytilina_brevispina Invertebrate)
(Inheritance Asplanchna_brightwelli Invertebrate)
(Inheritance Philodina_acuticornis Invertebrate)
(Inheritance Brachionus_manjavacas Invertebrate)
(Inheritance Adineta_vaga Invertebrate)
(Inheritance Aeolosoma_viride Invertebrate)

;; ==== FUNGI & PROTOZOA ====
(Inheritance Saccharomyces_cerevisiae Fungi)
(Inheritance Podospora_anserina Fungi)
(Inheritance Paramecium_tetraurelia Protozoa)