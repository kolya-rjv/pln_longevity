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
(Inheritance Danio_rerio Vertebrate)

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
(Inheritance Schizosaccharomyces_pombe Fungi)
;; ==== CAPITALISATION VARIANTS FROM THE DRUGAGE SOURCE ====
;; DrugAge's CSV is inconsistent about species capitalisation, and the ETL's
;; symbol sanitiser preserves whatever it is given, so the same organism can
;; reach the KB under two symbols. Without these three atoms, four real mouse /
;; rat / housefly rows fall through `species-tier-conf`'s default branch and are
;; scored at the INVERTEBRATE tier (0.35) instead of the vertebrate one (0.50) —
;; a mouse study down-weighted as if it were a fly. Mapping the variant to the
;; same clade is a data-quality correction, not a new claim: it asserts nothing
;; that the canonical spelling above does not already assert.
;; (Fixing the capitalisation in the ETL instead would change generated symbols
;; and break any stored reference to them.)
(Inheritance Mus_Musculus Vertebrate)
(Inheritance Rattus_Norvegicus Vertebrate)
(Inheritance Musca_Domestica Invertebrate)
