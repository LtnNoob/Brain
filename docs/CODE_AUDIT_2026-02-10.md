# Brain19 Code Audit — 2026-02-10

Durchgeführt von 4 parallelen Opus-Agenten. Jeder Agent hat jeden File in seinen Subsystemen gelesen und mit grep Aufrufe geprüft.

## Gesamtbewertung: 6.2/10

| Subsystem | Score | Status |
|-----------|-------|--------|
| ingestor | 8/10 | ✅ Solide |
| tools/cli | 8/10 | ✅ Solide |
| ltm | 7/10 | 🟡 OK |
| llm/ollama | 7/10 | 🟡 OK |
| memory/stm | 7/10 | 🟡 OK |
| epistemic | 7/10 | 🟡 OK |
| adapter | 7/10 | 🟡 OK |
| curiosity | 6/10 | 🟡 Halb fertig |
| understanding | 6/10 | 🟡 Bugs |
| integration | 6/10 | 🟡 Oberflächlich |
| importers | 6/10 | 🟡 Stubs |
| cognitive | 5/10 | 🔴 Systematisch falsch |
| snapshot | 5/10 | 🔴 Unvollständig |
| kan | 5/10 | 🔴 Architektur-Limits |
| micromodel | 5/10 | 🔴 Kern kaputt |

## CRITICAL Bugs

### BUG-C1: RelevanceMap::compute() — Target wird ignoriert
- **File:** micromodel/relevance_map.cpp:24-28
- **Problem:** model->predict(e, c) wird mit identischen e und c aufgerufen — das Target-Concept cid fließt nie in die Berechnung ein. JEDES Target-Concept bekommt den EXAKT GLEICHEN Score. Die gesamte RelevanceMap ist wertlos.
- **Code:**
```cpp
for (ConceptId cid : all_ids) {
    if (cid == source) continue;
    double score = model->predict(e, c);  // cid wird nicht benutzt!
    map.scores_[cid] = score;
}
```
- **Fix:** Target-Embedding aus EmbeddingManager holen und als c oder e nutzen. Ohne das ist die gesamte Phase-2-Pipeline funktional kaputt.

### BUG-C2: demo_integrated.cpp kompiliert nicht
- **File:** demo_integrated.cpp:91
- **Problem:** generate_json_snapshot() wird mit 3 Args aufgerufen, Signatur erwartet 4
- **Code:** `snapshot_gen.generate_json_snapshot(&brain, &curiosity, ctx);`
- **Fix:** `snapshot_gen.generate_json_snapshot(&brain, &ltm, &curiosity, ctx);`
- **Zusätzlich:** Makefile fehlen LTM-Sources in ALL_SOURCES

## HIGH Bugs

### BUG-H1: Cognitive — Recency hardcoded
- **File:** cognitive/cognitive_dynamics.cpp:232-236
- **Problem:** compute_recency_factor() returnt IMMER 0.5. Placeholder, nie implementiert. 10% jeder Salience-Berechnung ist fake.
- **Fix:** Basierend auf last_accessed_tick implementieren, oder recency_weight auf 0.0 setzen.

### BUG-H2: Cognitive — Connectivity fehlt bei Einzelaufrufen
- **File:** cognitive/cognitive_dynamics.cpp:195-197
- **Problem:** compute_salience() setzt connectivity_contrib = 0.0 mit Kommentar "Will be set in batch". Aber compute_salience() wird auch EINZELN aufgerufen (z.B. compute_path_score). 20% der Salience fehlen.
- **Fix:** Connectivity auch in der Einzel-Methode berechnen.

### BUG-H3: Understanding — Erfundene Concept-IDs
- **File:** understanding/understanding_layer.cpp:161-170
- **Problem:** perform_understanding_cycle() erzeugt Concept-IDs als 1..N basierend auf concepts_activated COUNT. Die IDs haben keinen Bezug zu den echten Konzepten.
- **Fix:** STM nach aktiven Konzepten abfragen.

### BUG-H4: RelevanceMap::combine() — Erste Map nicht gewichtet
- **File:** micromodel/relevance_map.cpp:112-118
- **Problem:** maps[0].scores_ wird 1:1 kopiert, nur maps[1..n] werden gewichtet. Bei WEIGHTED_AVERAGE hat Map 0 effektiv Gewicht 1.0 statt weights[0].
- **Fix:** Auch maps[0] mit weights[0] gewichten.

### BUG-H5: STM hash — 64→32bit Trunkierung
- **File:** memory/stm.cpp:~224
- **Problem:** hash_relation() schneidet 64-bit IDs auf 32 Bit ab. Bei IDs > 2^32 kollidieren verschiedene Relationen.
- **Fix:** Bessere Hash-Funktion oder std::pair als Key.

### BUG-H6: KAN — const_cast UB + Rand-Bug
- **File:** kan/kan_node.cpp:56-70
- **Problem 1:** const_cast in gradient() bricht const-correctness, thread-unsafe UB
- **Problem 2:** x=1.0 ergibt 0.0 wegen strict < am rechten B-Spline-Rand
- **Fix 1:** Analytische Gradienten (∂f/∂c_i = B_i(x)) statt numerischer Differenzen
- **Fix 2:** Sonderbedingung für rechten Rand

### BUG-H7: KAN — Nur 1-Schicht möglich
- **File:** kan/kan_layer.hpp
- **Problem:** KANLayer hat n_in Nodes mit je 1 Output statt n_in × n_out. Keine Multi-Layer KAN möglich.
- **Fix:** KANLayer sollte input_dim * output_dim Nodes haben.

### BUG-H8: KAN-Adapter — Dangling Pointer
- **File:** adapter/kan_adapter.cpp:32-37
- **Problem:** shared_ptr mit no-op Deleter auf unique_ptr-verwalteten Speicher. Use-after-free wenn FunctionHypothesis den KANAdapter überlebt.
- **Fix:** shared_from_this oder KANModule in shared_ptr verwalten.

### BUG-H9: Epistemic — Placement-New Hack
- **File:** ltm/long_term_memory.hpp:48-57
- **Problem:** ConceptInfo::operator= zerstört EpistemicMetadata und baut per placement new neu. UB-adjacent, bricht wenn EpistemicMetadata nicht-trivial wird.
- **Fix:** Assignment erlauben oder ConceptInfo immutable machen.

### BUG-H10: Cognitive — INVALIDATED Activation Leak
- **File:** cognitive/cognitive_dynamics.cpp:87-90
- **Problem:** INVALIDATED Concepts werden in STM aktiviert BEVOR der Check greift.
- **Fix:** INVALIDATED-Check VOR stm.activate_concept().

### BUG-H11: Snapshot — 50% fehlt
- **File:** snapshot_generator.cpp
- **Problem:** Relations immer leer, KAN/MicroModels/Ingestor/CognitiveDynamics fehlen.
- **Fix:** Alle Subsysteme serialisieren.

### BUG-H12: Ingestor — Doppeltes Chunking
- **File:** ingestor/ingestion_pipeline.cpp:96-101
- **Problem:** chunk_text() wird zweimal aufgerufen — einmal für Processing, einmal für Count.
- **Fix:** Einmal chunken, Count aus Ergebnis nehmen.

## MEDIUM Bugs

### BUG-M1: Epistemic assert(false) crash
- **File:** epistemic/epistemic_metadata.hpp:73-78
- **Problem:** assert(false) bei INVALIDATED + trust >= 0.2 killt in Debug statt zu warnen.

### BUG-M2: Keine State-Transition-Validierung
- **File:** epistemic/epistemic_metadata.hpp
- **Problem:** INVALIDATED → ACTIVE möglich, keine Prüfung.

### BUG-M3: Cognitive — Statistik-Inflation
- **File:** cognitive/cognitive_dynamics.cpp:463-464
- **Problem:** compute_path_score() inflated Salience-Statistiken unkontrolliert.

### BUG-M4: Cognitive — Source-Reihenfolge beeinflusst Ergebnis
- **File:** cognitive/cognitive_dynamics.cpp:102-115
- **Problem:** spread_activation_multi() teilt globales visited-Set. Reihenfolge = unterschiedliche Ergebnisse.

### BUG-M5: Cognitive — Focus-Flag irreführend
- **File:** cognitive/cognitive_dynamics.cpp:358
- **Problem:** enable_focus_decay deaktiviert gesamtes Focus-Feature, nicht nur Decay.

### BUG-M6: Curiosity — 50% Trigger nicht implementiert
- **File:** curiosity/curiosity_trigger.hpp:14-15
- **Problem:** MISSING_DEPTH und RECURRENT_WITHOUT_FUNCTION definiert aber nie erzeugt.

### BUG-M7: Curiosity — Leere Concept-IDs in Triggers
- **File:** curiosity/curiosity_engine.cpp:25-27
- **Problem:** SHALLOW_RELATIONS Trigger hat leere related_concept_ids.

### BUG-M8: MicroModel — Kein NaN/Inf-Schutz
- **File:** micromodel/micro_model.cpp
- **Problem:** Keine isnan/isinf Checks. z kann bei großen Weights overflow produzieren.

### BUG-M9: MicroModel — Asymmetrische Embedding-Initialisierung
- **File:** micromodel/embedding_manager.cpp:71-77
- **Problem:** Bereich [-0.1, +0.09998] statt [-0.1, +0.1].

### BUG-M10: Overlay WEIGHTED_AVERAGE inkonsistent
- **File:** micromodel/relevance_map.cpp:76-86
- **Problem:** Neue Entries bekommen score*weight, bestehende bleiben unverändert.

### BUG-M11: Null Thread-Safety
- **File:** Alle Subsysteme
- **Problem:** Kein Mutex, kein atomic. Jeder concurrent Zugriff = Data Race UB.

### BUG-M12: LLM — POST statt GET für /api/tags
- **File:** llm/ollama_client.cpp:42

### BUG-M13: LLM — curl_global_init nicht thread-safe
- **File:** llm/ollama_client.cpp:17-19

### BUG-M14: LLM — Kein Retry
- **File:** llm/ollama_client.hpp

### BUG-M15: Understanding — SpecializedMiniLLM nicht implementiert
- **File:** understanding/mini_llm_factory.hpp

### BUG-M16: KAN — Quadratisch langsames Training
- **File:** kan/kan_module.cpp:87-114
- **Problem:** Numerische Gradienten = O(n_data * n_coefs * n_data) pro Iteration.

### BUG-M17: Ingestor — Fragiler JSON-Parser
- **File:** ingestor/knowledge_ingestor.cpp

### BUG-M18: Demo — Separates LTM pro Demo
- **File:** demo_epistemic_complete.cpp:83+
- **Problem:** Jede Demo erstellt isoliertes LTM statt shared.

## Architektonische Stärken
- Epistemische Integrität konsequent durchgesetzt (ConceptInfo() = delete)
- Proposal-Queue-Pattern verhindert unkontrollierte LTM-Writes
- Ingestor-Pipeline gut durchdacht
- CLI-Tool verbindet Subsysteme sauber
- Brain19-Philosophie ("Mechanik statt Magie") konsistent umgesetzt

## Empfohlene Prioritäten
1. **SOFORT:** BUG-C1 (RelevanceMap) — ohne das ist MicroModel-Inference nutzlos
2. **SOFORT:** BUG-H1+H2 (Salience) — verfälscht alle kognitiven Berechnungen
3. **SOFORT:** BUG-H3 (Understanding IDs) — Understanding Cycle arbeitet mit falschen Daten
4. **BALD:** BUG-H6+H7 (KAN) — Architektur-Limits verhindern echtes KAN-Netzwerk
5. **BALD:** BUG-C2 + Makefile — Build-System fixen
6. **MITTELFRISTIG:** Thread-Safety (BUG-M11) — nötig für Multi-Stream
