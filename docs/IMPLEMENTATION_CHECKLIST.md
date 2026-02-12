# IMPLEMENTATION_PLAN.md vs. Code — Vollstaendiger Abgleich

**Datum:** 2026-02-12
**Methode:** Jeder Punkt des INTEGRATION_PLAN.md gegen den tatsaechlichen Code geprueft
**Code-Stand:** Nach Audit Iteration 3 (10/10)

---

## Zusammenfassung

| Status | Anzahl |
|--------|--------|
| Implementiert | 8 |
| Teilweise implementiert | 2 |
| Nicht implementiert | 5 |
| Bewusst nicht gebaut (Plan sagt "nicht bauen") | 1 |

---

## 1. ConceptInfo erweitert

**Status:** ✅ IMPLEMENTIERT

| Feld | Datei:Zeile | Default |
|------|-------------|---------|
| `activation_score` | `ltm/long_term_memory.hpp:30` | `= 0.0` |
| `salience_score` | `ltm/long_term_memory.hpp:31` | `= 0.0` |
| `structural_confidence` | `ltm/long_term_memory.hpp:32` | `= 0.0` |
| `semantic_confidence` | `ltm/long_term_memory.hpp:33` | `= 0.0` |

Persistence: Felder in `PersistentConceptRecord` (`persistent/persistent_records.hpp:48-51`) aus `_reserved` Bytes geschnitten. Lese-Mapping in `persistent/persistent_ltm.cpp:143-146`. Schreib-Mapping fehlt (Felder sind Laufzeitwerte, nicht persistent — konform mit Plan-Empfehlung "Neue Felder NICHT ins WAL schreiben").

---

## 2. RelationInfo erweitert

**Status:** ✅ IMPLEMENTIERT

| Feld | Datei:Zeile | Default |
|------|-------------|---------|
| `dynamic_weight` | `ltm/relation.hpp:37` | `= 0.0` |
| `inhibition_factor` | `ltm/relation.hpp:38` | `= 0.0` |
| `structural_strength` | `ltm/relation.hpp:39` | `= 0.0` |

Persistence: Felder in `PersistentRelationRecord` (`persistent/persistent_records.hpp:74-76`). Lese-Mapping in `persistent/persistent_ltm.cpp:287-289`.

---

## 3. GoalState Klasse komplett

**Status:** ✅ IMPLEMENTIERT

**Datei:** `cursor/goal_state.hpp`

| Feature | Zeile | Status |
|---------|-------|--------|
| `GoalType` enum (6 Typen) | `:12-19` | DEFINITION, CAUSAL_CHAIN, COMPARISON, PROPERTY_QUERY, EXPLORATION, CUSTOM |
| `target_concepts` | `:25` | `vector<ConceptId>` |
| `completion_metric` | `:27` | `double = 0.0` |
| `threshold` | `:28` | `double = 0.8` |
| `priority_weight` | `:29` | `double = 1.0` |
| `query_embedding` | `:26` | `Vec10{}` (Plan hatte das nicht, aber sinnvoll) |
| `query_text` | `:30` | `string` (fuer Template-Engine) |
| `is_complete()` | `:33-35` | `completion_metric >= threshold` |
| `update_progress()` | `:38-60` | Zwei Modi: target-based und chain-length-based |
| Factory: `definition_goal()` | `:63-71` | ✅ |
| Factory: `causal_goal()` | `:76-84` | ✅ (korrigiert: leere targets, kein seeds-Arg) |
| Factory: `exploration_goal()` | `:87-95` | ✅ |

**Abweichung vom Plan:** Plan definierte `GoalType` als `REACH_CONCEPT, ANSWER_QUERY, EXPLORE_REGION, VALIDATE_CLAIM`. Implementierung nutzt spezifischere Typen: `DEFINITION, CAUSAL_CHAIN, COMPARISON, PROPERTY_QUERY, EXPLORATION, CUSTOM`. Die implementierten Typen sind reichhaltiger und passen besser zur Template-Engine.

---

## 4. FocusCursor mit step(), deepen(), branch(), backtrack(), shift_focus()

**Status:** ✅ IMPLEMENTIERT

**Dateien:** `cursor/focus_cursor.hpp`, `cursor/focus_cursor.cpp`

| Methode | hpp:Zeile | cpp:Zeile | Status |
|---------|-----------|-----------|--------|
| `seed(ConceptId)` | `:38` | `:22-25` | ✅ |
| `seed(ConceptId, Vec10)` | `:39` | `:27-47` | ✅ |
| `step()` | `:43` | `:140-193` | ✅ Termination + goal update |
| `step_to(ConceptId)` | `:46` | `:195-253` | ✅ Mit termination + goal (Audit-Fix) |
| `backtrack()` | `:49` | `:255-267` | ✅ |
| `deepen()` | `:52` | `:270-279` | ✅ |
| `shift_focus(RelationType)` | `:55` | `:281-283` | ✅ |
| `branch(size_t k)` | `:73` | `:296-321` | ✅ Kopiert alle State inkl. preferred_relation_ |
| `get_view()` | `:58` | `:285-294` | ✅ |
| `result()` | `:76` | `:323-346` | ✅ chain_score excludes seed |
| `set_goal(GoalState)` | `:79` | inline | ✅ |
| `evaluate_edge()` | `:105` | `:49-66` | ✅ MicroModel::predict() |
| `accumulate_context()` | `:108` | `:68-76` | ✅ |
| `check_termination()` | `:111` | `:125-138` | ✅ depth, energy, goal |
| `get_candidates()` | `:120` | `:78-123` | ✅ outgoing + incoming, sorted |

**Abweichung vom Plan:** Plan spezifizierte `compute_weight()` mit ConceptEmbeddingStore. Implementierung nutzt `evaluate_edge()` mit `embeddings_.make_target_embedding()` statt concept embeddings — funktional aequivalent, braucht keinen separaten ConceptEmbeddingStore. Plan spezifizierte `ExplorationMode::EXPLORATORY` mit Randomisierung — implementiert als Enum-Wert, aber step() nutzt immer Greedy-Selektion.

---

## 5. FocusCursorManager mit process_seeds(), persist_to_stm()

**Status:** ✅ IMPLEMENTIERT

**Dateien:** `cursor/focus_cursor_manager.hpp`, `cursor/focus_cursor_manager.cpp`

| Methode | Zeile | Status |
|---------|-------|--------|
| `process_seeds(seeds, query_context)` | `hpp:32-35` | ✅ Delegates to goal-overload |
| `process_seeds(seeds, query_context, goal)` | `hpp:38-42` | ✅ Creates cursor per seed, runs deepen(), selects best |
| `persist_to_stm(ctx, chain)` | `hpp:45` | ✅ Activates concepts + relations in STM |

---

## 6. Termination Logic (check_termination)

**Status:** ✅ IMPLEMENTIERT

**Dateien:**
- In FocusCursor: `cursor/focus_cursor.cpp:125-138` — integriert in step()/step_to()
- Standalone: `cursor/termination.hpp:22-37` — check_termination(goal, view, config)

| Bedingung | FocusCursor | Standalone |
|-----------|-------------|------------|
| Max depth | ✅ `:127` | ✅ `:28` |
| Energy budget | ✅ `:130` | ✅ `:31` |
| Goal completion | ✅ `:133` | ✅ `:34` |
| No candidates / dead end | ✅ `:151-153` | ❌ (documented) |
| Below min_weight_threshold | ✅ `:157-159` | ❌ (documented) |

---

## 7. Conflict Resolution (effective_priority mit alpha/beta/gamma)

**Status:** ✅ IMPLEMENTIERT

**Datei:** `cursor/conflict_resolution.hpp`

| Feature | Zeile | Status |
|---------|-------|--------|
| `ConflictWeights{alpha=0.4, beta=0.4, gamma=0.2}` | `:23-27` | ✅ Exakt wie im Plan |
| `effective_priority(ConceptInfo, ConflictWeights)` | `:29-33` | ✅ `alpha*structural + beta*semantic + gamma*activation` |
| `resolves_in_favor(a, b, weights)` | `:36-39` | ✅ |

---

## 8. Template-Engine (alle RelationTypes -> Satzmuster)

**Status:** ✅ IMPLEMENTIERT

**Dateien:** `cursor/template_engine.hpp`, `cursor/template_engine.cpp`

| RelationType | Deutsches Satzmuster | Zeile |
|-------------|---------------------|-------|
| IS_A | "ist ein(e)" | `cpp:20` |
| HAS_PROPERTY | "hat die Eigenschaft" | `cpp:21` |
| CAUSES | "verursacht" | `cpp:22` |
| ENABLES | "ermoeglicht" | `cpp:23` |
| PART_OF | "ist Teil von" | `cpp:24` |
| SIMILAR_TO | "ist aehnlich wie" | `cpp:25` |
| CONTRADICTS | "widerspricht" | `cpp:26` |
| SUPPORTS | "unterstuetzt" | `cpp:27` |
| TEMPORAL_BEFORE | "geschieht vor" | `cpp:28` |
| CUSTOM | "steht in Beziehung zu" | `cpp:29` |

Alle 10 bestehenden RelationTypes abgedeckt. TemplateType-Klassifikation (KAUSAL_ERKLAEREND, DEFINITIONAL, AUFZAEHLEND, VERGLEICHEND) vorhanden.

**Abweichung vom Plan:** Plan definierte Template-Engine als Teil eines KAN-basierten SemanticScorers. Implementierung ist regelbasiert (keine KAN-Module). Das ist konform mit der Entscheidung "Template-Engine from Day 1" (Ollama-Ersatz), waehrend die KAN-basierte Version fuer Phase 9 (Language Engine) vorgesehen ist.

---

## 9. UNKNOWN als Output-State

**Status:** ⚠️ TEILWEISE IMPLEMENTIERT

`TriggerType::UNKNOWN` existiert in `curiosity/curiosity_trigger.hpp:17` als bestehender Enum-Wert (war vor der Implementation schon da). Es ist KEIN neuer Output-State fuer das System.

Der Plan spezifizierte "UNKNOWN" als moeglichen CuriosityTrigger-Typ. Dieser existiert. Aber der Plan forderte auch neue TriggerTypes:

| TriggerType | Status | Datei:Zeile |
|-------------|--------|-------------|
| `UNKNOWN` | ✅ Existiert (pre-existing) | `curiosity/curiosity_trigger.hpp:17` |
| `LOW_CONFIDENCE` | ❌ Fehlt | — |
| `CONTRADICTION` | ❌ Fehlt | — |
| `HIGH_NOVELTY` | ❌ Fehlt | — |

Auch die Plan-Erweiterung `CuriosityTrigger::priority` (double, default 0.0) fehlt.

---

## 10. Ollama komplett entfernt

**Status:** ✅ IMPLEMENTIERT

| Geloeschte Datei | Status |
|-----------------|--------|
| `llm/ollama_client.hpp` | ✅ Geloescht |
| `llm/ollama_client.cpp` | ✅ Geloescht |
| `understanding/ollama_mini_llm.hpp` | ✅ Geloescht |
| `understanding/ollama_mini_llm.cpp` | ✅ Geloescht |
| `Makefile.ollama` | ✅ Geloescht |

| Bereinigte Datei | Aenderung |
|-----------------|-----------|
| `llm/chat_interface.hpp` | OllamaClient-Referenzen entfernt |
| `llm/chat_interface.cpp` | Nur Knowledge-only Pfade |
| `core/system_orchestrator.hpp` | Ollama-Config + Include entfernt |
| `core/system_orchestrator.cpp` | Ollama-Init-Block entfernt |
| `main.cpp` | `--ollama-host/model` Optionen entfernt |
| `demo_chat.cpp` | OllamaConfig-Block ersetzt |
| `understanding/mini_llm_factory.hpp` | OllamaConfig entfernt |
| `Makefile` | Ollama-Sources entfernt |

`ChatInterface::is_llm_available()` gibt konstant `false` zurueck. Build: 0 Errors.

---

## 11. Brain19ControlLoop (run-Methode mit allen Schritten)

**Status:** ⚠️ BEWUSST NICHT ALS EIGENE KLASSE — In ThinkingPipeline integriert

Der Plan sagt explizit auf Zeile 2314:
> `Brain19ControlLoop` als eigene Klasse — stattdessen ThinkingPipeline erweitern (Step 8.1). Verhindert eine zweite Orchestrierungsschicht neben SystemOrchestrator.

Die Control-Loop-Logik ist stattdessen in `ThinkingPipeline::execute()` und `execute_with_goal()` implementiert:

| Pipeline-Step | Datei:Zeile | Status |
|---------------|-------------|--------|
| Step 1: Activate seeds in STM | `thinking_pipeline.cpp:38` | ✅ |
| Step 2: Spreading Activation | `thinking_pipeline.cpp:42` | ✅ |
| Step 2.5: FocusCursor traversal | `thinking_pipeline.cpp:46-53` | ✅ |
| Step 3: Salience + Focus | `thinking_pipeline.cpp:59-64` | ✅ |
| Step 4-5: RelevanceMaps | `thinking_pipeline.cpp:67-68` | ✅ |
| Step 6: ThoughtPaths | `thinking_pipeline.cpp:71` | ✅ |
| Step 7: CuriosityEngine | `thinking_pipeline.cpp:75-78` | ✅ |
| Step 8: UnderstandingLayer | `thinking_pipeline.cpp:81-89` | ✅ |
| Step 9: KAN-LLM Validation | `thinking_pipeline.cpp:93-98` | ✅ |
| Step 10: Complete | `thinking_pipeline.cpp:101` | ✅ |

`execute_with_goal()` (`thinking_pipeline.cpp:247-337`) fuehrt den gleichen Loop mit explizitem GoalState durch.

`SystemOrchestrator::run_thinking_cycle()` hat zwei Overloads:
- `run_thinking_cycle(seeds)` → `execute()` — `system_orchestrator.cpp:555`
- `run_thinking_cycle(seeds, goal)` → `execute_with_goal()` — `system_orchestrator.cpp:575`

---

## 12. CuriosityEngine Integration mit GoalState

**Status:** ❌ NICHT IMPLEMENTIERT

Der Plan (Phase 5, Step 5.1-5.2) forderte:
- Neue TriggerTypes: `LOW_CONFIDENCE`, `CONTRADICTION`, `HIGH_NOVELTY` → ❌ Fehlt
- `CuriosityTrigger::priority` Feld → ❌ Fehlt
- `CuriosityEngine::analyze_confidence_gaps()` → ❌ Fehlt
- `CuriosityEngine::enqueue_trigger()` → ❌ Fehlt
- `CuriosityEngine::get_pending_triggers()` → ❌ Fehlt
- GoalState-Suspension bei high-priority Curiosity Trigger → ❌ Fehlt

Die bestehende CuriosityEngine (`curiosity/curiosity_engine.hpp`) ist unveraendert. Sie wird im ThinkingPipeline Step 7 aufgerufen (`thinking_pipeline.cpp:75-78`), aber ohne GoalState-Integration.

---

## 13. Global Dynamics Operator

**Status:** ❌ NICHT IMPLEMENTIERT

Der Plan (Phase 7, Step 7.1) forderte:
- `cognitive/global_dynamics_operator.hpp` → Datei existiert nicht
- `GlobalDynamicsOperator` Klasse mit:
  - `update_activation_field(ctx)` → ❌
  - `apply_inhibition(ctx)` → ❌
  - `decay_all(ctx, dt)` → ❌
  - `sync_activation_scores(ctx)` → ❌

Die Funktionalitaet wird teilweise durch die bestehende CognitiveDynamics abgedeckt (spread_activation, focus_on, decay), aber der Wrapper-Layer mit Inhibition und Activation-Score-Synchronisation fehlt.

---

## 14. Dual-Mode Integration (Global + Focus zusammen)

**Status:** ❌ NICHT IMPLEMENTIERT

Der Plan (State Diagram 4.2, Sequence Diagram 2.1) beschreibt:
1. GlobalDynamicsOperator fuer Background-Aktivierung → ❌ GDO fehlt
2. FocusCursor fuer gezielte Traversal → ✅ Implementiert
3. KANTraversalPolicy fuer dynamische Steuerung → ❌ Fehlt
4. Goal-Suspension bei Curiosity-Triggers → ❌ Fehlt

`ThinkingPipeline::execute()` hat ein lineares Pipeline-Modell:
- Spreading → FocusCursor → Salience → Relevance → Curiosity → Understanding
- Es gibt keine parallele Global+Focus Interaktion
- Es gibt keine dynamische KANTraversalPolicy-Steuerung waehrend der Traversal

**Was fehlt konkret:**
| Komponente | Plan-Referenz | Status |
|------------|---------------|--------|
| `GlobalDynamicsOperator` | Phase 7 | ❌ |
| `KANTraversalPolicy` | Phase 6 | ❌ |
| `ConceptEmbeddingStore` | Phase 0, Step 0.1 | ❌ |
| Parallele Global+Focus Execution | Sequence Diagram 2.1 | ❌ |
| Goal-aware Curiosity Suspension | State Diagram 4.1 | ❌ |

---

## 15. Unit Tests fuer alle neuen Module

**Status:** ✅ IMPLEMENTIERT (fuer alle implementierten Module)

| Test-Datei | Tests | Status |
|------------|-------|--------|
| `cursor/test_focus_cursor.cpp` | 9 Tests: GoalState, seed, step, deepen, backtrack, max_depth, manager, persist, cycles | ✅ ALL PASS |
| `cursor/test_termination_conflict.cpp` | 8 Tests: depth, energy, goal, no-trigger, priority, resolves, zero, custom weights | ✅ ALL PASS |
| `cursor/test_template_engine.cpp` | 9 Tests: CAUSES, all types, chain, classify, TraversalResult, single, empty, definitional, relation_name_de | ✅ ALL PASS |
| `core/test_pipeline_cursor.cpp` | 4 Tests: cursor enabled, disabled, execute_with_goal, pipeline+template | ✅ ALL PASS |

Keine Tests fuer: GlobalDynamicsOperator, KANTraversalPolicy, CuriosityEngine-Erweiterung (da nicht implementiert).

---

## Nicht im Plan, aber implementiert

| Feature | Datei | Bemerkung |
|---------|-------|-----------|
| `TemplateEngine` (regelbasiert) | `cursor/template_engine.hpp/cpp` | Plan sah KAN-basierten SemanticScorer vor. Regelbasierte Version als Day-1-Loesung |
| `ThinkingPipeline::Config::thinking_config` | `system_orchestrator.hpp:69` | Audit-Fix: Config durchreichbar |

---

## Gesamte Language Engine (Phase 9) — NICHT IMPLEMENTIERT

Der Plan definiert eine komplette KAN-basierte Language Engine mit 8 neuen Klassen:

| Klasse | Plan-Datei | Status |
|--------|-----------|--------|
| `BPETokenizer` | `hybrid/tokenizer.hpp` | ❌ |
| `KANEncoder` | `hybrid/kan_encoder.hpp` | ❌ |
| `KANDecoder` | `hybrid/kan_decoder.hpp` | ❌ |
| `SemanticScorer` (KAN-basiert) | `hybrid/semantic_scorer.hpp` | ❌ |
| `FusionLayer` | `hybrid/fusion_layer.hpp` | ❌ |
| `KANLanguageEngine` | `hybrid/kan_language_engine.hpp` | ❌ |
| `LanguageConfig` | `hybrid/language_config.hpp` | ❌ |
| `LanguageTrainer` | `hybrid/language_training.hpp` | ❌ |

Diese Phase haengt von den Cursor-Grundlagen ab, die jetzt fertig sind. Die regelbasierte Template-Engine dient als Ueberbrueckung.

---

## Weitere fehlende Vorbereitungen (Phase 0)

| Feature | Plan-Step | Status |
|---------|-----------|--------|
| `ConceptEmbeddingStore` / `get_concept_embedding()` in EmbeddingManager | Step 0.1 | ❌ |
| Neue RelationTypes (PRODUCES, REQUIRES, USES, SOURCE_OF, HAS_PART) | Step 0.2 | ❌ |

---

## Fazit

**Implementierte Phasen:** 1, 2 (Persistence), 3, 4, 8 (teilweise), Ollama-Removal
**Fehlende Phasen:** 0 (Vorbereitungen), 5 (Curiosity-Erweiterung), 6 (KANTraversalPolicy), 7 (GlobalDynamicsOperator), 9 (Language Engine)

Der Kern des FocusCursor-Systems (Phasen 3-4, 8) ist vollstaendig und getestet. Die hoeher-level Integrationsschichten (GDO, KANPolicy, Curiosity-GoalState) und die komplette Language Engine stehen noch aus. Die regelbasierte Template-Engine ueberbrueckt die Language-Engine-Luecke funktional.
