# Brain19 System Integration Plan

> Generated: 2025-02-10  
> Status: Analysis complete, ready for Fix-Agent

---

## 1. Initialization Order Analysis

### Current 14-Stage Init

| Stage | Component | Dependencies | Status |
|-------|-----------|-------------|--------|
| 1 | LTM (in-memory) | none | ✅ OK |
| 2 | Persistence (dir only) | LTM | ⚠️ PersistentLTM unused, WAL unused |
| 3 | BrainController + STM | none | ✅ OK |
| 4 | MicroModels (Registry, Embeddings, Trainer) | none | ✅ OK |
| 5 | CognitiveDynamics | none | ✅ OK |
| 6 | CuriosityEngine | none | ✅ OK |
| 7 | KANAdapter | none | ✅ OK |
| 8 | UnderstandingLayer + MiniLLMs | Ollama (ext) | ✅ OK |
| 9 | KAN-LLM Hybrid (Validator, DomainMgr, RefinementLoop) | KanValidator | ✅ OK |
| 10 | IngestionPipeline + WikiImporter | LTM | ✅ OK |
| 11 | ChatInterface + OllamaClient | Ollama (ext) | ✅ OK |
| 12 | Shared Wrappers (SharedLTM/STM/Registry/Embeddings) | LTM, STM, Registry, Embeddings | ✅ OK |
| 13 | Streams (Orchestrator, Scheduler, Monitor) | Shared Wrappers | ✅ OK |
| 14 | Bootstrap Foundation Seeding | LTM | ✅ OK |
| post | ThinkingPipeline, active context, ensure_models_for, start streams | All | ⚠️ See issues |

### Missing Subsystems

| Subsystem | Files Exist | In Orchestrator | Priority |
|-----------|-------------|-----------------|----------|
| **PatternDiscovery** | `evolution/pattern_discovery.cpp` | ❌ MISSING | HIGH |
| **EpistemicPromotion** | `evolution/epistemic_promotion.cpp` | ❌ MISSING | HIGH |
| **ConceptProposer** | `evolution/concept_proposal.cpp` | ❌ MISSING | HIGH |
| **WAL** | `persistent/wal.cpp` | ❌ NOT USED (Stage 2 is empty) | MEDIUM |
| **ScholarImporter** | `importers/scholar_importer.cpp` | ❌ MISSING | LOW |
| **SnapshotGenerator** | `snapshot_generator.cpp` | ❌ MISSING | LOW |

### Dependency Order Issues

1. **No circular dependencies detected.** All subsystems have clean forward dependencies.
2. **Stage 2 is a no-op.** WAL + PersistentLTM are constructed nowhere. Stage 2 just creates the directory.
3. **ThinkingPipeline created AFTER Stage 14** — not in the stage counter, so `cleanup_from_stage()` doesn't clean it up.
4. **`ensure_models_for()` after init** — iterates ALL concepts, creates MicroModel for each. O(N) at startup, acceptable. But also called after every `ingest_text()` — see §4.

### Recommended New Init Order (17 Stages)

```
 1. LTM
 2. Persistence (WAL + PersistentLTM — actually wire them)
 3. BrainController + STM
 4. MicroModels (Registry, Embeddings, Trainer)
 5. CognitiveDynamics
 6. CuriosityEngine
 7. KANAdapter
 8. UnderstandingLayer + MiniLLMs
 9. KAN-LLM Hybrid
10. IngestionPipeline + Importers
11. ChatInterface + OllamaClient
12. Evolution (PatternDiscovery, EpistemicPromotion, ConceptProposer) ← NEW
13. Shared Wrappers
14. Streams
15. ThinkingPipeline ← MOVE INTO stages
16. Bootstrap Foundation
17. Post-init (active context, ensure_models, start streams)
```

---

## 2. Runtime Flows

### 2.1 Chat Flow

```
User question
  │
  ├─ SystemOrchestrator::ask(question)
  │    │
  │    ├─ NAIVE SEED SEARCH: iterate ALL concepts, string::find(label) ← BUG: O(N), substring match
  │    │   → seeds (max 5 ConceptIds)
  │    │
  │    ├─ run_thinking_cycle(seeds)
  │    │   └─ ThinkingPipeline::execute(...)
  │    │       ├─ Step 1: activate seeds in STM via BrainController
  │    │       ├─ Step 2: CognitiveDynamics::spread_activation_multi()
  │    │       ├─ Step 3: CognitiveDynamics::get_top_k_salient() + focus_on()
  │    │       ├─ Step 4-5: RelevanceMap::compute() per salient concept (MicroModels)
  │    │       ├─ Step 6: CognitiveDynamics::find_best_paths()
  │    │       ├─ Step 7: CuriosityEngine::observe_and_generate_triggers()
  │    │       ├─ Step 8: UnderstandingLayer::perform_understanding_cycle()
  │    │       ├─ Step 9: KanValidator::validate() per hypothesis
  │    │       └─ Return ThinkingResult
  │    │
  │    ├─ ChatInterface::ask(question, ltm)
  │    │   ├─ find_relevant_concepts() ← ANOTHER naive search, independent of thinking
  │    │   ├─ build_epistemic_context() → system prompt with concept data
  │    │   └─ OllamaClient::chat() → LLM response
  │    │
  │    └─ Return ChatResponse
```

**Problem:** ThinkingResult is computed but NEVER passed to ChatInterface. The thinking cycle runs, activates STM, but ChatInterface does its own independent concept search. The ThinkingPipeline's output (salient concepts, paths, understanding) is wasted.

### 2.2 Ingestion Flow

```
Text input
  │
  ├─ SystemOrchestrator::ingest_text(text)
  │    ├─ IngestionPipeline::ingest_text(text, source, auto_approve)
  │    │   ├─ TextChunker::chunk() → chunks
  │    │   ├─ EntityExtractor::extract() → entities per chunk
  │    │   ├─ RelationExtractor::extract() → relations per chunk
  │    │   ├─ TrustTagger::tag() → epistemic metadata
  │    │   ├─ ProposalQueue::enqueue() → proposals
  │    │   ├─ if auto_approve: approve all
  │    │   └─ KnowledgeIngestor::ingest() → store in LTM
  │    │
  │    └─ MicroModelRegistry::ensure_models_for(ltm) ← FULL SCAN every time
  │
  └─ Return IngestionResult
```

**Problem:** `ensure_models_for(*ltm_)` after each ingest scans ALL concepts, not just new ones. Should use `result.stored_concept_ids` to create models only for new concepts.

### 2.3 Thinking Flow (ThinkingPipeline)

```
Seeds → Step 1-10 (see Chat Flow above)
```

**Well-designed.** 10-step pipeline, clean separation. Each step receives only what it needs. Main issues:
- Step 8 (Understanding) only processes `salient_ids[0]` — first concept only
- Step 9 (KAN Validation) depends on understanding producing hypotheses — if understanding is stub, no validation happens
- Result is returned but caller (ask()) ignores most of it

### 2.4 Evolution Flow — NOT IMPLEMENTED

```
SHOULD BE:
  CuriosityEngine::triggers
    │
    ├─ ConceptProposer::from_curiosity(triggers) → proposals
    ├─ ConceptProposer::from_relevance_anomalies(map) → proposals
    ├─ ConceptProposer::from_analogies(analogies) → proposals
    │
    ├─ rank_proposals() → top K
    │
    ├─ Store in LTM as SPECULATION/HYPOTHESIS (trust ≤ 0.5)
    │
    ├─ EpistemicPromotion::run_maintenance()
    │   ├─ SPECULATION → HYPOTHESIS (auto, if supported)
    │   ├─ HYPOTHESIS → THEORY (auto, if validated)
    │   └─ THEORY → FACT (HUMAN REVIEW ONLY)
    │
    └─ PatternDiscovery::discover_all()
        ├─ find_clusters()
        ├─ find_hierarchies()
        ├─ find_bridges()
        ├─ find_cycles() → possible contradictions
        └─ find_gaps() → curiosity targets
```

**Status:** All three classes exist and compile. NONE are instantiated in SystemOrchestrator. No member variables, no init stage, no runtime integration.

### 2.5 Stream Flow

```
StreamScheduler::start()
  └─ creates N ThinkStreams (auto_scale)
      │
      ├─ ThinkStream::run() loop:
      │   ├─ tick()
      │   │   ├─ do_spreading()  → reads SharedLTM, writes SharedSTM
      │   │   ├─ do_salience()   → reads SharedSTM
      │   │   ├─ do_curiosity()  → reads SharedSTM
      │   │   └─ do_understanding() → (likely stub)
      │   └─ backoff if idle
      │
      └─ Each stream has its own ContextId
```

**Problem:** ThinkStreams do their own mini-thinking-cycle (spreading, salience, curiosity) but:
- They DON'T use ThinkingPipeline
- They DON'T use CognitiveDynamics (they access SharedLTM/STM directly)
- They DON'T produce ThinkingResults
- They DON'T feed back into the main system (no output channel)
- Essentially they spin up threads that activate concepts in STM with no consumer

---

## 3. Missing Integration Points

### 3.1 Evolution — NOT IN SystemOrchestrator

**Must add:**
```cpp
// Members:
std::unique_ptr<PatternDiscovery> pattern_discovery_;
std::unique_ptr<EpistemicPromotion> epistemic_promotion_;
std::unique_ptr<ConceptProposer> concept_proposer_;

// Init (new Stage 12):
pattern_discovery_ = std::make_unique<PatternDiscovery>(*ltm_);
epistemic_promotion_ = std::make_unique<EpistemicPromotion>(*ltm_);
concept_proposer_ = std::make_unique<ConceptProposer>(*ltm_);
```

**Runtime integration:**
- After each ThinkingCycle: feed curiosity triggers → ConceptProposer
- Periodic: EpistemicPromotion::run_maintenance()
- Periodic: PatternDiscovery::discover_all() → feed gaps to CuriosityEngine

### 3.2 Bootstrap — In Orchestrator, Correct

Foundation seeding at Stage 14 is fine. Only seeds if LTM is empty. Uses `FoundationConcepts::seed_all()`.

### 3.3 Checkpoint — Incomplete

Current checkpoint saves:
- ✅ STM state
- ✅ MicroModelRegistry
- ❌ LTM content (in-memory, lost on restart!)
- ❌ KAN Module weights (passed as nullptr)
- ❌ Cognitive state (passed as nullptr)
- ❌ Config (passed as nullptr)

**Critical:** Without LTM persistence, ALL learned knowledge is lost on restart. PersistentLTM exists but is never used.

### 3.4 Streams — Running but Disconnected

Streams start and tick, but:
- No output channel (results vanish)
- No connection to ThinkingPipeline
- No connection to Evolution subsystem
- No work distribution from main ask() flow

### 3.5 KAN-LLM Hybrid — Correctly Wired in ThinkingPipeline

KanValidator is called in Step 9 of ThinkingPipeline, but only if UnderstandingLayer produces hypothesis_proposals. With StubMiniLLM (no Ollama), this never happens → KAN validation never runs.

---

## 4. Fehler und Lücken im aktuellen Design

### 4.1 Concept Search is Naive — O(N) × O(M) per Query

```cpp
// ask() — iterates ALL concepts, does substring match
for (auto cid : ltm_->get_all_concept_ids()) {
    auto info = ltm_->retrieve_concept(cid);
    if (info && question.find(info->label) != std::string::npos) {
```

**Fix:** Build an inverted index (label words → ConceptIds) at init and after ingest. Or use EmbeddingManager for semantic search.

### 4.2 relation_count() is O(N) — Called in Status

```cpp
size_t SystemOrchestrator::relation_count() const {
    size_t count = 0;
    for (auto cid : ltm_->get_all_concept_ids()) {
        count += ltm_->get_outgoing_relations(cid).size();
    }
    return count;
}
```

**Fix:** Cache in LTM. Increment/decrement on add/remove.

### 4.3 ensure_models_for() After Every Ingest

```cpp
auto result = ingestion_->ingest_text(text, "", auto_approve);
if (result.success) {
    registry_->ensure_models_for(*ltm_);  // FULL SCAN
}
```

**Fix:** `registry_->ensure_models_for(result.stored_concept_ids)` — only new concepts.

### 4.4 No Periodic Tasks

Missing:
- Auto-checkpoint (config says 30min interval, never used)
- EpistemicPromotion::run_maintenance() (never called)
- PatternDiscovery (never called)
- STM decay (never called outside thinking cycle)
- Stream health monitoring alerts (callback never set)

**Fix:** Add a periodic task thread or use StreamScheduler for periodic work.

### 4.5 No MicroModel Training Trigger

`MicroTrainer` is created but never called. No training data generation, no training loop.

**Fix:** After sufficient ingestion, trigger `trainer_->train()` for new concepts. Could be periodic or threshold-based.

### 4.6 ThinkingResult Wasted in ask()

`run_thinking_cycle()` returns `ThinkingResult` with salient concepts, paths, understanding — but `ask()` ignores it completely and calls `chat_->ask(question, ltm)` independently.

**Fix:** Pass ThinkingResult (or at least activated concept IDs) to ChatInterface so LLM gets the thinking context.

### 4.7 main.cpp: run_command() Without init

`Brain19App::run_command()` is called from `main()` for single commands but the orchestrator is never initialized (no `orchestrator_.initialize()` call). Only `run_interactive()` calls `initialize()`.

**Fix:** Add `initialize()` call before `run_command()` in main.

---

## 5. Build Plan

### 5.1 Files to Link (BRAIN19_SRCS in Makefile)

Current Makefile has correct BRAIN19_SRCS — 41 .cpp files covering all core, memory, micromodel, KAN, cognitive, curiosity, hybrid, ingestor, importers, LLM, persistent, streams, understanding, bootstrap, and adapter.

**Missing from BRAIN19_SRCS:**
```makefile
$(BACKEND)/evolution/pattern_discovery.cpp \
$(BACKEND)/evolution/epistemic_promotion.cpp \
$(BACKEND)/evolution/concept_proposal.cpp \
$(BACKEND)/persistent/checkpoint_manager.cpp \
$(BACKEND)/persistent/checkpoint_restore.cpp \
$(BACKEND)/persistent/stm_snapshot.cpp \
$(BACKEND)/persistent/wal.cpp \
$(BACKEND)/persistent/persistent_ltm.cpp \
$(BACKEND)/streams/stream_monitor.cpp \
$(BACKEND)/streams/stream_scheduler.cpp \
```

Some persistent files are in STREAM_SRCS but NOT in BRAIN19_SRCS. Evolution files are missing entirely.

### 5.2 External Libraries

```makefile
LDLIBS = -lcurl -lpthread
```

- **libcurl** — Required by `http_client.cpp` (Wikipedia/Scholar import) and `ollama_client.cpp`
- **pthread** — Already in CXXFLAGS via `-pthread`
- No other external deps (KAN, MicroModels, etc. are all header/source)

**Missing from Makefile:** `-lcurl` is not passed to linker. Will fail at link time.

### 5.3 Expected Compile/Link Errors

1. **Missing `-lcurl`** — undefined reference to `curl_*`
2. **Missing evolution .cpp files** in BRAIN19_SRCS — after adding evolution to orchestrator
3. **Missing persistent .cpp files** — `checkpoint_manager.cpp`, `checkpoint_restore.cpp` are used by `create_checkpoint()`/`restore_checkpoint()` but not in BRAIN19_SRCS
4. **`stream_scheduler.cpp` and `stream_monitor.cpp`** — used in orchestrator but missing from BRAIN19_SRCS
5. **`run_command()` in non-interactive mode** — commands like "ask" will crash because orchestrator not initialized

### 5.4 Updated Makefile Target

```makefile
BRAIN19_SRCS = \
	$(BACKEND)/core/system_orchestrator.cpp \
	$(BACKEND)/core/brain19_app.cpp \
	$(BACKEND)/core/thinking_pipeline.cpp \
	$(BACKEND)/bootstrap/foundation_concepts.cpp \
	$(BACKEND)/bootstrap/bootstrap_interface.cpp \
	$(BACKEND)/bootstrap/context_accumulator.cpp \
	$(BACKEND)/ltm/long_term_memory.cpp \
	$(BACKEND)/memory/stm.cpp \
	$(BACKEND)/memory/brain_controller.cpp \
	$(BACKEND)/micromodel/micro_model.cpp \
	$(BACKEND)/micromodel/micro_model_registry.cpp \
	$(BACKEND)/micromodel/embedding_manager.cpp \
	$(BACKEND)/micromodel/micro_trainer.cpp \
	$(BACKEND)/micromodel/persistence.cpp \
	$(BACKEND)/micromodel/relevance_map.cpp \
	$(BACKEND)/kan/kan_node.cpp \
	$(BACKEND)/kan/kan_layer.cpp \
	$(BACKEND)/kan/kan_module.cpp \
	$(BACKEND)/cognitive/cognitive_dynamics.cpp \
	$(BACKEND)/curiosity/curiosity_engine.cpp \
	$(BACKEND)/adapter/kan_adapter.cpp \
	$(BACKEND)/understanding/understanding_layer.cpp \
	$(BACKEND)/understanding/mini_llm.cpp \
	$(BACKEND)/understanding/ollama_mini_llm.cpp \
	$(BACKEND)/hybrid/hypothesis_translator.cpp \
	$(BACKEND)/hybrid/epistemic_bridge.cpp \
	$(BACKEND)/hybrid/kan_validator.cpp \
	$(BACKEND)/hybrid/domain_manager.cpp \
	$(BACKEND)/hybrid/refinement_loop.cpp \
	$(BACKEND)/ingestor/ingestion_pipeline.cpp \
	$(BACKEND)/ingestor/text_chunker.cpp \
	$(BACKEND)/ingestor/entity_extractor.cpp \
	$(BACKEND)/ingestor/relation_extractor.cpp \
	$(BACKEND)/ingestor/trust_tagger.cpp \
	$(BACKEND)/ingestor/proposal_queue.cpp \
	$(BACKEND)/ingestor/knowledge_ingestor.cpp \
	$(BACKEND)/importers/wikipedia_importer.cpp \
	$(BACKEND)/importers/scholar_importer.cpp \
	$(BACKEND)/importers/http_client.cpp \
	$(BACKEND)/llm/ollama_client.cpp \
	$(BACKEND)/llm/chat_interface.cpp \
	$(BACKEND)/evolution/pattern_discovery.cpp \
	$(BACKEND)/evolution/epistemic_promotion.cpp \
	$(BACKEND)/evolution/concept_proposal.cpp \
	$(BACKEND)/persistent/persistent_ltm.cpp \
	$(BACKEND)/persistent/wal.cpp \
	$(BACKEND)/persistent/stm_snapshot.cpp \
	$(BACKEND)/persistent/checkpoint_manager.cpp \
	$(BACKEND)/persistent/checkpoint_restore.cpp \
	$(BACKEND)/streams/think_stream.cpp \
	$(BACKEND)/streams/stream_orchestrator.cpp \
	$(BACKEND)/streams/stream_scheduler.cpp \
	$(BACKEND)/streams/stream_monitor.cpp

LDLIBS = -lcurl

brain19: $(BACKEND)/main.cpp $(BRAIN19_SRCS)
	$(CXX) $(CXXFLAGS) -I$(BACKEND) -o brain19 \
		$(BACKEND)/main.cpp $(BRAIN19_SRCS) $(LDLIBS)
```

---

## 6. Konkreter Fix-Plan

### Priority 1: Lauffähige Demo (must-fix for `make brain19` to work)

| # | Fix | File(s) | Effort |
|---|-----|---------|--------|
| 1.1 | Add missing .cpp files to BRAIN19_SRCS | `Makefile` | 5 min |
| 1.2 | Add `-lcurl` to linker | `Makefile` | 1 min |
| 1.3 | Add `initialize()` before `run_command()` in main | `main.cpp` | 5 min |
| 1.4 | Fix any compile errors from missing headers/types | various | 30 min |

### Priority 2: Core Functionality Fixes

| # | Fix | File(s) | Effort |
|---|-----|---------|--------|
| 2.1 | Pass ThinkingResult to ChatInterface | `system_orchestrator.cpp`, `chat_interface.hpp/.cpp` | 1h |
| 2.2 | Replace naive concept search with index | `system_orchestrator.cpp` or new `concept_index.hpp` | 2h |
| 2.3 | Fix `ensure_models_for()` to only handle new concepts | `system_orchestrator.cpp` | 15 min |
| 2.4 | Cache relation_count in LTM | `long_term_memory.hpp/.cpp` | 30 min |

### Priority 3: Evolution Integration

| # | Fix | File(s) | Effort |
|---|-----|---------|--------|
| 3.1 | Add Evolution members + init stage to SystemOrchestrator | `system_orchestrator.hpp/.cpp` | 1h |
| 3.2 | Wire ConceptProposer after ThinkingCycle | `system_orchestrator.cpp` | 1h |
| 3.3 | Add periodic EpistemicPromotion maintenance | `system_orchestrator.cpp` | 2h |
| 3.4 | Wire PatternDiscovery → CuriosityEngine feedback | `system_orchestrator.cpp` | 1h |

### Priority 4: Persistence & Reliability

| # | Fix | File(s) | Effort |
|---|-----|---------|--------|
| 4.1 | Wire WAL to LTM (or PersistentLTM as backend) | `system_orchestrator.cpp` | 2h |
| 4.2 | Save LTM in checkpoints | `system_orchestrator.cpp` | 1h |
| 4.3 | Save KAN state + Cognitive state in checkpoints | `system_orchestrator.cpp` | 1h |
| 4.4 | Add periodic auto-checkpoint thread | `system_orchestrator.cpp` | 1h |

### Priority 5: Stream System Rewrite

| # | Fix | File(s) | Effort |
|---|-----|---------|--------|
| 5.1 | ThinkStreams should use ThinkingPipeline (or at least CognitiveDynamics) | `think_stream.cpp` | 4h |
| 5.2 | Add output channel for stream results | `think_stream.hpp`, `stream_orchestrator.hpp` | 2h |
| 5.3 | Feed stream results back to Evolution | `system_orchestrator.cpp` | 2h |

### Priority 6: Training & Learning

| # | Fix | File(s) | Effort |
|---|-----|---------|--------|
| 6.1 | Implement MicroModel training trigger (post-ingest or periodic) | `system_orchestrator.cpp` | 2h |
| 6.2 | Generate training data from LTM relations | `micro_trainer.cpp` | 4h |

---

## Summary: Critical Path to Working Demo

```
1. Fix Makefile (BRAIN19_SRCS + -lcurl)           ← 10 min
2. Fix main.cpp (init before run_command)          ← 5 min
3. Compile, fix any errors                         ← 30 min
4. Test: ./brain19 (interactive REPL)              ← verify
5. Test: ./brain19 ask "What is logic?"            ← verify
6. Wire ThinkingResult → ChatInterface             ← 1h (makes answers smart)
7. Add Evolution subsystem                         ← 3h (makes system grow)
8. Add LTM persistence                             ← 2h (makes knowledge survive restart)
```

Steps 1-5 get a running demo. Steps 6-8 make it actually useful.
