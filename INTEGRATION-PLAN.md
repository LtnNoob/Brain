# Brain19 KAN-LLM Integration Plan — `/api/ask` Pipeline Fix

**Status:** URGENT — Testing blocked  
**Date:** 2026-02-13  
**Problem:** `/api/ask` bypasses the cognitive pipeline. The thinking cycle EXISTS and WORKS, but its results aren't properly used for response generation.

---

## Current State Analysis

### What EXISTS and WORKS ✅
1. **ThinkingPipeline** (10 steps): Seeds → Spreading Activation → Salience → FocusCursor → RelevanceMaps → ThoughtPaths → Curiosity → UnderstandingLayer → KAN Validation → Done
2. **KAN-LLM Hybrid Layer**: `KanValidator`, `HypothesisTranslator`, `EpistemicBridge`, `DomainManager` — all implemented in `backend/hybrid/`
3. **Understanding Layer**: `MiniLLM`, `MiniLLMFactory`, proposals — all in `backend/understanding/`
4. **Cognitive Dynamics**: Spreading activation, salience, focus, thought paths
5. **GDO** (Global Dynamics Operator): Energy injection, activation landscapes

### What's BROKEN ❌
The `ask()` flow in `SystemOrchestrator::ask()` does this:
1. Keyword-match seeds from question → ✅ OK
2. Inject energy into GDO → ✅ OK  
3. `run_thinking_cycle(seeds)` → ✅ RUNS THE FULL PIPELINE (including Understanding + KAN validation)
4. **BUT THEN**: Extracts only `top_salient` concept IDs and `thought_path` summaries
5. Passes to `ChatInterface::ask_with_context()` which just **formats concept definitions as text**
6. **LOST**: Understanding results, KAN validations, curiosity triggers, relevance maps, cursor results

### The Gap
```
ThinkingPipeline produces:
  - activated_concepts ✅ used
  - top_salient ✅ used (IDs only)
  - best_paths ✅ used (summaries only)  
  - curiosity_triggers ❌ DISCARDED
  - combined_relevance ❌ DISCARDED
  - understanding (MiniLLM proposals) ❌ DISCARDED
  - validated_hypotheses (KAN results) ❌ DISCARDED
  - cursor_result ❌ DISCARDED
  - generated_goals ❌ DISCARDED
```

The entire KAN-LLM hybrid pipeline runs but its output is thrown away before response generation.

---

## Integration Plan

### Phase 1: Wire ThinkingResult into ChatInterface (CRITICAL, do first)

**File: `backend/core/system_orchestrator.cpp` lines ~616-640**

Current:
```cpp
auto thinking_result = run_thinking_cycle(seeds);
std::vector<ConceptId> salient_ids;
for (const auto& s : thinking_result.top_salient)
    salient_ids.push_back(s.concept_id);
// ... build path summaries ...
return chat_->ask_with_context(question, *ltm_, salient_ids, path_summaries, intent);
```

Change to pass full `ThinkingResult`:
```cpp
auto thinking_result = run_thinking_cycle(seeds);
return chat_->ask_with_thinking(question, *ltm_, thinking_result, intent);
```

**File: `backend/llm/chat_interface.hpp`** — Add new method:
```cpp
ChatResponse ask_with_thinking(
    const std::string& question,
    const LongTermMemory& ltm,
    const ThinkingResult& thinking,
    QueryIntent intent = QueryIntent::UNKNOWN
);
```

**File: `backend/llm/chat_interface.cpp`** — Implement `ask_with_thinking()`:
- Use understanding proposals for generated reasoning
- Use KAN validation results for confidence-weighted answers  
- Use curiosity triggers to suggest follow-up questions
- Use thought paths for explanation chains
- Use cursor results for traversal-informed context

### Phase 2: Response Generation with KAN-Validated Knowledge

The response should be structured as:
```
1. CORE ANSWER: From top salient concepts + definitions (existing)
2. REASONING CHAIN: From ThoughtPaths (existing but underused)
3. GENERATED INSIGHTS: From Understanding Layer proposals
4. KAN-VALIDATED: From validated_hypotheses (with trust scores)
5. OPEN QUESTIONS: From curiosity triggers
```

**New `format_thinking_response()` method:**
```cpp
std::string format_thinking_response(
    const std::vector<ConceptInfo>& relevant,
    const ThinkingResult& thinking,
    const LongTermMemory& ltm
);
```

### Phase 3: MiniLLM Integration for Generative Responses

Currently `ask_with_context` never calls Ollama — it just formats stored knowledge. The pipeline should:

1. Build context from ThinkingResult (salient concepts + paths + KAN results)
2. Send context + question to MiniLLM for generative response
3. Classify MiniLLM output epistemically (the EpistemicBridge already does this)
4. Fuse: deterministic knowledge + generated reasoning + confidence annotations

**File: `backend/llm/chat_interface.cpp`** — In `ask_with_thinking()`:
```cpp
// After collecting deterministic knowledge...
if (thinking.understanding.meaning_proposals.size() > 0 ||
    thinking.validated_hypotheses.size() > 0) {
    
    // Build LLM prompt from thinking context
    std::string llm_context = build_llm_context(relevant, thinking, ltm);
    
    // Call MiniLLM for generative synthesis
    auto llm_response = understanding_layer_->generate_response(question, llm_context);
    
    response.answer = fuse_response(deterministic_answer, llm_response, thinking);
    response.used_llm = true;
}
```

### Phase 4: Domain-Aware Routing

Use `DomainManager` to route questions to domain-specific processing:
```cpp
// In SystemOrchestrator::ask():
auto domains = domain_manager_->cluster_by_domain(seeds, *ltm_);
// Use domain-specific KAN validator configs
// Route to appropriate MiniLLM personality
```

---

## Implementation Priority

| Step | What | Impact | Effort |
|------|------|--------|--------|
| 1a | Pass ThinkingResult to ChatInterface | Unblocks everything | 1h |
| 1b | Format KAN results + understanding in response | Visible improvement | 2h |
| 2 | MiniLLM generative response fusion | Actual "thinking" answers | 3h |
| 3 | Domain routing via DomainManager | Smarter responses | 2h |
| 4 | Cursor-informed context building | Deeper reasoning | 2h |

**Total: ~10h of focused work to wire everything together.**

---

## Key Insight

The architecture Felix envisioned ALREADY EXISTS in code:
- KAN-Relations ✅ (`backend/kan/`)
- Pattern Matching ✅ (RelevanceMaps, MicroModels)
- Topic Recognition ✅ (DomainManager clustering)
- Generated Thinking ✅ (UnderstandingLayer proposals)
- MiniLLM Orchestration ✅ (MiniLLMFactory)
- KAN Validation ✅ (KanValidator, EpistemicBridge)
- Fusion ❌ **THIS IS THE MISSING PIECE**

The ThinkingPipeline runs all 10 steps. The results just get dropped at the `SystemOrchestrator::ask()` → `ChatInterface` boundary. **The fix is integration/wiring, not new architecture.**

---

## Files to Modify

1. `backend/llm/chat_interface.hpp` — Add `ask_with_thinking()` 
2. `backend/llm/chat_interface.cpp` — Implement generative response with full ThinkingResult
3. `backend/core/system_orchestrator.cpp` — Pass ThinkingResult to new method
4. `backend/core/system_orchestrator.hpp` — (minor: expose understanding_layer to chat if needed)

## Files NOT to Modify
- ThinkingPipeline — already correct
- KAN subsystem — already works  
- Understanding Layer — already works
- Hybrid Layer — already works
- Epistemic system — never touch this
