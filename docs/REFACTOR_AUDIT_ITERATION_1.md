# REFACTOR AUDIT — Iteration 1
## FocusCursor + Template-Engine + Ollama Removal + Pipeline Integration

**Date:** 2026-02-12
**Scope:** All new/changed files from INTEGRATION_PLAN.md implementation
**Method:** Manual code review of every new/changed file, cross-referencing APIs

---

## SCORE: 6/10

14 findings total: 6 critical, 5 medium, 3 low

---

## CRITICAL (Bugs / UB)

### C1: `TemplateResult::template_type` uninitialized — UB on read
**File:** `cursor/template_engine.hpp:37`
```cpp
struct TemplateResult {
    std::string text;
    TemplateType template_type;   // ← NO DEFAULT INITIALIZER
    size_t sentences_generated = 0;
    double confidence = 1.0;
};
```
`TemplateType` is an enum class. Default-constructing a `TemplateResult` leaves `template_type` with an indeterminate value. Reading it before assignment is **undefined behavior** per [dcl.init]/12.

**Impact:** Any code that default-constructs `TemplateResult` and reads `template_type` before calling `generate()` triggers UB.

**Fix:** Add `= TemplateType::DEFINITIONAL` default initializer.

---

### C2: `step_to()` skips termination checks — can exceed limits
**File:** `cursor/focus_cursor.cpp:195-241`

`step()` checks `check_termination()` before moving (line 143). `step_to()` does **not**. This means:
- Can exceed `max_depth`
- Can exceed `energy_budget`
- Can continue after goal completion

`branch()` calls `step_to()` (line 303), so branches can also exceed limits.

**Impact:** Traversal can overshoot configured limits when using `step_to()` or `branch()`.

**Fix:** Add `check_termination()` at the beginning of `step_to()`, returning false if terminated.

---

### C3: `step_to()` never updates goal progress
**File:** `cursor/focus_cursor.cpp:195-241`

`step()` calls `goal_.update_progress(visited_vec, depth_)` at lines 184-187. `step_to()` has no equivalent call. This means:
- Goal completion_metric is never updated when using `step_to()`
- Goal-directed termination will never trigger for step_to-based traversal
- `branch()` uses `step_to()` internally, so branched cursors don't update goals

**Impact:** Goal-directed traversal breaks when using `step_to()` or `branch()`.

**Fix:** Add goal progress update at the end of `step_to()`, mirroring `step()`.

---

### C4: `causal_goal()` factory sets targets = seeds — immediate completion
**File:** `cursor/goal_state.hpp:74-82`
```cpp
static GoalState causal_goal(const std::vector<ConceptId>& seeds, ...) {
    gs.target_concepts = seeds;  // BUG: targets == seeds
    gs.threshold = 0.7;
}
```

The `target_concepts` field means "concepts we want to **reach**." Setting it to seeds means the goal is "reach the concepts we already started from." After the first `step()`, `update_progress()` will find all seeds in `visited_concepts` (since seeds are always visited), giving `completion_metric = 1.0`. The cursor terminates after **exactly one step**.

For a causal query ("Was passiert wenn X?"), the cursor should follow causal links as far as possible, not terminate immediately.

**Impact:** `causal_goal()` produces chains of length 1, defeating its purpose.

**Fix:** `causal_goal()` should use empty `target_concepts` (like exploration) or accept explicit target concepts as parameter. Use exploration-style progress based on chain_length.

---

### C5: `MiniLLMFactory::created_count_` not initialized
**File:** `understanding/mini_llm_factory.hpp:80`
```cpp
size_t created_count_;  // ← NO INITIALIZER
```
Reading via `get_created_count()` returns indeterminate value if no creation occurred. Pre-existing issue but exacerbated by removing OllamaConfig constructor which may have initialized it.

**Fix:** Add `= 0` default member initializer.

---

### C6: `SpecializedMiniLLM::proposal_counter_` not initialized
**File:** `understanding/mini_llm_factory.hpp:164`
```cpp
mutable uint64_t proposal_counter_;  // ← NO INITIALIZER
```
Same issue. Used as counter, reading before first increment gives UB.

**Fix:** Add `= 0` default member initializer.

---

## MEDIUM (Inconsistencies / Design Issues)

### M1: `branch()` doesn't copy `preferred_relation_`
**File:** `cursor/focus_cursor.cpp:284-308`

`branch()` manually copies: current_, depth_, context_embedding_, accumulated_energy_, seeded_, terminated_, mode_, history_, visited_, goal_. But **not** `preferred_relation_`. If `shift_focus()` was called before `branch()`, branches lose the preference.

**Impact:** Low in practice since `preferred_relation_` is cleared after each `step()`, so it's only set between `shift_focus()` and the next `step()`. But inconsistent.

**Fix:** Add `copy.preferred_relation_ = preferred_relation_;` after line 300.

---

### M2: `classify()` tie-breaking silently favors KAUSAL_ERKLAEREND
**File:** `cursor/template_engine.cpp:74`
```cpp
if (causal_count >= def_count) {
    return TemplateType::KAUSAL_ERKLAEREND;
}
```
When `causal_count == def_count` (including both 0), returns KAUSAL_ERKLAEREND. This happens when:
- All relations are SUPPORTS or CUSTOM (neither causal nor definitional)
- Mixed chain with exactly equal counts

For the 0/0 case (all SUPPORTS/CUSTOM), KAUSAL_ERKLAEREND is misleading.

**Fix:** Change to `causal_count > def_count` (strict), add explicit fallback for tie/zero case.

---

### M3: Standalone `check_termination()` is incomplete vs `FocusCursor::check_termination()`
**File:** `cursor/termination.hpp:22-37`

The standalone version checks 3 conditions: max_depth, energy, goal completion.
FocusCursor::step() additionally checks:
- No candidates available (dead end)
- Best candidate below `min_weight_threshold`

These are missing from the standalone function. The header comment documents this but users of the standalone version may expect complete termination logic.

**Impact:** External code using `check_termination()` will miss dead-end and low-weight termination.

**Fix:** Document the limitation more prominently, or add a note that this function only checks static limits.

---

### M4: `chain_score` inflated by seed weight (always 1.0)
**File:** `cursor/focus_cursor.cpp:43,326-330`

Seed step has `weight_at_entry = 1.0` (hardcoded). Chain_score is average of all weights including seed. This inflates short chains: a 2-step chain with actual weight 0.3 scores (1.0+0.3)/2 = 0.65, while a 5-step chain with weights [0.6,0.5,0.5,0.4] scores (1.0+0.6+0.5+0.5+0.4)/5 = 0.6. The short chain scores higher despite being less informative.

`FocusCursorManager::process_seeds` selects best chain by chain_score, so this biases toward shorter, less useful chains.

**Fix:** Either exclude seed weight from score calculation, or weight the seed at 0.0.

---

### M5: Unused include in `thinking_pipeline.cpp`
**File:** `core/thinking_pipeline.cpp:2`
```cpp
#include "../cursor/template_engine.hpp"
```
`TemplateEngine` is never used in thinking_pipeline.cpp. Only used in test_pipeline_cursor.cpp.

**Fix:** Remove the include.

---

## LOW (Minor Issues)

### L1: `SystemOrchestrator` doesn't expose `ThinkingPipeline::Config`
**File:** `core/system_orchestrator.cpp:170`
```cpp
thinking_ = std::make_unique<ThinkingPipeline>();  // Uses all defaults
```
FocusCursor settings (max_depth, energy_budget, enable_focus_cursor, etc.) are not configurable from `SystemOrchestrator::Config`. The thinking pipeline always runs with default values.

**Impact:** No way to tune or disable FocusCursor in production without code change.

**Fix:** Add `ThinkingPipeline::Config thinking_config` to `SystemOrchestrator::Config` and forward it.

---

### L2: `run_thinking_cycle()` never uses `execute_with_goal()`
**File:** `core/system_orchestrator.cpp:555-573`

`SystemOrchestrator::run_thinking_cycle()` always calls `execute()` (exploration mode). The `execute_with_goal()` method exists but is unreachable from the orchestrator. Goal-directed traversal is effectively dead code in production.

**Impact:** No way to run goal-directed FocusCursor traversal via SystemOrchestrator.

**Fix:** Add `run_thinking_cycle(seeds, goal)` overload or detect goal from question type.

---

### L3: `GoalState::update_progress` is O(n*m)
**File:** `cursor/goal_state.hpp:50-58`

Nested loop iterates over `target_concepts * visited_concepts`. For realistic graph sizes (< 100 targets, < 50 visited) this is negligible, but for completeness: could use `std::set` or `std::unordered_set` for visited lookup.

**Impact:** Negligible for current usage patterns.

---

## FILES AUDITED

### New Files (10)
- `cursor/goal_state.hpp` — C4
- `cursor/traversal_types.hpp` — Clean
- `cursor/focus_cursor.hpp` — Clean
- `cursor/focus_cursor.cpp` — C2, C3, M1, M4
- `cursor/focus_cursor_manager.hpp` — Clean
- `cursor/focus_cursor_manager.cpp` — Clean
- `cursor/termination.hpp` — M3
- `cursor/conflict_resolution.hpp` — Clean
- `cursor/template_engine.hpp` — C1
- `cursor/template_engine.cpp` — M2, M5

### Modified Files (7)
- `core/thinking_pipeline.hpp` — Clean
- `core/thinking_pipeline.cpp` — M5
- `core/system_orchestrator.hpp` — Clean
- `core/system_orchestrator.cpp` — L1, L2
- `llm/chat_interface.hpp` — Clean
- `llm/chat_interface.cpp` — Clean
- `understanding/mini_llm_factory.hpp` — C5, C6

### Modified Files (Ollama removal, no new issues)
- `main.cpp` — Clean
- `demo_chat.cpp` — Clean

---

## NEXT: Fix all C1-C6, M1-M5. Then re-audit.
