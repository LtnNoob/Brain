# Brain19 — Graph Architecture Refactor

**Datum:** 2026-02-12  
**Status:** Merged Design Document  
**Quellen:** Database & Graph Architecture Refactor Plan (Felix, 12.02) + FOCUS_CURSOR_DESIGN.md (11.02)  
**Ersetzt:** FOCUS_CURSOR_DESIGN.md (wird Subset dieses Dokuments)

---

## 1. Architectural Principle

One unified graph structure. Two operators:

1. **Global Dynamics Operator** — maintains activation landscape, inhibition, damping
2. **Focus Traversal Operator** (FocusCursor) — navigates graph sequentially for intentional reasoning

**No shadow graphs. No duplicated memory. Single Source of Truth.**

---

## 2. Extended Data Structures

### 2.1 Concept (Node) — Extended

```cpp
struct Concept {
    ConceptId id;
    std::string label;
    VecN embedding;                    // existing
    
    // --- NEW: Dual-Mode Fields ---
    double activation_score;           // computed by Global Dynamics Operator
    double salience_score;             // context-dependent, recomputed per query
    double structural_confidence;      // from KAN validation (structure-derived trust)
    double semantic_confidence;        // from MiniLLM/embedding similarity
    
    // --- EXISTING (unchanged) ---
    EpistemicMetadata epistemic;       // type, trust, etc.
};
```

**Notes:**
- `activation_score` → maintained by Global Field (background process)
- `salience_score` → recomputed per query context
- `structural_confidence` → KAN validates structural consistency of this node
- `semantic_confidence` → MiniLLM/embedding validates semantic coherence

### 2.2 Relation (Edge) — Extended

```cpp
struct Relation {
    ConceptId source_id;
    ConceptId target_id;
    RelationType type;
    double weight;                     // existing: base/static weight
    
    // --- NEW: Dual-Mode Fields ---
    double dynamic_weight;             // recomputed per thinking cycle via MicroModel::predict()
    double inhibition_factor;          // dampens this edge (0.0 = no inhibition, 1.0 = fully inhibited)
    double structural_strength;        // from KAN validation (how structurally justified is this edge?)
};
```

**Notes:**
- `dynamic_weight` = `MicroModel::predict(e_rel, context)` — changes with cursor context
- `inhibition_factor` = set by Global Dynamics when activation patterns conflict
- `structural_strength` = KAN's confidence in this edge's structural validity

### 2.3 GoalState — NEW

```cpp
struct GoalState {
    enum class GoalType {
        REACH_CONCEPT,      // navigate to a specific concept
        ANSWER_QUERY,       // find path explaining a query
        EXPLORE_REGION,     // curiosity-driven exploration
        VALIDATE_CLAIM      // check if a relation/claim holds
    };
    
    GoalType goal_type;
    std::vector<ConceptId> target_concepts;   // goal targets
    
    // Completion criteria
    double completion_metric = 0.0;           // current progress (0..1)
    double threshold = 0.8;                   // completion_metric >= threshold → done
    double priority_weight = 1.0;             // importance vs other active goals
    
    // Evaluation
    bool is_complete() const { return completion_metric >= threshold; }
    void update_progress(const CursorView& view, const std::vector<TraversalStep>& history);
};
```

**Used to:**
- Guide ThinkingPipeline (what are we trying to achieve?)
- Evaluate termination (are we done?)
- Constrain CuriosityEngine (don't wander off-goal)

### 2.4 FocusCursor — Extended (from FOCUS_CURSOR_DESIGN.md)

```cpp
struct FocusCursorState {
    ConceptId current_node_id;
    std::vector<TraversalStep> path_history;
    size_t depth;
    double accumulated_energy;          // NEW: energy budget for traversal
    
    enum class ExplorationMode {
        GREEDY,         // follow strongest edge
        EXPLORATORY,    // allow weaker edges (curiosity-driven)
        GOAL_DIRECTED   // constrained by GoalState
    } exploration_mode = ExplorationMode::GREEDY;
};
```

**Important:** Stores only references (Node IDs). Does NOT duplicate graph data.

The full FocusCursor class (with `step()`, `deepen()`, `branch()`, `backtrack()`, `shift_focus()`) remains as specified in §3 of FOCUS_CURSOR_DESIGN.md — see Appendix A.

---

## 3. Termination Logic

```cpp
bool check_termination(
    const GoalState& goal,
    const CursorView& view,
    const FocusCursorState& cursor,
    const FocusCursorConfig& config
) {
    // 1. Goal reached
    if (goal.is_complete()) return true;
    
    // 2. Target concept reached
    for (auto target : goal.target_concepts) {
        if (cursor.current_node_id == target) return true;
    }
    
    // 3. Confidence above threshold
    if (view.focus_activation > goal.threshold) {
        // Check if we're at a high-confidence answer
        // Only terminate if structural + semantic agree
        // (uses conflict resolution formula)
    }
    
    // 4. Information gain below epsilon
    if (cursor.path_history.size() >= 2) {
        auto& last = cursor.path_history.back();
        auto& prev = cursor.path_history[cursor.path_history.size() - 2];
        double info_gain = std::abs(last.weight_at_entry - prev.weight_at_entry);
        if (info_gain < 0.01) return true;  // diminishing returns
    }
    
    // 5. Max depth reached
    if (cursor.depth >= config.max_depth) return true;
    
    // 6. Energy exhausted
    if (cursor.accumulated_energy <= 0.0) return true;
    
    return false;
}
```

---

## 4. Conflict Resolution

When KAN (structural) and MiniLLM (semantic) disagree on a concept's importance:

```cpp
double effective_priority(const Concept& c) {
    constexpr double alpha = 0.4;   // structural weight
    constexpr double beta  = 0.4;   // semantic weight  
    constexpr double gamma = 0.2;   // activation weight
    
    return alpha * c.structural_confidence
         + beta  * c.semantic_confidence
         + gamma * c.activation_score;
}
```

**α, β, γ configurable.** Prevents dominance of either structure or semantics.

**Use cases:**
- KAN says edge is structurally strong but MiniLLM gives low semantic score → medium priority
- Both agree → high/low priority (clear signal)
- Activation is high but confidence is low → needs validation, not trust

---

## 5. Dual-Mode Integration

### 5.1 Global Dynamics Operator

Runs as **background process** (not per-query):

```
On every thinking cycle:
    for each concept c in active_set:
        c.activation_score = damped_propagation(c, neighbors)
        apply_inhibition(c)
        decay(c.activation_score)
```

**Operates on:** activation_score, inhibition, damping  
**Purpose:** Maintains stability and overview — the "ambient awareness" of the graph  
**Does NOT:** Navigate or make decisions

### 5.2 Focus Traversal Operator (FocusCursor)

Runs **per query/goal**:

```
Algorithm:
    initialize_goal(query)
    seed_cursor(find_seeds(query))
    
    while not check_termination(goal, view, cursor, config):
        next_node = argmax(dynamic_weight(current_node, context))
        update_cursor(next_node)
        update_local_activation(next_node)  // boost activation of traversed nodes
        evaluate_goal_progress(goal, cursor)
```

**Key difference from FOCUS_CURSOR_DESIGN:** Now goal-directed, not just greedy traversal.

### 5.3 How They Interact

```
Global Dynamics          Focus Traversal
     │                        │
     │  activation_score ────→│  (cursor uses global activation as input)
     │                        │
     │←──── local_boost ──────│  (traversed nodes get activation boost)
     │                        │
     │  inhibition ──────────→│  (inhibited edges get lower dynamic_weight)
     │                        │
     │←──── goal_feedback ────│  (goal progress influences global field)
```

---

## 6. CuriosityEngine Integration

```cpp
struct CuriosityTrigger {
    enum class TriggerType {
        LOW_CONFIDENCE,     // concept has low structural/semantic confidence
        UNEXPLORED_REGION,  // area of graph rarely visited
        CONTRADICTION,      // conflicting edges detected
        HIGH_NOVELTY        // new ingested concept with few connections
    };
    
    TriggerType trigger_type;
    ConceptId target_node_id;
    double priority;
};
```

**Curiosity must:**
- Respect GoalState — suggestions are **deprioritized** if a goal is active
- Suggest but not override goal priority
- Queue triggers for when current goal completes or energy is available

```cpp
// In control loop:
void process_curiosity(GoalState& goal, CuriosityEngine& curiosity) {
    auto triggers = curiosity.get_pending_triggers();
    
    for (auto& trigger : triggers) {
        if (goal.is_complete() || trigger.priority > goal.priority_weight * 1.5) {
            // Only act on curiosity if goal is done OR trigger is urgent
            create_exploration_goal(trigger);
        }
    }
}
```

---

## 7. Unified Control Loop

```cpp
class Brain19ControlLoop {
public:
    void run(const std::string& query) {
        // 1. Parse query → GoalState
        GoalState goal = initialize_goal(query);
        
        // 2. Find seeds → initialize cursor
        auto seeds = find_seeds(query);
        FocusCursor cursor = initialize_cursor(seeds, query);
        
        // 3. Main loop
        while (!check_termination(goal, cursor.get_view(), cursor.state(), config_)) {
            // a) Update global activation field (background, not blocking)
            update_global_field();
            
            // b) Process curiosity triggers (may spawn side-goals)
            process_curiosity(goal, curiosity_);
            
            // c) Maybe shift focus based on KAN policy
            auto policy = kan_policy_.evaluate(cursor);
            if (policy.should_shift_focus) {
                cursor.shift_focus(policy.suggested_relation);
            }
            
            // d) Traverse one step
            auto next = cursor.step();
            if (!next) break;  // dead end
            
            // e) Resolve conflicts at new position
            resolve_conflicts(cursor.position());
            
            // f) Update goal progress
            goal.update_progress(cursor.get_view(), cursor.history());
        }
        
        // 4. Build result from cursor chain
        auto result = cursor.result();
        
        // 5. Persist to STM
        cursor_manager_.persist_to_stm(current_context_, result);
        
        // 6. Generate language output
        auto response = language_engine_.generate(result);
    }
    
private:
    FocusCursorConfig config_;
    CuriosityEngine curiosity_;
    KANPolicy kan_policy_;
    FocusCursorManager cursor_manager_;
    LanguageEngine language_engine_;
    GlobalDynamicsOperator global_dynamics_;
};
```

---

## 8. Components NOT Requiring Changes

These existing systems work as-is:
- **Persistence Layer** (WAL, mmap, checkpoints)
- **LTM Storage** (concepts + relations storage)
- **Checkpoint System** (atomic brain-state snapshots)
- **MicroModels** (per-concept learning — used by cursor for `predict()`)
- **Existing Graph Storage** (only extended, not replaced)

**Only extensions required** — no breaking changes to existing data structures.

---

## 9. Implementation Priority

| Priority | Component | Effort | Dependencies |
|----------|-----------|--------|-------------|
| 1 | Extended Concept/Relation fields (§2.1, §2.2) | 1 day | None |
| 2 | GoalState (§2.3) | 1 day | None |
| 3 | FocusCursor with goal-awareness (§2.4) | 2-3 days | Phase 0 fixes |
| 4 | Termination Logic (§3) | 0.5 day | GoalState + FocusCursor |
| 5 | Conflict Resolution (§4) | 0.5 day | Extended fields |
| 6 | Global Dynamics Operator (§5.1) | 2 days | Extended fields |
| 7 | Dual-Mode Integration (§5.3) | 1-2 days | Both operators |
| 8 | CuriosityEngine Integration (§6) | 1 day | GoalState |
| 9 | Unified Control Loop (§7) | 2-3 days | All above |

**Total: ~11-14 days** (Abende + Wochenenden)

---

## 10. Summary

Brain19 does not require a new memory system. It requires:

1. **Extended node/edge metadata** — structural + semantic confidence, dynamic weights
2. **FocusCursor state object** — sequential graph navigation (detailed in Appendix A)
3. **GoalState object** — intentional, goal-directed cognition
4. **Deterministic conflict resolution** — α·structural + β·semantic + γ·activation
5. **Dual-Mode operation** — Global Dynamics (ambient) + Focus Traversal (intentional)
6. **Unified Control Loop** — orchestrates all components coherently

This enables **intentional, coherent cognition without memory duplication.**

---

## Appendix A: FocusCursor Detailed Specification

*The complete FocusCursor class specification, algorithms (step, evaluate_neighbors, compute_weight, accumulate_context, deepen, branch, shift_focus), concrete examples ("Was passiert wenn Eis schmilzt?", "Warum brauchen Pflanzen Licht?"), integration details (MicroModels, KAN, STM, Language Engine), and stability guarantees are preserved from FOCUS_CURSOR_DESIGN.md.*

*See: FOCUS_CURSOR_DESIGN.md for the full 860-line specification — it remains valid as the detailed implementation reference for the Focus Traversal Operator described in this document.*
