#include "focus_cursor_manager.hpp"
#include "../memory/activation_level.hpp"
#include <algorithm>
#include <set>

namespace brain19 {

FocusCursorManager::FocusCursorManager(
    const LongTermMemory& ltm,
    MicroModelRegistry& registry,
    EmbeddingManager& embeddings,
    ShortTermMemory& stm,
    FocusCursorConfig config
)
    : ltm_(ltm)
    , registry_(registry)
    , embeddings_(embeddings)
    , stm_(stm)
    , config_(config)
{
}

QueryResult FocusCursorManager::process_seeds(
    const std::vector<ConceptId>& seeds,
    const Vec10& query_context
) {
    GoalState goal = GoalState::exploration_goal(query_context, "");
    return process_seeds(seeds, query_context, goal);
}

QueryResult FocusCursorManager::process_seeds(
    const std::vector<ConceptId>& seeds,
    const Vec10& query_context,
    const GoalState& goal
) {
    QueryResult qr;
    std::set<ConceptId> all_activated;

    for (ConceptId seed_id : seeds) {
        // Verify seed exists
        if (!ltm_.exists(seed_id)) continue;

        FocusCursor cursor(ltm_, registry_, embeddings_, config_);
        cursor.set_goal(goal);
        cursor.seed(seed_id, query_context);

        TraversalResult chain = cursor.deepen();

        // Collect activated concepts
        for (ConceptId cid : chain.concept_sequence) {
            all_activated.insert(cid);
        }

        qr.chains.push_back(std::move(chain));
    }

    // Select best chain by score
    if (!qr.chains.empty()) {
        auto best_it = std::max_element(qr.chains.begin(), qr.chains.end(),
            [](const TraversalResult& a, const TraversalResult& b) {
                return a.chain_score < b.chain_score;
            });
        qr.best_chain = *best_it;
    }

    qr.all_activated.assign(all_activated.begin(), all_activated.end());
    return qr;
}

void FocusCursorManager::persist_to_stm(ContextId ctx, const TraversalResult& chain) {
    for (size_t i = 0; i < chain.chain.size(); ++i) {
        const TraversalStep& step = chain.chain[i];

        // Activate concept with weight from traversal
        double activation = step.weight_at_entry;
        stm_.activate_concept(ctx, step.concept_id, activation, ActivationClass::CONTEXTUAL);

        // Activate relation between consecutive steps
        if (i > 0) {
            stm_.activate_relation(
                ctx,
                chain.chain[i - 1].concept_id,
                step.concept_id,
                step.relation_from,
                activation
            );
        }
    }
}

} // namespace brain19
