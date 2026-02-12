#include "focus_cursor.hpp"
#include <algorithm>
#include <cmath>

namespace brain19 {

FocusCursor::FocusCursor(
    const LongTermMemory& ltm,
    MicroModelRegistry& registry,
    EmbeddingManager& embeddings,
    FocusCursorConfig config
)
    : ltm_(ltm)
    , registry_(registry)
    , embeddings_(embeddings)
    , config_(config)
    , mode_(config.default_mode)
{
    context_embedding_.fill(0.0);
}

void FocusCursor::seed(ConceptId start) {
    Vec10 ctx = embeddings_.make_context_embedding("query");
    seed(start, ctx);
}

void FocusCursor::seed(ConceptId start, const Vec10& initial_context) {
    current_ = start;
    depth_ = 0;
    context_embedding_ = initial_context;
    accumulated_energy_ = 0.0;
    seeded_ = true;
    terminated_ = false;
    history_.clear();
    visited_.clear();

    visited_.insert(start);

    // Record seed as first step
    TraversalStep step;
    step.concept_id = start;
    step.relation_from = RelationType::CUSTOM;  // No relation for seed
    step.weight_at_entry = 1.0;                 // Full weight for seed
    step.context_at_entry = context_embedding_;
    step.depth = 0;
    history_.push_back(step);
}

double FocusCursor::evaluate_edge(ConceptId from, ConceptId to, RelationType type) const {
    MicroModel* model = registry_.get_model(from);
    if (!model) return 0.0;

    const Vec10& e = embeddings_.get_relation_embedding(type);

    // Mix context with target info for context embedding
    Vec10 c_mixed = context_embedding_;
    // Blend in target-specific information
    Vec10 target_emb = embeddings_.make_target_embedding(
        0, static_cast<uint64_t>(from), static_cast<uint64_t>(to));
    for (size_t i = 0; i < EMBED_DIM; ++i) {
        c_mixed[i] = (1.0 - config_.context_mix_rate) * c_mixed[i]
                    + config_.context_mix_rate * target_emb[i];
    }

    return model->predict(e, c_mixed);
}

void FocusCursor::accumulate_context(ConceptId new_concept) {
    // Blend current context with information from the new concept
    Vec10 new_emb = embeddings_.make_target_embedding(
        0, static_cast<uint64_t>(current_), static_cast<uint64_t>(new_concept));
    for (size_t i = 0; i < EMBED_DIM; ++i) {
        context_embedding_[i] = (1.0 - config_.context_mix_rate) * context_embedding_[i]
                               + config_.context_mix_rate * new_emb[i];
    }
}

std::vector<FocusCursor::Candidate> FocusCursor::get_candidates() const {
    std::vector<Candidate> candidates;

    // Outgoing relations
    auto outgoing = ltm_.get_outgoing_relations(current_);
    for (const auto& rel : outgoing) {
        if (visited_.count(rel.target)) continue;  // Skip visited

        auto cinfo = ltm_.retrieve_concept(rel.target);
        if (!cinfo || cinfo->epistemic.is_invalidated()) continue;

        double score = evaluate_edge(current_, rel.target, rel.type);

        // Bonus for preferred relation type
        if (preferred_relation_ && rel.type == *preferred_relation_) {
            score *= 1.3;
            if (score > 1.0) score = 1.0;
        }

        candidates.push_back({rel.target, rel.type, score, true});
    }

    // Incoming relations (traverse backwards with discount)
    auto incoming = ltm_.get_incoming_relations(current_);
    for (const auto& rel : incoming) {
        if (visited_.count(rel.source)) continue;

        auto cinfo = ltm_.retrieve_concept(rel.source);
        if (!cinfo || cinfo->epistemic.is_invalidated()) continue;

        double score = evaluate_edge(current_, rel.source, rel.type) * 0.8;

        candidates.push_back({rel.source, rel.type, score, false});
    }

    // Sort by score descending
    std::sort(candidates.begin(), candidates.end(),
        [](const Candidate& a, const Candidate& b) { return a.score > b.score; });

    // Limit evaluation
    if (candidates.size() > config_.max_neighbors_to_evaluate) {
        candidates.resize(config_.max_neighbors_to_evaluate);
    }

    return candidates;
}

bool FocusCursor::check_termination() const {
    // 1. Max depth
    if (depth_ >= config_.max_depth) return true;

    // 2. Energy exhaustion
    if (accumulated_energy_ >= config_.energy_budget) return true;

    // 3. Goal completion
    if (mode_ == ExplorationMode::GOAL_DIRECTED && goal_.is_complete()) return true;

    // 4. No progress possible (checked in step())

    return false;
}

std::optional<ConceptId> FocusCursor::step() {
    if (!seeded_ || terminated_) return std::nullopt;

    if (check_termination()) {
        terminated_ = true;
        return std::nullopt;
    }

    auto candidates = get_candidates();

    // 4. No candidates = dead end
    if (candidates.empty()) {
        terminated_ = true;
        return std::nullopt;
    }

    // 5. Best candidate below threshold
    if (candidates[0].score < config_.min_weight_threshold) {
        terminated_ = true;
        return std::nullopt;
    }

    // Pick best candidate
    const Candidate& best = candidates[0];

    // Accumulate context before moving
    accumulate_context(best.target);

    // Move
    current_ = best.target;
    ++depth_;
    accumulated_energy_ += config_.energy_per_step;
    visited_.insert(current_);

    // Record step
    TraversalStep ts;
    ts.concept_id = current_;
    ts.relation_from = best.relation;
    ts.weight_at_entry = best.score;
    ts.context_at_entry = context_embedding_;
    ts.depth = depth_;
    history_.push_back(ts);

    // Update goal progress
    if (mode_ == ExplorationMode::GOAL_DIRECTED) {
        std::vector<ConceptId> visited_vec(visited_.begin(), visited_.end());
        goal_.update_progress(visited_vec, depth_);
    }

    // Clear preferred relation after use
    preferred_relation_.reset();

    return current_;
}

bool FocusCursor::step_to(ConceptId target) {
    if (!seeded_ || terminated_) return false;

    // Check termination before moving (same as step())
    if (check_termination()) {
        terminated_ = true;
        return false;
    }

    // Check target is actually a neighbor
    auto outgoing = ltm_.get_outgoing_relations(current_);
    auto incoming = ltm_.get_incoming_relations(current_);

    RelationType rel_type = RelationType::CUSTOM;
    double score = 0.0;
    bool found = false;

    for (const auto& rel : outgoing) {
        if (rel.target == target) {
            rel_type = rel.type;
            score = evaluate_edge(current_, target, rel.type);
            found = true;
            break;
        }
    }
    if (!found) {
        for (const auto& rel : incoming) {
            if (rel.source == target) {
                rel_type = rel.type;
                score = evaluate_edge(current_, target, rel.type) * 0.8;
                found = true;
                break;
            }
        }
    }
    if (!found) return false;

    accumulate_context(target);
    current_ = target;
    ++depth_;
    accumulated_energy_ += config_.energy_per_step;
    visited_.insert(current_);

    TraversalStep ts;
    ts.concept_id = current_;
    ts.relation_from = rel_type;
    ts.weight_at_entry = score;
    ts.context_at_entry = context_embedding_;
    ts.depth = depth_;
    history_.push_back(ts);

    // Update goal progress (same as step())
    if (mode_ == ExplorationMode::GOAL_DIRECTED) {
        std::vector<ConceptId> visited_vec(visited_.begin(), visited_.end());
        goal_.update_progress(visited_vec, depth_);
    }

    // Note: preferred_relation_ is NOT cleared here (unlike step()).
    // step_to() is a forced move — the preference wasn't consumed by scoring
    // and should remain available for the next free step().

    return true;
}

bool FocusCursor::backtrack() {
    if (history_.size() <= 1) return false;  // Can't backtrack past seed

    history_.pop_back();
    const TraversalStep& prev = history_.back();
    current_ = prev.concept_id;
    context_embedding_ = prev.context_at_entry;
    depth_ = prev.depth;
    // Don't remove from visited_ — prevents re-visiting
    // Don't refund energy — backtracking costs energy too

    terminated_ = false;  // Allow continuing after backtrack
    return true;
}

TraversalResult FocusCursor::deepen() {
    if (!seeded_) return {};

    while (!terminated_) {
        auto next = step();
        if (!next) break;
    }

    return result();
}

void FocusCursor::shift_focus(RelationType preferred_type) {
    preferred_relation_ = preferred_type;
}

CursorView FocusCursor::get_view() const {
    return CursorView{
        current_,
        depth_,
        context_embedding_,
        accumulated_energy_,
        mode_,
        history_.size()
    };
}

std::vector<FocusCursor> FocusCursor::branch(size_t k) const {
    std::vector<FocusCursor> branches;
    auto candidates = get_candidates();

    // Create one branch per top-k candidate (skip the best, which main cursor takes)
    for (size_t i = 1; i < candidates.size() && branches.size() < k; ++i) {
        FocusCursor copy(ltm_, registry_, embeddings_, config_);
        copy.current_ = current_;
        copy.depth_ = depth_;
        copy.context_embedding_ = context_embedding_;
        copy.accumulated_energy_ = accumulated_energy_;
        copy.seeded_ = true;
        copy.terminated_ = false;
        copy.mode_ = mode_;
        copy.history_ = history_;
        copy.visited_ = visited_;
        copy.goal_ = goal_;
        copy.preferred_relation_ = preferred_relation_;

        // Force-step each branch to a different candidate
        copy.step_to(candidates[i].target);
        branches.push_back(std::move(copy));
    }

    return branches;
}

TraversalResult FocusCursor::result() const {
    TraversalResult res;
    res.chain = history_;
    res.total_steps = history_.size();

    // Build concept sequence
    for (const auto& step : history_) {
        res.concept_sequence.push_back(step.concept_id);
    }

    // Build relation sequence (between consecutive steps)
    for (size_t i = 1; i < history_.size(); ++i) {
        res.relation_sequence.push_back(history_[i].relation_from);
    }

    // Compute average score (exclude seed step, which always has weight 1.0)
    double sum = 0.0;
    for (size_t i = 1; i < history_.size(); ++i) {
        sum += history_[i].weight_at_entry;
    }
    res.chain_score = (history_.size() <= 1) ? 0.0 : sum / static_cast<double>(history_.size() - 1);

    return res;
}

} // namespace brain19
