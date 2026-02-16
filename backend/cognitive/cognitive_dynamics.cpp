#include "cognitive_dynamics.hpp"
#include <cmath>
#include <algorithm>

namespace brain19 {

// =============================================================================
// CONSTRUCTOR / DESTRUCTOR
// =============================================================================

CognitiveDynamics::CognitiveDynamics(CognitiveDynamicsConfig config)
    : config_(config)
    , stats_{}
    , current_tick_(0)
{
    if (!config_.is_valid()) {
        config_ = CognitiveDynamicsConfig();  // Use defaults
    }
}

CognitiveDynamics::~CognitiveDynamics() = default;

void CognitiveDynamics::set_config(const CognitiveDynamicsConfig& config) {
    if (config.is_valid()) {
        config_ = config;
    }
}

void CognitiveDynamics::reset_stats() {
    stats_ = Stats{};
}

// =============================================================================
// CLAMPING HELPERS
// =============================================================================

double CognitiveDynamics::clamp_activation(double value) const {
    if (value < config_.spreader.min_activation) 
        return config_.spreader.min_activation;
    if (value > config_.spreader.max_activation) 
        return config_.spreader.max_activation;
    return value;
}

double CognitiveDynamics::clamp_salience(double value) const {
    if (value < 0.0) return 0.0;
    if (value > config_.salience.max_salience) 
        return config_.salience.max_salience;
    return value;
}

double CognitiveDynamics::clamp_focus(double value) const {
    if (value < 0.0) return 0.0;
    if (value > config_.focus.max_focus) 
        return config_.focus.max_focus;
    return value;
}

// =============================================================================
// SPREADING ACTIVATION
// =============================================================================

SpreadingStats CognitiveDynamics::spread_activation(
    ConceptId source,
    double initial_activation,
    ContextId context,
    const LongTermMemory& ltm,
    ShortTermMemory& stm
) {
    SpreadingStats stats;
    stats.reset();
    
    if (!config_.enable_spreading) {
        return stats;
    }
    
    // Validate source exists
    if (!ltm.exists(source)) {
        return stats;
    }
    
    // Validate initial activation
    initial_activation = clamp_activation(initial_activation);
    if (initial_activation < config_.spreader.activation_threshold) {
        return stats;
    }
    
    // Activate source in STM
    stm.activate_concept(context, source, initial_activation, ActivationClass::CONTEXTUAL);
    
    // Track visited to prevent cycles
    std::unordered_set<ConceptId> visited;
    
    // Start recursive spreading
    spread_recursive(
        source,
        initial_activation,
        0,  // depth = 0
        context,
        ltm,
        stm,
        visited,
        stats
    );
    
    stats.concepts_activated = visited.size();
    
    // Update global stats
    stats_.total_spreads++;
    stats_.last_spread = stats;
    
    return stats;
}

void CognitiveDynamics::spread_recursive(
    ConceptId current,
    double activation,
    size_t depth,
    ContextId context,
    const LongTermMemory& ltm,
    ShortTermMemory& stm,
    std::unordered_set<ConceptId>& visited,
    SpreadingStats& stats
) {
    // BASE CASE 1: Max depth reached
    if (depth >= config_.spreader.max_depth) {
        return;
    }
    
    // BASE CASE 2: Activation too weak
    if (activation < config_.spreader.activation_threshold) {
        return;
    }
    
    // BASE CASE 3: Already visited (cycle detection)
    if (visited.find(current) != visited.end()) {
        return;
    }
    
    // Mark visited
    visited.insert(current);
    
    // Get source cid for trust
    auto concept_opt = ltm.retrieve_concept(current);
    if (!concept_opt.has_value()) {
        return;
    }
    
    const ConceptInfo& source_concept = concept_opt.value();
    
    // Skip INVALIDATED concepts - they should not propagate
    if (source_concept.epistemic.is_invalidated()) {
        return;
    }
    
    // Get source trust (read-only access)
    double source_trust = config_.spreader.trust_weighted 
        ? source_concept.epistemic.trust 
        : 1.0;
    
    // Get outgoing relations
    auto relations = ltm.get_outgoing_relations(current);
    
    // Track max depth
    if (depth > stats.max_depth_reached) {
        stats.max_depth_reached = depth;
    }
    
    // Propagate to each target — RELATION-TYPE-AWARE (Convergence v2)
    for (const RelationInfo& rel : relations) {
        // Check target not invalidated (Audit #10)
        auto target_opt = ltm.retrieve_concept(rel.target);
        if (!target_opt.has_value() || target_opt->epistemic.is_invalidated()) {
            continue;
        }

        // Relation-type-aware behavior
        const RelationBehavior& behavior = get_behavior(rel.type);

        // FORMULA: propagated = activation × rel_weight × trust × spread_weight × direction × damping
        double weight = config_.spreader.relation_weighted ? rel.weight : 1.0;
        double damping = std::pow(config_.spreader.damping_factor,
                                  static_cast<double>(depth + 1));

        double propagated = activation * weight * source_trust
                          * behavior.spreading_weight * damping;
        propagated = clamp_activation(propagated);

        // Update statistics
        stats.total_propagations++;
        stats.total_activation_added += propagated;

        // Apply activation to target in STM
        if (behavior.spreading_direction > 0) {
            // Excitatory: activate or boost
            double existing = stm.get_concept_activation(context, rel.target);
            if (existing > 0.0) {
                stm.boost_concept(context, rel.target, propagated);
            } else {
                stm.activate_concept(context, rel.target, propagated, ActivationClass::CONTEXTUAL);
            }
        } else {
            // Inhibitory (OPPOSITION/CONTRADICTS): reduce target activation
            stm.inhibit_concept(context, rel.target, propagated);
        }

        // Temporal: only propagate forward in time
        if (rel.type == RelationType::TEMPORAL_AFTER) continue;

        // Recursively spread from target (always with positive magnitude)
        spread_recursive(
            rel.target,
            propagated * 0.8,  // reduced recursive propagation
            depth + 1,
            context,
            ltm,
            stm,
            visited,
            stats
        );
    }
}

SpreadingStats CognitiveDynamics::spread_activation_multi(
    const std::vector<ConceptId>& sources,
    double initial_activation,
    ContextId context,
    const LongTermMemory& ltm,
    ShortTermMemory& stm
) {
    SpreadingStats combined_stats;
    combined_stats.reset();
    
    if (!config_.enable_spreading || sources.empty()) {
        return combined_stats;
    }
    
    // Global visited set to avoid re-spreading
    std::unordered_set<ConceptId> global_visited;
    
    for (ConceptId source : sources) {
        if (!ltm.exists(source)) {
            continue;
        }
        
        double act = clamp_activation(initial_activation);
        if (act < config_.spreader.activation_threshold) {
            continue;
        }
        
        // Activate source
        stm.activate_concept(context, source, act, ActivationClass::CONTEXTUAL);
        
        // Spread from this source
        spread_recursive(
            source,
            act,
            0,
            context,
            ltm,
            stm,
            global_visited,
            combined_stats
        );
    }
    
    combined_stats.concepts_activated = global_visited.size();
    
    stats_.total_spreads++;
    stats_.last_spread = combined_stats;
    
    return combined_stats;
}

// =============================================================================
// SALIENCE COMPUTATION
// =============================================================================

double CognitiveDynamics::compute_activation_factor(
    ConceptId cid,
    ContextId context,
    const ShortTermMemory& stm
) const {
    return stm.get_concept_activation(context, cid);
}

double CognitiveDynamics::compute_trust_factor(
    ConceptId cid,
    const LongTermMemory& ltm
) const {
    auto info = ltm.retrieve_concept(cid);
    if (!info.has_value()) {
        return 0.0;
    }
    return info->epistemic.trust;
}

double CognitiveDynamics::compute_connectivity_factor(
    ConceptId cid,
    const LongTermMemory& ltm,
    size_t max_connectivity
) const {
    size_t count = ltm.get_relation_count(cid);
    if (max_connectivity == 0) {
        max_connectivity = std::max(size_t(1), count);
    }
    // Unified log normalization (Convergence v2, Audit #12)
    // log(count+1)/log(max+1) — same formula for single and batch modes
    return std::log(static_cast<double>(count + 1))
         / std::log(static_cast<double>(max_connectivity + 1));
}

double CognitiveDynamics::compute_recency_factor(
    ConceptId cid,
    ContextId context,
    uint64_t current_tick
) const {
    // Search only in the relevant context (not all contexts)
    auto it = focus_sets_.find(context);
    if (it == focus_sets_.end()) {
        return 0.0;
    }
    for (const auto& entry : it->second) {
        if (entry.concept_id == cid && current_tick > 0) {
            uint64_t ticks_since = (current_tick > entry.last_accessed_tick)
                ? (current_tick - entry.last_accessed_tick) : 0;
            // Exponential decay with half-life of ~10 ticks
            return std::exp(-0.07 * static_cast<double>(ticks_since));
        }
    }
    return 0.0;
}

double CognitiveDynamics::compute_query_boost(
    ConceptId cid,
    const std::vector<ConceptId>& query_concepts,
    const LongTermMemory& ltm
) const {
    if (query_concepts.empty()) {
        return 0.0;
    }
    
    // Direct match
    for (ConceptId q : query_concepts) {
        if (q == cid) {
            return config_.salience.query_boost_factor;
        }
    }
    
    // Check if connected to any query cid
    auto outgoing = ltm.get_outgoing_relations(cid);
    auto incoming = ltm.get_incoming_relations(cid);
    
    for (const auto& rel : outgoing) {
        for (ConceptId q : query_concepts) {
            if (rel.target == q) {
                return config_.salience.query_boost_factor * 0.5;
            }
        }
    }
    
    for (const auto& rel : incoming) {
        for (ConceptId q : query_concepts) {
            if (rel.source == q) {
                return config_.salience.query_boost_factor * 0.5;
            }
        }
    }
    
    return 0.0;
}

SalienceScore CognitiveDynamics::compute_salience(
    ConceptId cid,
    ContextId context,
    const LongTermMemory& ltm,
    const ShortTermMemory& stm,
    uint64_t current_tick
) const {
    SalienceScore score(cid);
    
    if (!config_.enable_salience) {
        return score;
    }
    
    if (!ltm.exists(cid)) {
        return score;
    }
    
    // Compute individual factors
    score.activation_contrib = compute_activation_factor(cid, context, stm);
    score.trust_contrib = compute_trust_factor(cid, ltm);
    // Single-concept: pass 0 for self-normalize mode
    score.connectivity_contrib = compute_connectivity_factor(cid, ltm, 0);
    score.recency_contrib = compute_recency_factor(cid, context, current_tick);
    score.query_boost = 0.0;
    
    // Weighted sum
    score.salience = 
        config_.salience.activation_weight * score.activation_contrib +
        config_.salience.trust_weight * score.trust_contrib +
        config_.salience.connectivity_weight * score.connectivity_contrib +
        config_.salience.recency_weight * score.recency_contrib +
        score.query_boost;
    
    score.salience = clamp_salience(score.salience);
    
    stats_.total_salience_computations++;
    
    return score;
}

std::vector<SalienceScore> CognitiveDynamics::compute_salience_batch(
    const std::vector<ConceptId>& concepts,
    ContextId context,
    const LongTermMemory& ltm,
    const ShortTermMemory& stm,
    uint64_t current_tick
) const {
    std::vector<SalienceScore> results;
    results.reserve(concepts.size());
    
    if (!config_.enable_salience || concepts.empty()) {
        return results;
    }
    
    // Find max connectivity for normalization
    size_t max_connectivity = 0;
    for (ConceptId c : concepts) {
        size_t count = ltm.get_relation_count(c);
        if (count > max_connectivity) {
            max_connectivity = count;
        }
    }
    if (max_connectivity == 0) {
        max_connectivity = 1;  // Avoid division by zero
    }
    
    // Compute salience for each cid
    for (ConceptId c : concepts) {
        SalienceScore score(c);
        
        if (!ltm.exists(c)) {
            results.push_back(score);
            continue;
        }
        
        score.activation_contrib = compute_activation_factor(c, context, stm);
        score.trust_contrib = compute_trust_factor(c, ltm);
        score.connectivity_contrib = compute_connectivity_factor(c, ltm, max_connectivity);
        score.recency_contrib = compute_recency_factor(c, context, current_tick);
        score.query_boost = 0.0;
        
        score.salience = 
            config_.salience.activation_weight * score.activation_contrib +
            config_.salience.trust_weight * score.trust_contrib +
            config_.salience.connectivity_weight * score.connectivity_contrib +
            config_.salience.recency_weight * score.recency_contrib +
            score.query_boost;
        
        score.salience = clamp_salience(score.salience);
        
        results.push_back(score);
        stats_.total_salience_computations++;
    }
    
    // Sort by salience (descending)
    std::sort(results.begin(), results.end());
    
    return results;
}

std::vector<SalienceScore> CognitiveDynamics::get_top_k_salient(
    const std::vector<ConceptId>& candidates,
    size_t k,
    ContextId context,
    const LongTermMemory& ltm,
    const ShortTermMemory& stm,
    uint64_t current_tick
) const {
    auto all_scores = compute_salience_batch(candidates, context, ltm, stm, current_tick);
    
    if (all_scores.size() <= k) {
        return all_scores;
    }
    
    all_scores.resize(k);
    return all_scores;
}

std::vector<SalienceScore> CognitiveDynamics::compute_query_salience(
    const std::vector<ConceptId>& query_concepts,
    const std::vector<ConceptId>& candidates,
    ContextId context,
    const LongTermMemory& ltm,
    const ShortTermMemory& stm,
    uint64_t current_tick
) const {
    std::vector<SalienceScore> results;
    results.reserve(candidates.size());
    
    if (!config_.enable_salience || candidates.empty()) {
        return results;
    }
    
    // Find max connectivity
    size_t max_connectivity = 1;
    for (ConceptId c : candidates) {
        size_t count = ltm.get_relation_count(c);
        if (count > max_connectivity) {
            max_connectivity = count;
        }
    }
    
    // Compute salience with query boost
    for (ConceptId c : candidates) {
        SalienceScore score(c);
        
        if (!ltm.exists(c)) {
            results.push_back(score);
            continue;
        }
        
        score.activation_contrib = compute_activation_factor(c, context, stm);
        score.trust_contrib = compute_trust_factor(c, ltm);
        score.connectivity_contrib = compute_connectivity_factor(c, ltm, max_connectivity);
        score.recency_contrib = compute_recency_factor(c, context, current_tick);
        score.query_boost = compute_query_boost(c, query_concepts, ltm);
        
        score.salience = 
            config_.salience.activation_weight * score.activation_contrib +
            config_.salience.trust_weight * score.trust_contrib +
            config_.salience.connectivity_weight * score.connectivity_contrib +
            config_.salience.recency_weight * score.recency_contrib +
            score.query_boost;
        
        score.salience = clamp_salience(score.salience);
        
        results.push_back(score);
        stats_.total_salience_computations++;
    }
    
    std::sort(results.begin(), results.end());
    
    return results;
}

// =============================================================================
// FOCUS MANAGEMENT
// =============================================================================

void CognitiveDynamics::init_focus(ContextId context) {
    focus_sets_[context] = std::vector<FocusEntry>();
}

void CognitiveDynamics::focus_on(ContextId context, ConceptId cid, double boost) {
    if (!config_.enable_focus_decay) {
        return;
    }
    
    auto& focus_set = focus_sets_[context];
    
    // Check if already focused
    for (auto& entry : focus_set) {
        if (entry.concept_id == cid) {
            // Boost existing focus
            entry.focus_score = clamp_focus(
                entry.focus_score + config_.focus.attention_boost + boost
            );
            entry.last_accessed_tick = current_tick_;
            stats_.total_focus_updates++;
            prune_focus_set(context);
            return;
        }
    }
    
    // Add new focus entry
    double initial_focus = clamp_focus(0.5 + config_.focus.attention_boost + boost);
    focus_set.push_back(FocusEntry(cid, initial_focus, current_tick_));
    
    stats_.total_focus_updates++;
    prune_focus_set(context);
}

void CognitiveDynamics::decay_focus(ContextId context) {
    if (!config_.enable_focus_decay) {
        return;
    }
    
    current_tick_++;
    
    auto it = focus_sets_.find(context);
    if (it == focus_sets_.end()) {
        return;
    }
    
    auto& focus_set = it->second;
    
    // Apply decay
    for (auto& entry : focus_set) {
        entry.focus_score *= (1.0 - config_.focus.decay_rate);
    }
    
    prune_focus_set(context);
}

void CognitiveDynamics::prune_focus_set(ContextId context) {
    auto it = focus_sets_.find(context);
    if (it == focus_sets_.end()) {
        return;
    }
    
    auto& focus_set = it->second;
    
    // Remove entries below threshold
    focus_set.erase(
        std::remove_if(focus_set.begin(), focus_set.end(),
            [this](const FocusEntry& e) {
                return e.focus_score < config_.focus.focus_threshold;
            }),
        focus_set.end()
    );
    
    // Sort by focus score (descending)
    std::sort(focus_set.begin(), focus_set.end());
    
    // Enforce max size
    if (focus_set.size() > config_.focus.max_focus_size) {
        focus_set.resize(config_.focus.max_focus_size);
    }
}

void CognitiveDynamics::update_access_time(ContextId context, ConceptId cid) {
    auto it = focus_sets_.find(context);
    if (it == focus_sets_.end()) {
        return;
    }
    for (auto& entry : it->second) {
        if (entry.concept_id == cid) {
            entry.last_accessed_tick = current_tick_;
            return;
        }
    }
}

std::vector<FocusEntry> CognitiveDynamics::get_focus_set(ContextId context) const {
    auto it = focus_sets_.find(context);
    if (it == focus_sets_.end()) {
        return {};
    }
    return it->second;
}

bool CognitiveDynamics::is_focused(ContextId context, ConceptId cid) const {
    auto it = focus_sets_.find(context);
    if (it == focus_sets_.end()) {
        return false;
    }
    
    for (const auto& entry : it->second) {
        if (entry.concept_id == cid) {
            return true;
        }
    }
    return false;
}

double CognitiveDynamics::get_focus_score(ContextId context, ConceptId cid) const {
    auto it = focus_sets_.find(context);
    if (it == focus_sets_.end()) {
        return 0.0;
    }
    
    for (const auto& entry : it->second) {
        if (entry.concept_id == cid) {
            return entry.focus_score;
        }
    }
    return 0.0;
}

void CognitiveDynamics::clear_focus(ContextId context) {
    focus_sets_.erase(context);
}

// =============================================================================
// THOUGHT PATH RANKING
// =============================================================================

double CognitiveDynamics::compute_path_score(
    const ThoughtPath& path,
    ContextId context,
    const LongTermMemory& ltm,
    const ShortTermMemory& stm
) const {
    if (path.empty()) {
        return 0.0;
    }
    
    double total_salience = 0.0;
    double total_trust = 0.0;
    double coherence = 1.0;  // Simplified coherence
    
    for (const auto& node : path.nodes) {
        auto score = compute_salience(node.concept_id, context, ltm, stm, current_tick_);
        total_salience += score.salience;
        
        auto info = ltm.retrieve_concept(node.concept_id);
        if (info.has_value()) {
            total_trust += info->epistemic.trust;
        }
    }
    
    size_t n = path.nodes.size();
    double avg_salience = total_salience / n;
    double avg_trust = total_trust / n;
    
    // Apply depth penalty
    double depth_factor = std::pow(1.0 - config_.thought_path.depth_penalty, n);
    
    double final_score = 
        config_.thought_path.salience_weight * avg_salience +
        config_.thought_path.trust_weight * avg_trust +
        config_.thought_path.coherence_weight * coherence;
    
    return final_score * depth_factor;
}

double CognitiveDynamics::score_path(
    const ThoughtPath& path,
    ContextId context,
    const LongTermMemory& ltm,
    const ShortTermMemory& stm
) const {
    return compute_path_score(path, context, ltm, stm);
}

std::vector<ThoughtPath> CognitiveDynamics::find_best_paths(
    ConceptId source,
    ContextId context,
    const LongTermMemory& ltm,
    const ShortTermMemory& stm
) const {
    std::vector<ThoughtPath> paths;
    
    if (!config_.enable_path_ranking) {
        return paths;
    }
    
    if (!ltm.exists(source)) {
        return paths;
    }
    
    // Initialize with source
    ThoughtPath initial;
    ThoughtPathNode root(source, 0, 1.0, 1.0, 0);
    initial.nodes.push_back(root);
    initial.total_score = 1.0;
    paths.push_back(initial);
    
    // Expand paths iteratively (beam search)
    for (size_t iter = 0; iter < config_.thought_path.max_depth; ++iter) {
        expand_paths(paths, context, ltm, stm, nullptr);
        
        // Keep top-K paths
        if (paths.size() > config_.thought_path.max_paths) {
            std::sort(paths.begin(), paths.end());
            paths.resize(config_.thought_path.max_paths);
        }
    }
    
    // Final sort and score
    for (auto& path : paths) {
        path.total_score = compute_path_score(path, context, ltm, stm);
    }
    std::sort(paths.begin(), paths.end());
    
    stats_.total_path_searches++;
    
    return paths;
}

std::vector<ThoughtPath> CognitiveDynamics::find_paths_to(
    ConceptId source,
    ConceptId target,
    ContextId context,
    const LongTermMemory& ltm,
    const ShortTermMemory& stm
) const {
    std::vector<ThoughtPath> paths;
    
    if (!config_.enable_path_ranking) {
        return paths;
    }
    
    if (!ltm.exists(source) || !ltm.exists(target)) {
        return paths;
    }
    
    // Initialize with source
    ThoughtPath initial;
    ThoughtPathNode root(source, 0, 1.0, 1.0, 0);
    initial.nodes.push_back(root);
    initial.total_score = 1.0;
    paths.push_back(initial);
    
    // Expand paths iteratively
    std::vector<ThoughtPath> complete_paths;
    
    for (size_t iter = 0; iter < config_.thought_path.max_depth; ++iter) {
        expand_paths(paths, context, ltm, stm, &target);
        
        // Extract paths that reached target
        for (auto it = paths.begin(); it != paths.end(); ) {
            if (!it->empty() && it->nodes.back().concept_id == target) {
                complete_paths.push_back(*it);
                it = paths.erase(it);
            } else {
                ++it;
            }
        }
        
        // Keep top-K paths for further expansion
        if (paths.size() > config_.thought_path.max_paths) {
            std::sort(paths.begin(), paths.end());
            paths.resize(config_.thought_path.max_paths);
        }
        
        if (paths.empty()) {
            break;
        }
    }
    
    // Score complete paths
    for (auto& path : complete_paths) {
        path.total_score = compute_path_score(path, context, ltm, stm);
    }
    std::sort(complete_paths.begin(), complete_paths.end());
    
    // Return top-K complete paths
    if (complete_paths.size() > config_.thought_path.max_paths) {
        complete_paths.resize(config_.thought_path.max_paths);
    }
    
    stats_.total_path_searches++;
    
    return complete_paths;
}

void CognitiveDynamics::expand_paths(
    std::vector<ThoughtPath>& paths,
    ContextId /*context*/,
    const LongTermMemory& ltm,
    const ShortTermMemory& /*stm*/,
    const ConceptId* target
) const {
    std::vector<ThoughtPath> expanded;
    
    for (const auto& path : paths) {
        if (path.empty()) continue;
        
        // Check depth limit
        if (path.nodes.size() >= config_.thought_path.max_depth) {
            expanded.push_back(path);  // Keep as-is
            continue;
        }
        
        ConceptId current = path.nodes.back().concept_id;
        
        // If we already reached target, keep path
        if (target && current == *target) {
            expanded.push_back(path);
            continue;
        }
        
        // Get outgoing relations
        auto relations = ltm.get_outgoing_relations(current);
        
        if (relations.empty()) {
            expanded.push_back(path);  // Dead end, keep path
            continue;
        }
        
        // Expand to each neighbor
        for (const auto& rel : relations) {
            // Skip if already in path (avoid cycles)
            bool in_path = false;
            for (const auto& node : path.nodes) {
                if (node.concept_id == rel.target) {
                    in_path = true;
                    break;
                }
            }
            if (in_path) continue;
            
            // Create expanded path
            ThoughtPath new_path = path;
            
            double local_score = rel.weight;
            auto info = ltm.retrieve_concept(rel.target);
            if (info.has_value()) {
                local_score *= info->epistemic.trust;
            }
            
            double cumulative = path.nodes.back().cumulative_score * local_score;
            
            ThoughtPathNode node(
                rel.target,
                rel.id,
                local_score,
                cumulative,
                path.nodes.size()
            );
            
            new_path.nodes.push_back(node);
            new_path.total_score = cumulative;
            
            expanded.push_back(new_path);
        }
    }
    
    paths = std::move(expanded);
}

} // namespace brain19
