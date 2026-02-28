#include "curiosity_engine.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <sstream>
#include <unordered_set>

namespace brain19 {

// =============================================================================
// Construction
// =============================================================================

CuriosityEngine::CuriosityEngine(CuriosityConfig config)
    : config_(std::move(config))
{
    normalize_weights();
}

CuriosityEngine::~CuriosityEngine() = default;

void CuriosityEngine::normalize_weights() {
    weights_[0]  = config_.w_pain;
    weights_[1]  = config_.w_trust;
    weights_[2]  = config_.w_model;
    weights_[3]  = config_.w_nn_kan;
    weights_[4]  = config_.w_topology;
    weights_[5]  = config_.w_contradiction;
    weights_[6]  = config_.w_pred_error;
    weights_[7]  = config_.w_novelty;
    weights_[8]  = config_.w_episodic;
    weights_[9]  = config_.w_activation;
    weights_[10] = config_.w_edge_weight;
    weights_[11] = config_.w_quality_deg;

    double sum = 0.0;
    for (double w : weights_) sum += w;
    if (sum > 0.0) {
        for (double& w : weights_) w /= sum;
    }
}

// =============================================================================
// Phase 1: OBSERVE — Build SystemSnapshot from subsystem refs
// =============================================================================

SystemSnapshot CuriosityEngine::observe(
    const LongTermMemory& ltm,
    const ConceptModelRegistry& registry,
    const EpisodicMemory& episodic,
    const ErrorCollector& error_collector,
    const std::unordered_map<ConceptId, double>& seed_pain_scores
) const {
    SystemSnapshot snap;

    auto now = std::chrono::steady_clock::now();
    snap.timestamp_ms = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()).count());

    auto all_ids = ltm.get_all_concept_ids();
    snap.system.total_concepts = all_ids.size();
    snap.system.total_relations = ltm.total_relation_count();
    snap.system.total_episodes = episodic.episode_count();
    snap.system.total_corrections = error_collector.total_corrections();

    if (snap.system.total_concepts > 0) {
        snap.system.avg_degree = static_cast<double>(snap.system.total_relations)
                                / static_cast<double>(snap.system.total_concepts);
        double max_edges = static_cast<double>(snap.system.total_concepts)
                         * static_cast<double>(snap.system.total_concepts - 1);
        snap.system.graph_density = (max_edges > 0.0)
            ? static_cast<double>(snap.system.total_relations) / max_edges
            : 0.0;
    }

    // Aggregate model stats
    double total_model_loss = 0.0;
    size_t model_count = 0;

    snap.concepts.reserve(all_ids.size());
    for (ConceptId cid : all_ids) {
        ConceptSignals cs;
        cs.concept_id = cid;

        // [5] Graph Topology
        cs.relation_count = ltm.get_relation_count(cid);

        // [7] Trust + epistemic type
        auto info = ltm.retrieve_concept(cid);
        if (info) {
            cs.trust = info->epistemic.trust;
            cs.epistemic_type = info->epistemic.type;
        }

        // [3] CM Confidence
        const ConceptModel* model = registry.get_model(cid);
        if (model) {
            cs.has_model = true;
            cs.model_converged = model->is_converged();
            cs.model_loss = model->final_loss();
            total_model_loss += cs.model_loss;
            ++model_count;
            if (cs.model_converged) ++snap.system.converged_models;
        }

        // [4] NN vs KAN Divergence — computed from episodic data
        // Look at recent episodes for this concept to detect NN/KAN disagreement
        auto concept_episodes = episodic.episodes_for_concept(cid);
        cs.episode_count = concept_episodes.size();

        if (!concept_episodes.empty()) {
            double quality_sum = 0.0;
            double divergence_sum = 0.0;
            size_t div_count = 0;
            for (const auto* ep : concept_episodes) {
                quality_sum += ep->quality;
                for (const auto& step : ep->steps) {
                    if (step.concept_id == cid || step.from_concept == cid) {
                        double div = std::abs(step.nn_quality - step.kan_quality);
                        divergence_sum += div;
                        ++div_count;
                    }
                }
            }
            cs.avg_episode_quality = quality_sum / static_cast<double>(concept_episodes.size());
            if (div_count > 0) {
                cs.nn_kan_divergence = divergence_sum / static_cast<double>(div_count);
            }
        }

        // [8] Novelty: inverse of episode count relative to average
        // (computed after all concepts are gathered)

        // [1] Pain/Reward — seed pain EMA
        auto pain_it = seed_pain_scores.find(cid);
        if (pain_it != seed_pain_scores.end()) {
            cs.seed_pain_ema = pain_it->second;
        }

        // [1] Pain/Reward — avg edge pain from outgoing relations
        auto outgoing = ltm.get_outgoing_relations(cid);
        if (!outgoing.empty()) {
            double weight_sum = 0.0;
            double weight_var_sum = 0.0;
            size_t contradiction_count = 0;

            for (const auto& rel : outgoing) {
                double pain = std::max(0.0, 1.0 - rel.weight);
                cs.avg_edge_pain += pain;
                weight_sum += rel.weight;
                if (rel.type == RelationType::CONTRADICTS) {
                    ++contradiction_count;
                }
            }
            cs.avg_edge_pain /= static_cast<double>(outgoing.size());
            cs.avg_edge_weight = weight_sum / static_cast<double>(outgoing.size());

            // [9] Contradictions
            if (contradiction_count > 0) {
                cs.has_contradictions = true;
                cs.contradiction_ratio = static_cast<double>(contradiction_count)
                                       / static_cast<double>(outgoing.size());
            }

            // Edge weight variance
            for (const auto& rel : outgoing) {
                double diff = rel.weight - cs.avg_edge_weight;
                weight_var_sum += diff * diff;
            }
            cs.edge_weight_variance = weight_var_sum / static_cast<double>(outgoing.size());
        }

        // [2] Prediction Error — correction count
        const auto& corrections = error_collector.get_corrections(cid);
        cs.correction_count = corrections.size();

        snap.concept_index[cid] = snap.concepts.size();
        snap.concepts.push_back(std::move(cs));
    }

    // System-level model stats
    if (model_count > 0) {
        snap.system.avg_model_loss = total_model_loss / static_cast<double>(model_count);
    }

    // [8] Novelty: compute relative to average episode count
    if (!snap.concepts.empty()) {
        double avg_episodes = static_cast<double>(snap.system.total_episodes)
                            / static_cast<double>(snap.concepts.size());
        for (auto& cs : snap.concepts) {
            // Low episodes relative to average = high novelty
            cs.novelty_score = std::clamp(
                1.0 - static_cast<double>(cs.episode_count) / (avg_episodes + 1.0),
                0.0, 1.0);
        }
    }

    return snap;
}

// =============================================================================
// Phase 2: SCORE — Per-concept curiosity scores from weighted dimensions
// =============================================================================

static inline double clamp01(double v) {
    return std::clamp(v, 0.0, 1.0);
}

double CuriosityEngine::score_pain(const ConceptSignals& cs) const {
    return clamp01(0.5 * cs.avg_edge_pain + 0.5 * cs.seed_pain_ema);
}

double CuriosityEngine::score_trust_deficit(const ConceptSignals& cs) const {
    return clamp01(1.0 - cs.trust);
}

double CuriosityEngine::score_model_uncertainty(const ConceptSignals& cs) const {
    if (!cs.has_model) return 0.8;
    if (!cs.model_converged) return 1.0;
    return clamp01(cs.model_loss);
}

double CuriosityEngine::score_nn_kan_conflict(const ConceptSignals& cs) const {
    return clamp01(cs.nn_kan_divergence * 3.0);  // divergence > 0.33 = max
}

double CuriosityEngine::score_topology_gap(const ConceptSignals& cs,
                                            const SystemSignals& sys) const {
    if (cs.relation_count == 0) return 1.0;  // isolated = max curiosity
    double degree = static_cast<double>(cs.relation_count);
    double avg = sys.avg_degree;
    return clamp01(1.0 - degree / (2.0 * avg + 1.0));
}

double CuriosityEngine::score_contradiction(const ConceptSignals& cs) const {
    if (!cs.has_contradictions) return 0.0;
    return clamp01(cs.contradiction_ratio * 2.0);
}

double CuriosityEngine::score_prediction_error(const ConceptSignals& cs,
                                                const SystemSignals& sys) const {
    if (cs.correction_count == 0) return 0.0;
    double avg_corrections = (sys.total_concepts > 0)
        ? static_cast<double>(sys.total_corrections) / static_cast<double>(sys.total_concepts)
        : 0.0;
    return clamp01(static_cast<double>(cs.correction_count) / (avg_corrections + 1.0));
}

double CuriosityEngine::score_novelty(const ConceptSignals& cs) const {
    return cs.novelty_score;  // already in [0,1]
}

double CuriosityEngine::score_episodic_revisit(const ConceptSignals& cs) const {
    // Low quality episodes → should revisit
    if (cs.episode_count == 0) return 0.5;  // never visited, moderate curiosity
    return clamp01(1.0 - cs.avg_episode_quality);
}

double CuriosityEngine::score_activation_anomaly(const ConceptSignals& cs) const {
    // High trust but high pain = anomaly
    if (cs.trust > 0.6 && cs.avg_edge_pain > 0.4) {
        return clamp01((cs.trust - 0.5) * cs.avg_edge_pain * 2.0);
    }
    return 0.0;
}

double CuriosityEngine::score_edge_weight_anomaly(const ConceptSignals& cs) const {
    // High variance in edge weights = interesting
    return clamp01(cs.edge_weight_variance * 4.0);
}

double CuriosityEngine::score_quality_degradation(const ConceptSignals& cs) const {
    // Low average episode quality for a concept with many episodes
    if (cs.episode_count < 3) return 0.0;
    return clamp01(1.0 - cs.avg_episode_quality);
}

double CuriosityEngine::score_cross_signal(
    const std::array<double, CURIOSITY_DIM_COUNT>& dims) const {
    size_t active = 0;
    for (size_t i = 0; i < 12; ++i) {  // exclude CROSS_SIGNAL itself
        if (dims[i] > 0.3) ++active;
    }
    if (active >= config_.cross_signal_min_dims) return 1.0;
    return 0.0;
}

std::vector<CuriosityScore> CuriosityEngine::score_concepts(
    const SystemSnapshot& snap) const
{
    std::vector<CuriosityScore> scores;
    scores.reserve(snap.concepts.size());

    for (const auto& cs : snap.concepts) {
        CuriosityScore sc;
        sc.concept_id = cs.concept_id;

        // Compute each dimension
        sc.dimension_scores[0]  = score_pain(cs);
        sc.dimension_scores[1]  = score_trust_deficit(cs);
        sc.dimension_scores[2]  = score_model_uncertainty(cs);
        sc.dimension_scores[3]  = score_nn_kan_conflict(cs);
        sc.dimension_scores[4]  = score_topology_gap(cs, snap.system);
        sc.dimension_scores[5]  = score_contradiction(cs);
        sc.dimension_scores[6]  = score_prediction_error(cs, snap.system);
        sc.dimension_scores[7]  = score_novelty(cs);
        sc.dimension_scores[8]  = score_episodic_revisit(cs);
        sc.dimension_scores[9]  = score_activation_anomaly(cs);
        sc.dimension_scores[10] = score_edge_weight_anomaly(cs);
        sc.dimension_scores[11] = score_quality_degradation(cs);
        sc.dimension_scores[12] = score_cross_signal(sc.dimension_scores);

        // Weighted sum of first 12 dimensions
        double raw = 0.0;
        for (size_t i = 0; i < 12; ++i) {
            raw += weights_[i] * sc.dimension_scores[i];
        }

        // Cross-signal bonus: when 3+ independent signals co-fire
        if (sc.dimension_scores[12] > 0.5) {
            raw *= config_.cross_signal_bonus;
        }

        // Trend boost: pain trend amplifies score
        double pain_trend = trends_.get_concept_pain_trend(cs.concept_id);
        sc.total_score = raw * (1.0 + 0.2 * pain_trend);
        sc.total_score = std::max(0.0, sc.total_score);

        // Find primary dimension (highest-scoring)
        size_t primary_idx = 0;
        double max_dim = sc.dimension_scores[0];
        for (size_t i = 1; i < CURIOSITY_DIM_COUNT; ++i) {
            if (sc.dimension_scores[i] > max_dim) {
                max_dim = sc.dimension_scores[i];
                primary_idx = i;
            }
        }
        sc.primary_dimension = static_cast<CuriosityDimension>(primary_idx);

        // Build reason string
        std::ostringstream oss;
        switch (sc.primary_dimension) {
            case CuriosityDimension::PAIN_DRIVEN:
                oss << "high pain (edge=" << cs.avg_edge_pain << ", seed=" << cs.seed_pain_ema << ")";
                break;
            case CuriosityDimension::TRUST_DEFICIT:
                oss << "low trust (" << cs.trust << ")";
                break;
            case CuriosityDimension::MODEL_UNCERTAINTY:
                oss << "model " << (cs.has_model ? (cs.model_converged ? "high loss" : "not converged") : "missing");
                break;
            case CuriosityDimension::NN_KAN_CONFLICT:
                oss << "NN/KAN divergence (" << cs.nn_kan_divergence << ")";
                break;
            case CuriosityDimension::TOPOLOGY_GAP:
                oss << "low connectivity (" << cs.relation_count << " rels)";
                break;
            case CuriosityDimension::CONTRADICTION_ALERT:
                oss << "contradictions (" << cs.contradiction_ratio << " ratio)";
                break;
            case CuriosityDimension::PREDICTION_ERROR:
                oss << "prediction errors (" << cs.correction_count << " corrections)";
                break;
            case CuriosityDimension::NOVELTY_EXPLORATION:
                oss << "novel concept (" << cs.episode_count << " episodes)";
                break;
            case CuriosityDimension::EPISODIC_REVISIT:
                oss << "low episodic quality (" << cs.avg_episode_quality << ")";
                break;
            case CuriosityDimension::ACTIVATION_ANOMALY:
                oss << "trust/pain anomaly";
                break;
            case CuriosityDimension::EDGE_WEIGHT_ANOMALY:
                oss << "edge weight variance (" << cs.edge_weight_variance << ")";
                break;
            case CuriosityDimension::QUALITY_DEGRADATION:
                oss << "quality degradation";
                break;
            case CuriosityDimension::CROSS_SIGNAL:
                oss << "cross-signal hotspot";
                break;
            default:
                oss << "unknown";
                break;
        }
        sc.reason = oss.str();

        scores.push_back(std::move(sc));
    }

    // Sort by total_score descending
    std::sort(scores.begin(), scores.end(),
        [](const CuriosityScore& a, const CuriosityScore& b) {
            return a.total_score > b.total_score;
        });

    return scores;
}

// =============================================================================
// Phase 3: PLAN — Diversity-aware seed selection
// =============================================================================

GoalType CuriosityEngine::dimension_to_goal(CuriosityDimension dim) {
    switch (dim) {
        case CuriosityDimension::PAIN_DRIVEN:
        case CuriosityDimension::PREDICTION_ERROR:
            return GoalType::CAUSAL_CHAIN;

        case CuriosityDimension::TRUST_DEFICIT:
        case CuriosityDimension::CONTRADICTION_ALERT:
            return GoalType::PROPERTY_QUERY;

        case CuriosityDimension::MODEL_UNCERTAINTY:
        case CuriosityDimension::NN_KAN_CONFLICT:
        case CuriosityDimension::CROSS_SIGNAL:
            return GoalType::EXPLORATION;

        case CuriosityDimension::TOPOLOGY_GAP:
        case CuriosityDimension::NOVELTY_EXPLORATION:
        case CuriosityDimension::EPISODIC_REVISIT:
        case CuriosityDimension::ACTIVATION_ANOMALY:
        case CuriosityDimension::EDGE_WEIGHT_ANOMALY:
        case CuriosityDimension::QUALITY_DEGRADATION:
        default:
            return GoalType::EXPLORATION;
    }
}

TriggerType CuriosityEngine::dimension_to_trigger_type(CuriosityDimension dim) {
    switch (dim) {
        case CuriosityDimension::PAIN_DRIVEN:
            return TriggerType::PAIN_CLUSTER;
        case CuriosityDimension::TRUST_DEFICIT:
            return TriggerType::TRUST_DECAY_REGION;
        case CuriosityDimension::MODEL_UNCERTAINTY:
        case CuriosityDimension::NN_KAN_CONFLICT:
            return TriggerType::MODEL_DIVERGENCE;
        case CuriosityDimension::CONTRADICTION_ALERT:
            return TriggerType::CONTRADICTION_REGION;
        case CuriosityDimension::PREDICTION_ERROR:
            return TriggerType::PREDICTION_FAILURE_ZONE;
        case CuriosityDimension::CROSS_SIGNAL:
            return TriggerType::CROSS_SIGNAL_HOTSPOT;
        case CuriosityDimension::QUALITY_DEGRADATION:
            return TriggerType::QUALITY_REGRESSION;
        case CuriosityDimension::EPISODIC_REVISIT:
            return TriggerType::EPISODIC_STALENESS;
        default:
            return TriggerType::LOW_EXPLORATION;
    }
}

void CuriosityEngine::diversify_seeds(std::vector<SeedEntry>& seeds,
                                       const LongTermMemory& ltm,
                                       size_t target) const {
    if (seeds.size() <= target) return;

    // Greedy farthest-first by graph distance approximation
    // We approximate graph distance by checking if concepts share neighbors
    std::vector<SeedEntry> selected;
    selected.reserve(target);

    // Always take the highest-scoring seed
    selected.push_back(seeds[0]);

    // Precompute neighbor sets for all candidates
    std::unordered_map<ConceptId, std::unordered_set<ConceptId>> neighbors;
    for (const auto& s : seeds) {
        auto rels = ltm.get_outgoing_relations(s.concept_id);
        auto& nset = neighbors[s.concept_id];
        for (const auto& rel : rels) {
            nset.insert(rel.target);
        }
        auto incoming = ltm.get_incoming_relations(s.concept_id);
        for (const auto& rel : incoming) {
            nset.insert(rel.source);
        }
    }

    // For each remaining slot, pick the seed that maximizes:
    //   combined = (1 - diversity_weight) * score + diversity_weight * min_distance
    while (selected.size() < target && selected.size() < seeds.size()) {
        double best_combined = -1.0;
        size_t best_idx = 0;

        for (size_t i = 0; i < seeds.size(); ++i) {
            // Skip already selected
            bool already = false;
            for (const auto& sel : selected) {
                if (sel.concept_id == seeds[i].concept_id) { already = true; break; }
            }
            if (already) continue;

            // Compute min graph distance to any selected seed
            double min_dist = 1.0;  // max distance (normalized)
            const auto& cand_neighbors = neighbors[seeds[i].concept_id];
            for (const auto& sel : selected) {
                // Distance = 0 if same, 0.33 if direct neighbor, 1.0 if unconnected
                if (cand_neighbors.count(sel.concept_id)) {
                    min_dist = std::min(min_dist, 0.33);
                } else {
                    // Check 2-hop: does the selected concept have any shared neighbors?
                    const auto& sel_neighbors = neighbors[sel.concept_id];
                    bool shared = false;
                    for (ConceptId cn : cand_neighbors) {
                        if (sel_neighbors.count(cn)) { shared = true; break; }
                    }
                    if (shared) {
                        min_dist = std::min(min_dist, 0.66);
                    }
                    // else: unconnected, min_dist stays at 1.0
                }
            }

            // Normalize score to [0,1] using the top score as reference
            double norm_score = (seeds[0].priority > 0.0)
                ? seeds[i].priority / seeds[0].priority
                : 0.0;

            double combined = (1.0 - config_.diversity_weight) * norm_score
                            + config_.diversity_weight * min_dist;

            if (combined > best_combined) {
                best_combined = combined;
                best_idx = i;
            }
        }

        selected.push_back(seeds[best_idx]);
    }

    seeds = std::move(selected);
}

SeedPlan CuriosityEngine::generate_seed_plan(
    const std::vector<CuriosityScore>& scores,
    const SystemSnapshot& snap,
    const LongTermMemory& ltm)
{
    SeedPlan plan;

    // Convert top scores to seed entries
    size_t limit = std::min(scores.size(), config_.max_seeds * 2);  // oversample for diversity
    std::vector<SeedEntry> candidates;
    candidates.reserve(limit);

    for (size_t i = 0; i < limit; ++i) {
        const auto& sc = scores[i];
        SeedEntry entry;
        entry.concept_id = sc.concept_id;
        entry.priority = sc.total_score;
        entry.primary_reason = sc.primary_dimension;
        entry.suggested_goal = dimension_to_goal(sc.primary_dimension);
        entry.reason_text = sc.reason;
        candidates.push_back(std::move(entry));
    }

    // Apply diversity-aware selection
    diversify_seeds(candidates, ltm, config_.max_seeds);

    plan.seeds = std::move(candidates);

    // Compute system health (0 = very sick, 1 = healthy)
    double health = 1.0;
    // Penalize high average model loss
    health -= 0.3 * std::clamp(snap.system.avg_model_loss, 0.0, 1.0);
    // Penalize low convergence rate
    if (snap.system.total_concepts > 0 && snap.system.converged_models > 0) {
        double convergence_rate = static_cast<double>(snap.system.converged_models)
                                / static_cast<double>(snap.system.total_concepts);
        health -= 0.2 * (1.0 - convergence_rate);
    }
    // Penalize low density
    health -= 0.2 * (1.0 - std::min(1.0, snap.system.graph_density * 100.0));
    // Penalize high correction rate
    if (snap.system.total_concepts > 0) {
        double corr_rate = static_cast<double>(snap.system.total_corrections)
                         / static_cast<double>(snap.system.total_concepts);
        health -= 0.3 * std::clamp(corr_rate * 0.1, 0.0, 1.0);
    }
    plan.system_health = std::clamp(health, 0.0, 1.0);

    // Build health summary
    std::ostringstream oss;
    oss << "health=" << plan.system_health
        << " concepts=" << snap.system.total_concepts
        << " relations=" << snap.system.total_relations
        << " density=" << snap.system.graph_density
        << " converged=" << snap.system.converged_models
        << " avg_loss=" << snap.system.avg_model_loss
        << " episodes=" << snap.system.total_episodes
        << " corrections=" << snap.system.total_corrections;
    plan.health_summary = oss.str();

    return plan;
}

// =============================================================================
// Full refresh: observe → score → plan
// =============================================================================

void CuriosityEngine::refresh(
    const LongTermMemory& ltm,
    const ConceptModelRegistry& registry,
    const EpisodicMemory& episodic,
    const ErrorCollector& error_collector,
    const std::unordered_map<ConceptId, double>& seed_pain_scores,
    const EpistemicPromotion* /*promotion*/,
    const StreamMonitor* /*monitor*/)
{
    // Phase 1: Observe
    cached_snapshot_ = observe(ltm, registry, episodic, error_collector, seed_pain_scores);

    // Update trends
    trends_.avg_model_loss.update(cached_snapshot_.system.avg_model_loss);
    trends_.graph_density.update(cached_snapshot_.system.graph_density);

    // Update per-concept pain trends
    for (const auto& cs : cached_snapshot_.concepts) {
        if (cs.seed_pain_ema > 0.0 || cs.avg_edge_pain > 0.0) {
            double pain = 0.5 * cs.avg_edge_pain + 0.5 * cs.seed_pain_ema;
            trends_.update_concept(trends_.concept_pain, cs.concept_id, pain);
        }
    }

    // Phase 2: Score
    cached_scores_ = score_concepts(cached_snapshot_);

    // Phase 3: Plan
    cached_plan_ = generate_seed_plan(cached_scores_, cached_snapshot_, ltm);
}

// =============================================================================
// Seed selection (reads cached plan)
// =============================================================================

std::vector<ConceptId> CuriosityEngine::select_seeds(size_t count) const {
    std::vector<ConceptId> result;
    size_t limit = std::min(count, cached_plan_.seeds.size());
    result.reserve(limit);
    for (size_t i = 0; i < limit; ++i) {
        result.push_back(cached_plan_.seeds[i].concept_id);
    }
    return result;
}

std::vector<SeedEntry> CuriosityEngine::select_seed_entries(size_t count) const {
    size_t limit = std::min(count, cached_plan_.seeds.size());
    return std::vector<SeedEntry>(
        cached_plan_.seeds.begin(),
        cached_plan_.seeds.begin() + static_cast<ptrdiff_t>(limit));
}

// =============================================================================
// Trigger generation (reads cached scores, clusters by dimension)
// =============================================================================

std::vector<CuriosityTrigger> CuriosityEngine::generate_triggers(ContextId ctx) const {
    std::vector<CuriosityTrigger> triggers;

    // Group high-scoring concepts by their primary dimension
    std::unordered_map<int, std::vector<const CuriosityScore*>> clusters;
    for (const auto& sc : cached_scores_) {
        if (sc.total_score < config_.trigger_threshold) break;  // sorted desc
        int dim = static_cast<int>(sc.primary_dimension);
        clusters[dim].push_back(&sc);
    }

    for (const auto& [dim_idx, members] : clusters) {
        if (members.size() < config_.min_cluster_size) continue;
        if (triggers.size() >= config_.max_triggers) break;

        auto dim = static_cast<CuriosityDimension>(dim_idx);
        TriggerType ttype = dimension_to_trigger_type(dim);

        // Collect concept IDs in cluster
        std::vector<ConceptId> concept_ids;
        double max_priority = 0.0;
        std::array<double, 13> avg_scores{};
        for (const auto* sc : members) {
            concept_ids.push_back(sc->concept_id);
            max_priority = std::max(max_priority, sc->total_score);
            for (size_t i = 0; i < CURIOSITY_DIM_COUNT; ++i) {
                avg_scores[i] += sc->dimension_scores[i];
            }
        }
        for (size_t i = 0; i < CURIOSITY_DIM_COUNT; ++i) {
            avg_scores[i] /= static_cast<double>(members.size());
        }

        // Build description
        std::ostringstream desc;
        desc << members.size() << " concepts with "
             << members[0]->reason << " (priority=" << max_priority << ")";

        triggers.emplace_back(
            ttype, ctx, concept_ids, desc.str(),
            max_priority, dim_idx, avg_scores);
    }

    return triggers;
}

// =============================================================================
// Backward-compat wrapper for ThinkingPipeline
// =============================================================================

std::vector<CuriosityTrigger> CuriosityEngine::observe_and_generate_triggers(
    const std::vector<SystemObservation>& observations)
{
    std::vector<CuriosityTrigger> triggers;

    // If we have cached scores from a recent refresh(), generate triggers from those
    if (!cached_scores_.empty()) {
        ContextId ctx = 0;
        if (!observations.empty()) ctx = observations[0].context_id;
        return generate_triggers(ctx);
    }

    // Fallback: legacy behavior for when refresh() hasn't been called
    for (const auto& obs : observations) {
        if (obs.active_concept_count > 0) {
            double ratio = static_cast<double>(obs.active_relation_count)
                         / static_cast<double>(obs.active_concept_count);
            if (ratio < 0.3) {
                triggers.emplace_back(
                    TriggerType::SHALLOW_RELATIONS,
                    obs.context_id,
                    std::vector<ConceptId>{},
                    "Many concepts activated but few relations");
            }
        }
        if (obs.active_concept_count > 0 && obs.active_concept_count < 5) {
            triggers.emplace_back(
                TriggerType::LOW_EXPLORATION,
                obs.context_id,
                std::vector<ConceptId>{},
                "Stable context with minimal variation");
        }
    }

    return triggers;
}

// =============================================================================
// Health
// =============================================================================

double CuriosityEngine::system_health() const {
    return cached_plan_.system_health;
}

} // namespace brain19
