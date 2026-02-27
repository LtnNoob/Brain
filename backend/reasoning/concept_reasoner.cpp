#include "concept_reasoner.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace brain19 {

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

ConceptReasoner::ConceptReasoner(
    const LongTermMemory& ltm,
    ConceptModelRegistry& registry,
    EmbeddingManager& embeddings,
    ReasonerConfig config
)
    : ltm_(ltm)
    , registry_(registry)
    , embeddings_(embeddings)
    , config_(config)
{
    chain_kan_.initialize();
    initialize_chain_ctx_projection();
}

void ConceptReasoner::initialize_chain_ctx_projection() {
    // Xavier init for 32→16 projection
    double scale = std::sqrt(2.0 / static_cast<double>(32 + CORE_DIM));
    for (size_t i = 0; i < 32 * CORE_DIM; ++i) {
        double x = std::sin(static_cast<double>(i * 41 + 67)) * 43758.5453;
        x = x - std::floor(x);
        chain_ctx_proj_W_[i] = (x * 2.0 - 1.0) * scale;
    }
    chain_ctx_proj_b_.fill(0.0);
}

// ---------------------------------------------------------------------------
// project_chain_to_core — 32D chain state → 16D core vector
// ---------------------------------------------------------------------------
CoreVec ConceptReasoner::project_chain_to_core(
    const std::array<double, 32>& chain_state) const
{
    CoreVec result{};
    for (size_t i = 0; i < CORE_DIM; ++i) {
        double sum = chain_ctx_proj_b_[i];
        for (size_t j = 0; j < 32; ++j) {
            sum += chain_ctx_proj_W_[i * 32 + j] * chain_state[j];
        }
        result[i] = std::tanh(sum);  // bounded [-1, 1]
    }
    return result;
}

// ---------------------------------------------------------------------------
// score_edge — local CM scores one edge
// ---------------------------------------------------------------------------
double ConceptReasoner::score_edge(
    ConceptId source, ConceptId target,
    RelationType type, const FlexEmbedding& ctx) const
{
    ConceptModel* model = registry_.get_model(source);
    if (!model) {
        auto emb_src = embeddings_.concept_embeddings().get_or_default(source);
        auto emb_tgt = embeddings_.concept_embeddings().get_or_default(target);
        return 0.5 * (1.0 + core_similarity(emb_src, emb_tgt));
    }

    const FlexEmbedding& rel_emb = embeddings_.get_relation_embedding(type);

    FlexEmbedding ctx_mixed = ctx;
    FlexEmbedding target_emb = embeddings_.concept_embeddings().get_or_default(target);
    for (size_t i = 0; i < CORE_DIM; ++i) {
        ctx_mixed.core[i] = (1.0 - config_.context_alpha) * ctx.core[i]
                          + config_.context_alpha * target_emb.core[i];
    }

    FlexEmbedding concept_from = embeddings_.concept_embeddings().get_or_default(source);
    FlexEmbedding concept_to = embeddings_.concept_embeddings().get_or_default(target);

    return model->predict_refined(rel_emb, ctx_mixed, concept_from, concept_to);
}

// ---------------------------------------------------------------------------
// is_causal_relation — does this edge shift the reasoning focus?
// ---------------------------------------------------------------------------
bool ConceptReasoner::is_causal_relation(RelationType type) {
    switch (type) {
        case RelationType::CAUSES:
        case RelationType::ENABLES:
        case RelationType::REQUIRES:
        case RelationType::PRODUCES:
        case RelationType::IMPLIES:
        case RelationType::SUPPORTS:
            return true;
        default:
            return false;
    }
}

// ---------------------------------------------------------------------------
// relation_reasoning_weight — tiered weights for logical reasoning
// ---------------------------------------------------------------------------
double ConceptReasoner::relation_reasoning_weight(RelationType type) {
    switch (type) {
        case RelationType::CAUSES:
        case RelationType::ENABLES:
        case RelationType::REQUIRES:
        case RelationType::PRODUCES:
        case RelationType::IMPLIES:
        case RelationType::SUPPORTS:
            return 1.3;

        case RelationType::IS_A:
        case RelationType::HAS_PROPERTY:
        case RelationType::DERIVED_FROM:
            return 1.0;

        case RelationType::PART_OF:
        case RelationType::HAS_PART:
        case RelationType::CONTRADICTS:
            return 0.7;

        case RelationType::SIMILAR_TO:
        case RelationType::TEMPORAL_BEFORE:
        case RelationType::CUSTOM:
        default:
            return 0.5;
    }
}

// ---------------------------------------------------------------------------
// build_composition_features — 90D input for ConvergencePort
// ---------------------------------------------------------------------------
std::array<double, 90> ConceptReasoner::build_composition_features(
    ConceptId current, ConceptId target, RelationType rel,
    const FlexEmbedding& ctx, const PredictFeatures& pf) const
{
    std::array<double, 90> features{};

    auto emb_current = embeddings_.concept_embeddings().get_or_default(current);
    auto emb_target = embeddings_.concept_embeddings().get_or_default(target);
    auto rel_emb = embeddings_.get_relation_embedding(rel);

    // [0..15] concept_emb(current).core
    for (size_t i = 0; i < CORE_DIM; ++i)
        features[i] = emb_current.core[i];

    // [16..31] concept_emb(target).core
    for (size_t i = 0; i < CORE_DIM; ++i)
        features[16 + i] = emb_target.core[i];

    // [32..47] relation_emb.core
    for (size_t i = 0; i < CORE_DIM; ++i)
        features[32 + i] = rel_emb.core[i];

    // [48..63] ctx_emb.core
    for (size_t i = 0; i < CORE_DIM; ++i)
        features[48 + i] = ctx.core[i];

    // [64..69] local CM intermediate features
    features[64] = pf.multihead_scores[0];
    features[65] = pf.multihead_scores[1];
    features[66] = pf.multihead_scores[2];
    features[67] = pf.multihead_scores[3];
    features[68] = pf.bilinear_score;
    features[69] = pf.dim_fraction;

    // [70..73] structural features
    double rel_weight = relation_reasoning_weight(rel);
    double is_causal_val = is_causal_relation(rel) ? 1.0 : 0.0;

    double epistemic_trust = 0.5;
    auto cinfo = ltm_.retrieve_concept(target);
    if (cinfo) {
        epistemic_trust = cinfo->epistemic.trust;
    }

    double degree = 0.0;
    auto out_rels = ltm_.get_outgoing_relations(target);
    degree = std::log(1.0 + static_cast<double>(out_rels.size())) / 5.0;

    features[70] = rel_weight;
    features[71] = epistemic_trust;
    features[72] = degree;
    features[73] = is_causal_val;

    // [74..89] padding zeros (already initialized to 0)

    return features;
}

// ---------------------------------------------------------------------------
// forward_chain_state — run features through CM's ConvergencePort
// ---------------------------------------------------------------------------
std::array<double, 32> ConceptReasoner::forward_chain_state(
    ConceptId concept_id,
    const std::array<double, 90>& features,
    const std::array<double, 32>& prev_state) const
{
    double input[122];
    for (size_t i = 0; i < 90; ++i)
        input[i] = features[i];
    for (size_t i = 0; i < 32; ++i)
        input[90 + i] = prev_state[i];

    std::array<double, 32> output{};

    ConceptModel* cm = registry_.get_model(concept_id);
    if (cm) {
        cm->forward_convergence(input, output.data());
    }

    return output;
}

// ---------------------------------------------------------------------------
// score_candidates — focus-gated scoring + ConvergencePort composition
// ---------------------------------------------------------------------------
std::vector<ConceptReasoner::ScoredCandidate> ConceptReasoner::score_candidates(
    ConceptId current, const std::vector<FocusEntry>& focus_stack,
    const FlexEmbedding& ctx,
    const std::unordered_set<ConceptId>& visited,
    const std::unordered_set<uint16_t>& used_rels,
    const std::array<double, 32>& chain_state,
    ConceptId seed_id, size_t step_index) const
{
    std::vector<ScoredCandidate> candidates;
    const double fgw = config_.focus_gate_weight;
    const double ccw = config_.chain_coherence_weight;
    const bool compose = config_.enable_composition;

    // Seed-anchor: precompute seed embedding for distance penalty
    FlexEmbedding seed_emb;
    double anchor_strength = 0.0;
    if (config_.seed_anchor_weight > 0.0 && seed_id != 0) {
        seed_emb = embeddings_.concept_embeddings().get_or_default(seed_id);
        // Decay anchor strength over steps (early steps anchor more)
        anchor_strength = config_.seed_anchor_weight *
            std::exp(-config_.seed_anchor_decay * static_cast<double>(step_index));
    }

    auto process_edge = [&](ConceptId target, RelationType type,
                             double cm_score, bool outgoing) {
        bool causal = is_causal_relation(type);

        // Non-causal edges: focus stack gate
        if (!causal && !focus_stack.empty()) {
            const FlexEmbedding& rel_emb = embeddings_.get_relation_embedding(type);
            FlexEmbedding t_emb = embeddings_.concept_embeddings().get_or_default(target);

            double best_fscore = 0.0;
            for (const auto& fe : focus_stack) {
                ConceptModel* fcm = registry_.get_model(fe.id);
                if (!fcm) continue;
                FlexEmbedding f_emb = embeddings_.concept_embeddings().get_or_default(fe.id);
                double fs = fcm->predict_refined(rel_emb, fe.emb, f_emb, t_emb);
                if (fs > best_fscore) best_fscore = fs;
            }

            if (best_fscore < config_.focus_min_gate) return;
            cm_score = (1.0 - fgw) * cm_score + fgw * best_fscore;
        }

        double score = cm_score * relation_reasoning_weight(type);

        if (!used_rels.count(static_cast<uint16_t>(type))) {
            score += config_.diversity_bonus;
        }

        // Seed-anchor penalty: penalize candidates far from seed topic
        if (anchor_strength > 0.0) {
            FlexEmbedding t_emb = embeddings_.concept_embeddings().get_or_default(target);
            double sim = core_similarity(seed_emb, t_emb);
            // sim in [-1, 1], map to penalty: low similarity → higher penalty
            // penalty = anchor_strength * (1 - (sim+1)/2) = anchor_strength * (1 - sim) / 2
            double penalty = anchor_strength * (1.0 - sim) * 0.5;
            score -= penalty;
        }

        // Composition: simulate chain state + ChainKAN coherence
        std::array<double, 32> sim_state{};
        double coherence = 0.0;

        if (compose) {
            ConceptModel* cm_model = registry_.get_model(current);
            PredictFeatures pf;
            if (cm_model) {
                const FlexEmbedding& rel_emb = embeddings_.get_relation_embedding(type);
                FlexEmbedding ctx_mixed = ctx;
                FlexEmbedding target_emb = embeddings_.concept_embeddings().get_or_default(target);
                for (size_t i = 0; i < CORE_DIM; ++i) {
                    ctx_mixed.core[i] = (1.0 - config_.context_alpha) * ctx.core[i]
                                      + config_.context_alpha * target_emb.core[i];
                }
                FlexEmbedding from_emb = embeddings_.concept_embeddings().get_or_default(current);
                pf = cm_model->predict_refined_with_features(rel_emb, ctx_mixed, from_emb, target_emb);
            } else {
                pf.bilinear_score = cm_score;
                pf.dim_fraction = 0.0;
            }

            auto features = build_composition_features(current, target, type, ctx, pf);
            sim_state = forward_chain_state(target, features, chain_state);
            coherence = chain_kan_.evaluate(chain_state.data(), sim_state.data());

            // Blend local score with chain coherence
            score = (1.0 - ccw) * score + ccw * coherence;
        }

        candidates.push_back({target, type, score, outgoing, causal, sim_state, coherence});
    };

    // Outgoing relations
    auto outgoing = ltm_.get_outgoing_relations(current);
    for (const auto& rel : outgoing) {
        if (visited.count(rel.target)) continue;
        auto cinfo = ltm_.retrieve_concept(rel.target);
        if (!cinfo || cinfo->epistemic.is_invalidated()) continue;
        double cm = score_edge(current, rel.target, rel.type, ctx);
        cm *= std::pow(rel.weight, config_.relation_weight_power);
        process_edge(rel.target, rel.type, cm, true);
    }

    // Incoming relations
    auto incoming = ltm_.get_incoming_relations(current);
    for (const auto& rel : incoming) {
        if (visited.count(rel.source)) continue;
        auto cinfo = ltm_.retrieve_concept(rel.source);
        if (!cinfo || cinfo->epistemic.is_invalidated()) continue;
        double cm = score_edge(current, rel.source, rel.type, ctx);
        cm *= std::pow(rel.weight, config_.relation_weight_power);
        cm *= config_.incoming_discount;
        process_edge(rel.source, rel.type, cm, false);
    }

    std::sort(candidates.begin(), candidates.end(),
        [](const ScoredCandidate& a, const ScoredCandidate& b) {
            return a.score > b.score;
        });

    if (candidates.size() > config_.max_candidates) {
        candidates.resize(config_.max_candidates);
    }

    return candidates;
}

// ---------------------------------------------------------------------------
// update_context — EMA of concept embeddings + chain state feedback
// ---------------------------------------------------------------------------
FlexEmbedding ConceptReasoner::update_context(
    const FlexEmbedding& ctx, ConceptId new_concept,
    const std::array<double, 32>& chain_state) const
{
    FlexEmbedding emb = embeddings_.concept_embeddings().get_or_default(new_concept);
    FlexEmbedding result;

    if (config_.enable_composition && config_.chain_ctx_blend > 0.0) {
        // Project chain state to core space and blend into context
        CoreVec chain_proj = project_chain_to_core(chain_state);
        double beta = config_.chain_ctx_blend;
        double alpha = config_.context_alpha;
        // Renormalize: alpha + (1-alpha-beta) + beta = 1
        double ctx_weight = 1.0 - alpha - beta;
        if (ctx_weight < 0.0) {
            // If alpha + beta > 1, scale them down
            alpha = alpha / (alpha + beta);
            beta = 1.0 - alpha;
            ctx_weight = 0.0;
        }
        for (size_t i = 0; i < CORE_DIM; ++i) {
            result.core[i] = alpha * emb.core[i]
                           + ctx_weight * ctx.core[i]
                           + beta * chain_proj[i];
        }
    } else {
        // Original EMA (no composition feedback)
        for (size_t i = 0; i < CORE_DIM; ++i) {
            result.core[i] = config_.context_alpha * emb.core[i]
                           + (1.0 - config_.context_alpha) * ctx.core[i];
        }
    }
    return result;
}

// ---------------------------------------------------------------------------
// reason_from(seed) — focus-shifting reasoning loop with CM composition
// ---------------------------------------------------------------------------
ReasoningChain ConceptReasoner::reason_from(ConceptId seed) const {
    ReasoningChain chain;

    FlexEmbedding ctx = embeddings_.concept_embeddings().get_or_default(seed);
    const FlexEmbedding seed_emb_val = embeddings_.concept_embeddings().get_or_default(seed);

    std::vector<FocusEntry> focus_stack;
    focus_stack.push_back({seed, ctx});

    std::unordered_set<ConceptId> visited;
    std::unordered_set<uint16_t> used_rels;
    visited.insert(seed);

    // Initialize chain state
    std::array<double, 32> chain_state{};

    if (config_.enable_composition) {
        PredictFeatures seed_pf;
        seed_pf.bilinear_score = 0.5;
        auto seed_features = build_composition_features(seed, seed, RelationType::CUSTOM, ctx, seed_pf);
        chain_state = forward_chain_state(seed, seed_features, chain_state);
    }

    ReasoningStep seed_step;
    seed_step.concept_id = seed;
    seed_step.relation_type = RelationType::CUSTOM;
    seed_step.confidence = 1.0;
    seed_step.is_outgoing = true;
    seed_step.focus_shifted = false;
    seed_step.chain_state = chain_state;
    seed_step.coherence_score = 0.0;
    seed_step.seed_similarity = 1.0;
    chain.steps.push_back(seed_step);

    ConceptId current = seed;

    for (size_t step = 0; step < config_.max_steps; ++step) {
        auto candidates = score_candidates(current, focus_stack, ctx,
                                            visited, used_rels, chain_state,
                                            seed, step);

        if (candidates.empty()) {
            std::vector<FocusEntry> empty_stack;
            candidates = score_candidates(current, empty_stack, ctx,
                                           visited, used_rels, chain_state,
                                           seed, step);
        }

        if (candidates.empty()) break;
        if (candidates[0].score < config_.min_confidence) break;

        const auto& best = candidates[0];

        // Coherence-gated termination: stop if ChainKAN says this transition is incoherent
        if (config_.enable_composition && step > 0 &&
            best.coherence < config_.min_coherence_gate) {
            break;
        }

        // Update chain state from winner
        if (config_.enable_composition) {
            chain_state = best.simulated_state;
        }

        // Update EMA context (now includes chain state feedback)
        ctx = update_context(ctx, best.target, chain_state);

        // Focus shift
        bool shifted = best.is_causal;
        if (shifted) {
            FlexEmbedding new_emb = embeddings_.concept_embeddings().get_or_default(best.target);
            focus_stack.push_back({best.target, new_emb});
            if (focus_stack.size() > 3) {
                focus_stack.erase(focus_stack.begin());
            }
        }

        ReasoningStep rs;
        rs.concept_id = best.target;
        rs.relation_type = best.rel;
        rs.confidence = best.score;
        rs.is_outgoing = best.outgoing;
        rs.focus_shifted = shifted;
        rs.chain_state = chain_state;
        rs.coherence_score = best.coherence;

        // Compute seed similarity for chain validation
        {
            FlexEmbedding step_emb = embeddings_.concept_embeddings().get_or_default(best.target);
            rs.seed_similarity = core_similarity(seed_emb_val, step_emb);
        }

        // Store runner-up state for hard negative mining
        if (candidates.size() >= 2 && config_.enable_composition) {
            rs.runner_up_state = candidates[1].simulated_state;
            rs.has_runner_up = true;
        }

        chain.steps.push_back(rs);

        // Early termination: stop if seed_similarity below threshold for N consecutive steps
        if (config_.enable_chain_validation && chain.steps.size() >= 3) {
            size_t consecutive_low = 0;
            for (size_t s = chain.steps.size(); s >= 2; --s) {
                if (chain.steps[s - 1].seed_similarity < config_.min_seed_similarity) {
                    ++consecutive_low;
                } else {
                    break;
                }
            }
            if (consecutive_low >= config_.max_consecutive_drops) {
                break;  // Will truncate below
            }
        }

        current = best.target;
        visited.insert(current);
        used_rels.insert(static_cast<uint16_t>(best.rel));
    }

    // Best-prefix truncation: remove drifted tail
    if (config_.enable_chain_validation && chain.steps.size() >= 3) {
        // Find last step where seed_similarity was above minimum
        size_t best_prefix = chain.steps.size() - 1;
        for (size_t s = chain.steps.size() - 1; s >= 2; --s) {
            if (chain.steps[s].seed_similarity < config_.min_seed_similarity) {
                best_prefix = s - 1;
            } else {
                break;
            }
        }
        // Truncate if we found a better prefix (keep at least seed + 1 step)
        if (best_prefix < chain.steps.size() - 1 && best_prefix >= 1) {
            chain.steps.resize(best_prefix + 1);
        }
    }

    // avg_confidence (exclude seed)
    if (chain.steps.size() > 1) {
        double sum = 0.0;
        for (size_t i = 1; i < chain.steps.size(); ++i) {
            sum += chain.steps[i].confidence;
        }
        chain.avg_confidence = sum / static_cast<double>(chain.steps.size() - 1);
    }

    return chain;
}

// ---------------------------------------------------------------------------
// reason_from(seeds) — run each seed, return best chain
// ---------------------------------------------------------------------------
ReasoningChain ConceptReasoner::reason_from(const std::vector<ConceptId>& seeds) const {
    ReasoningChain best;
    double best_quality = -1.0;
    for (ConceptId seed : seeds) {
        auto chain = reason_from(seed);
        double q = compute_chain_quality(chain);
        if (q > best_quality) {
            best = std::move(chain);
            best_quality = q;
        }
    }
    return best;
}

// ---------------------------------------------------------------------------
// compute_chain_quality — holistic chain quality metric
// ---------------------------------------------------------------------------
double ConceptReasoner::compute_chain_quality(const ReasoningChain& chain) const {
    if (chain.steps.size() < 2) return 0.0;

    // Factor 1: average confidence (already computed)
    double conf = chain.avg_confidence;

    // Factor 2: length reward (log scale, diminishing returns)
    double length_reward = std::log(1.0 + static_cast<double>(chain.steps.size())) / 3.0;

    // Factor 3: topical coherence — mean pairwise core_similarity across chain
    double coherence_sum = 0.0;
    size_t pair_count = 0;
    for (size_t i = 0; i < chain.steps.size(); ++i) {
        auto emb_i = embeddings_.concept_embeddings().get_or_default(chain.steps[i].concept_id);
        for (size_t j = i + 1; j < chain.steps.size(); ++j) {
            auto emb_j = embeddings_.concept_embeddings().get_or_default(chain.steps[j].concept_id);
            coherence_sum += core_similarity(emb_i, emb_j);
            ++pair_count;
        }
    }
    double topical_coherence = (pair_count > 0)
        ? (coherence_sum / static_cast<double>(pair_count) + 1.0) / 2.0  // normalize [-1,1] to [0,1]
        : 0.5;

    // Factor 4: min coherence score (weakest link, PURE-inspired)
    double min_coherence = 1.0;
    for (size_t i = 1; i < chain.steps.size(); ++i) {
        if (chain.steps[i].coherence_score > 0.0 &&
            chain.steps[i].coherence_score < min_coherence) {
            min_coherence = chain.steps[i].coherence_score;
        }
    }
    if (min_coherence > 0.99) min_coherence = 0.5;  // no valid coherence scores

    // Combined: geometric mean of factors
    return std::pow(conf * length_reward * topical_coherence * min_coherence, 0.25);
}

// ---------------------------------------------------------------------------
// train_composition — contrastive training of ChainKAN + ConvergencePorts
// ---------------------------------------------------------------------------
ChainTrainingResult ConceptReasoner::train_composition(
    const std::vector<ReasoningChain>& chains,
    const ChainTrainingConfig& tcfg)
{
    ChainTrainingResult result;
    result.chains_used = chains.size();

    if (chains.empty()) return result;

    // ── Step 1: Compute chain-terminal quality for each chain ──
    std::vector<double> qualities(chains.size());
    for (size_t c = 0; c < chains.size(); ++c) {
        qualities[c] = compute_chain_quality(chains[c]);
    }

    // Sort by quality to find thresholds
    std::vector<double> sorted_q = qualities;
    std::sort(sorted_q.begin(), sorted_q.end());
    double good_threshold = sorted_q[static_cast<size_t>(
        sorted_q.size() * tcfg.good_chain_threshold)];
    double bad_threshold = sorted_q[static_cast<size_t>(
        sorted_q.size() * tcfg.bad_chain_threshold)];

    // ── Step 2: Collect ChainKAN training samples ──
    // Good chains → target based on quality, bad chains → low target
    // Hard negatives: use runner-up candidate states (near-miss, more informative)
    // Fallback: random bad chain states when no runner-up available
    std::vector<ChainKAN::Sample> samples;

    // Collect bad chain states as fallback negatives
    std::vector<std::array<double, 32>> bad_states;
    for (size_t c = 0; c < chains.size(); ++c) {
        if (qualities[c] <= bad_threshold) {
            for (size_t i = 1; i < chains[c].steps.size(); ++i) {
                bad_states.push_back(chains[c].steps[i].chain_state);
            }
        }
    }

    size_t neg_idx = 0;

    for (size_t c = 0; c < chains.size(); ++c) {
        const auto& chain = chains[c];
        if (chain.steps.size() < 2) continue;

        double quality = qualities[c];
        double target = 0.2 + 0.6 * std::min(1.0, quality / 0.7);

        for (size_t i = 1; i < chain.steps.size(); ++i) {
            // Positive sample: actual transition with chain-quality-based target
            ChainKAN::Sample pos;
            pos.prev = chain.steps[i - 1].chain_state;
            pos.next = chain.steps[i].chain_state;
            pos.target = target;
            samples.push_back(pos);

            // Hard negative: runner-up candidate (the path not taken)
            // This is much more informative than random negatives because
            // it teaches "why THIS transition, not THAT one"
            if (quality > good_threshold && chain.steps[i].has_runner_up) {
                ChainKAN::Sample neg;
                neg.prev = chain.steps[i - 1].chain_state;
                neg.next = chain.steps[i].runner_up_state;
                neg.target = std::max(0.1, target - 0.3);  // slightly worse than winner
                samples.push_back(neg);
            }
            // Fallback: random bad chain state when no runner-up
            else if (quality > good_threshold && !bad_states.empty()) {
                ChainKAN::Sample neg;
                neg.prev = chain.steps[i - 1].chain_state;
                neg.next = bad_states[neg_idx % bad_states.size()];
                neg.target = 0.15;
                samples.push_back(neg);
                neg_idx++;
            }
        }
    }

    result.samples_collected = samples.size();
    if (samples.empty()) return result;

    // ── Step 3: Measure initial loss ──
    double init_loss = 0.0;
    for (const auto& s : samples) {
        double pred = chain_kan_.evaluate(s.prev.data(), s.next.data());
        double err = pred - s.target;
        init_loss += 0.5 * err * err;
    }
    result.initial_kan_loss = init_loss / static_cast<double>(samples.size());

    // ── Step 4: Train ChainKAN ──
    chain_kan_.train(samples, tcfg.learning_rate, tcfg.kan_epochs);

    // Measure final loss
    double final_loss = 0.0;
    for (const auto& s : samples) {
        double pred = chain_kan_.evaluate(s.prev.data(), s.next.data());
        double err = pred - s.target;
        final_loss += 0.5 * err * err;
    }
    result.final_kan_loss = final_loss / static_cast<double>(samples.size());

    // ── Step 5: Fine-tune ConvergencePorts via backward_convergence ──
    // For each transition in good chains, push gradient from ChainKAN
    // through the ConvergencePort to teach it to produce coherent states
    size_t ports_updated = 0;

    for (size_t epoch = 0; epoch < tcfg.convergence_epochs; ++epoch) {
        for (size_t c = 0; c < chains.size(); ++c) {
            if (qualities[c] <= bad_threshold) continue;  // only train on decent chains
            const auto& chain = chains[c];
            if (chain.steps.size() < 3) continue;

            double target = 0.2 + 0.6 * std::min(1.0, qualities[c] / 0.7);

            for (size_t i = 1; i < chain.steps.size(); ++i) {
                ConceptId cid = chain.steps[i].concept_id;
                ConceptModel* cm = registry_.get_model(cid);
                if (!cm) continue;

                // Reconstruct the 122D input that produced this chain state
                ConceptId prev_cid = chain.steps[i - 1].concept_id;
                FlexEmbedding ctx_approx = embeddings_.concept_embeddings().get_or_default(prev_cid);
                PredictFeatures pf;
                pf.bilinear_score = chain.steps[i].confidence;
                auto features = build_composition_features(
                    prev_cid, cid, chain.steps[i].relation_type, ctx_approx, pf);

                double input_122[122];
                for (size_t k = 0; k < 90; ++k) input_122[k] = features[k];
                for (size_t k = 0; k < 32; ++k) input_122[90 + k] = chain.steps[i - 1].chain_state[k];

                // Compute gradient of ChainKAN w.r.t. new_state
                // Numerical gradient: d(loss)/d(state[j]) ≈ (loss(state+eps) - loss(state-eps)) / 2eps
                // This is simpler and more robust than analytical backprop through the KAN
                double grad_state[32];
                constexpr double eps = 1e-4;

                double base_pred = chain_kan_.evaluate(
                    chain.steps[i - 1].chain_state.data(),
                    chain.steps[i].chain_state.data());
                double base_loss_grad = base_pred - target;  // d(MSE)/d(pred) * d(pred)/d(state)

                // Efficient: use finite differences on the ChainKAN output
                std::array<double, 32> state_plus = chain.steps[i].chain_state;
                for (size_t j = 0; j < 32; ++j) {
                    double orig = state_plus[j];

                    state_plus[j] = orig + eps;
                    double pred_plus = chain_kan_.evaluate(
                        chain.steps[i - 1].chain_state.data(), state_plus.data());

                    state_plus[j] = orig - eps;
                    double pred_minus = chain_kan_.evaluate(
                        chain.steps[i - 1].chain_state.data(), state_plus.data());

                    state_plus[j] = orig;

                    // d(loss)/d(state[j]) = (pred - target) * d(pred)/d(state[j])
                    double dpred_dstate = (pred_plus - pred_minus) / (2.0 * eps);
                    grad_state[j] = base_loss_grad * dpred_dstate;
                }

                // Push gradient through ConvergencePort via backward_convergence
                cm->backward_convergence(input_122, chain.steps[i].chain_state.data(),
                                          grad_state, tcfg.convergence_lr);
                if (epoch == 0) ports_updated++;
            }
        }
    }

    result.convergence_ports_updated = ports_updated;
    return result;
}

// ---------------------------------------------------------------------------
// ReasoningChain helpers
// ---------------------------------------------------------------------------
std::vector<ConceptId> ReasoningChain::concept_sequence() const {
    std::vector<ConceptId> seq;
    seq.reserve(steps.size());
    for (const auto& s : steps) {
        seq.push_back(s.concept_id);
    }
    return seq;
}

std::vector<RelationType> ReasoningChain::relation_sequence() const {
    std::vector<RelationType> seq;
    if (steps.size() <= 1) return seq;
    seq.reserve(steps.size() - 1);
    for (size_t i = 1; i < steps.size(); ++i) {
        seq.push_back(steps[i].relation_type);
    }
    return seq;
}

} // namespace brain19
