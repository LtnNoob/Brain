#include "graph_reasoner.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace brain19 {

// =============================================================================
// Construction
// =============================================================================

GraphReasoner::GraphReasoner(
    const LongTermMemory& ltm,
    ConceptModelRegistry& registry,
    EmbeddingManager& embeddings,
    GraphReasonerConfig config
)
    : ltm_(ltm)
    , registry_(registry)
    , embeddings_(embeddings)
    , config_(config)
{
    chain_kan_.initialize();
    initialize_chain_ctx_projection();
}

void GraphReasoner::set_logger(ReasoningLogger* logger) {
    logger_ = logger;
}

void GraphReasoner::initialize_chain_ctx_projection() {
    // Xavier init for OUTPUT_DIM->CORE_DIM projection
    double scale = std::sqrt(2.0 / static_cast<double>(convergence::OUTPUT_DIM + CORE_DIM));
    for (size_t i = 0; i < convergence::OUTPUT_DIM * CORE_DIM; ++i) {
        double x = std::sin(static_cast<double>(i * 41 + 67)) * 43758.5453;
        x = x - std::floor(x);
        chain_ctx_proj_W_[i] = (x * 2.0 - 1.0) * scale;
    }
    chain_ctx_proj_b_.fill(0.0);
}

CoreVec GraphReasoner::project_chain_to_core(
    const std::array<double, convergence::OUTPUT_DIM>& chain_state) const
{
    CoreVec result{};
    for (size_t i = 0; i < CORE_DIM; ++i) {
        double sum = chain_ctx_proj_b_[i];
        for (size_t j = 0; j < convergence::OUTPUT_DIM; ++j)
            sum += chain_ctx_proj_W_[i * convergence::OUTPUT_DIM + j] * chain_state[j];
        result[i] = std::tanh(sum);
    }
    return result;
}

// =============================================================================
// forward_edge --- The core transformation
// =============================================================================
//
// Instead of predict() -> scalar -> score -> pick best:
//
// 1. v = W * activation.core + b       (concept transforms the activation)
// 2. v_mod[i] = v[i] * (1 + alpha * rel_emb[i])  (relation modulates per dim)
// 3. output.core = tanh(v_mod)          (bounded nonlinearity)
// 4. output.detail = input.detail       (passthrough, Phase 1)
// 5. transform_quality = |output| / |input|  (magnitude preservation)
// 6. coherence = cosine(output, target_emb)  (goal alignment)
//

EdgeResult GraphReasoner::forward_edge(
    ConceptId source, ConceptId target,
    RelationType rel_type,
    const Activation& input,
    const FlexEmbedding& /*ctx*/) const
{
    EdgeResult result;

    ConceptModel* model = registry_.get_model(source);
    const FlexEmbedding& rel_emb = embeddings_.get_relation_embedding(rel_type);
    FlexEmbedding source_emb = embeddings_.concept_embeddings().get_or_default(source);
    FlexEmbedding target_emb = embeddings_.concept_embeddings().get_or_default(target);

    CoreVec nn_output{};   // NN path result (before gating)

    if (model) {
        // =================================================================
        // NN PATH: W*activation.core + b → relation modulation → tanh
        // (Pattern matching, statistical correlations — visual cortex analog)
        // =================================================================

        const CoreMat& W = model->weights();
        const CoreVec& b = model->bias();
        CoreVec v{};
        for (size_t i = 0; i < CORE_DIM; ++i) {
            double sum = b[i];
            for (size_t j = 0; j < CORE_DIM; ++j)
                sum += W[i * CORE_DIM + j] * input.core()[j];
            v[i] = sum;
        }
        result.dimensional_contrib = v;

        // Relation modulation per dimension
        constexpr double rel_alpha = 0.3;
        CoreVec v_mod{};
        for (size_t i = 0; i < CORE_DIM; ++i)
            v_mod[i] = v[i] * (1.0 + rel_alpha * rel_emb.core[i]);

        // tanh nonlinearity
        for (size_t i = 0; i < CORE_DIM; ++i)
            nn_output[i] = std::tanh(v_mod[i]);

        // NN quality: magnitude preservation of W*x+b path
        double nn_mag = 0.0;
        for (size_t i = 0; i < CORE_DIM; ++i)
            nn_mag += nn_output[i] * nn_output[i];
        nn_mag = std::sqrt(nn_mag);
        double in_mag_nn = input.core_magnitude();
        double nn_ratio = in_mag_nn > 1e-12 ? nn_mag / in_mag_nn : 0.0;
        if (nn_ratio > 1e-12) {
            double lr = std::log(nn_ratio);
            result.nn_quality = std::exp(-lr * lr / 4.5);
        }

        // =================================================================
        // KAN PATH: MultiHeadBilinear + FlexKAN → gating value
        // (Reasoning, exact ops, abstraction — prefrontal cortex analog)
        // The KAN evaluates: "is this particular transform sensible?"
        // =================================================================

        // MultiHeadBilinear: assess transform from multiple angles
        // Uses input-as-embedding vs output-as-embedding
        FlexEmbedding input_as_emb = input.to_embedding();
        FlexEmbedding output_as_emb;
        output_as_emb.core = nn_output;
        output_as_emb.detail = input.detail();

        std::array<double, MultiHeadBilinear::K> mh_scores;
        model->multihead_compute(input_as_emb, output_as_emb, mh_scores);

        // Bilinear score: dot(target_emb, v) for direction assessment
        double bilinear_z = 0.0;
        for (size_t i = 0; i < CORE_DIM; ++i)
            bilinear_z += target_emb.core[i] * v[i];
        double bilinear_score = 1.0 / (1.0 + std::exp(-bilinear_z));

        double dim_fraction = static_cast<double>(
            std::min(source_emb.detail.size(), target_emb.detail.size())) / static_cast<double>(FlexConfig::MAX_DIM - CORE_DIM);

        // FlexKAN: [4 head scores, bilinear_score, dim_fraction] → gate ∈ (0,1)
        std::array<double, FlexKAN::INPUT_DIM> kan_input;
        for (size_t i = 0; i < MultiHeadBilinear::K; ++i)
            kan_input[i] = 1.0 / (1.0 + std::exp(-mh_scores[i]));
        kan_input[4] = bilinear_score;
        kan_input[5] = dim_fraction;

        double kan_gate = model->kan_evaluate(kan_input);
        result.kan_gate = kan_gate;
        result.kan_quality = kan_gate;  // Higher gate = KAN approves the transform

        // =================================================================
        // COMBINATION: Additive residual gating
        // gate=1 → full NN transform (KAN approves)
        // gate=0 → input passes through unchanged (no signal loss)
        // Prevents activation vanishing in long chains
        // =================================================================

        CoreVec gated_output{};
        for (size_t i = 0; i < CORE_DIM; ++i)
            gated_output[i] = kan_gate * nn_output[i] + (1.0 - kan_gate) * input.core()[i];

        result.output = Activation::from_embedding(FlexEmbedding{});
        result.output.core_mut() = gated_output;
        result.output.detail_mut() = input.detail();

    } else {
        // No model: blend input with target embedding (fallback)
        CoreVec blended{};
        for (size_t i = 0; i < CORE_DIM; ++i)
            blended[i] = 0.7 * input.core()[i] + 0.3 * target_emb.core[i];
        result.output = Activation::from_embedding(FlexEmbedding{});
        result.output.core_mut() = blended;
        result.output.detail_mut() = input.detail();
        result.dimensional_contrib.fill(0.0);
        nn_output = blended;
        result.nn_quality = 0.3;
        result.kan_quality = 0.5;
        result.kan_gate = 1.0;
    }

    // Transform quality: magnitude preservation of final (gated) output
    double in_mag = input.core_magnitude();
    double out_mag = result.output.core_magnitude();
    double ratio = in_mag > 1e-12 ? out_mag / in_mag : 0.0;
    if (ratio > 1e-12) {
        double log_ratio = std::log(ratio);
        result.transform_quality = std::exp(-log_ratio * log_ratio / 4.5);
    } else {
        result.transform_quality = 0.0;
    }

    // Coherence = cosine(output.core, target_embedding.core)
    result.coherence = result.output.core_cosine(
        Activation::from_embedding(target_emb));

    // Epistemic alignment
    auto src_info = ltm_.retrieve_concept(source);
    auto tgt_info = ltm_.retrieve_concept(target);
    if (src_info && tgt_info) {
        result.epistemic_alignment = compute_epistemic_alignment(
            src_info->epistemic.trust, tgt_info->epistemic.trust,
            src_info->epistemic.type, tgt_info->epistemic.type);
    } else {
        result.epistemic_alignment = 0.5;
    }

    return result;
}

// =============================================================================
// compute_epistemic_alignment
// =============================================================================

int GraphReasoner::epistemic_type_rank(EpistemicType type) {
    switch (type) {
        case EpistemicType::FACT:        return 0;
        case EpistemicType::DEFINITION:  return 0;
        case EpistemicType::THEORY:      return 1;
        case EpistemicType::HYPOTHESIS:  return 2;
        case EpistemicType::INFERENCE:   return 2;
        case EpistemicType::SPECULATION: return 3;
    }
    return 3;
}

double GraphReasoner::compute_epistemic_alignment(
    double source_trust, double target_trust,
    EpistemicType source_type, EpistemicType target_type)
{
    // Trust compatibility: penalize large trust drops
    double trust_compat = 1.0 - std::max(0.0, source_trust - target_trust);

    // Type compatibility: FACT -> SPECULATION is suspicious
    int rank_diff = std::abs(epistemic_type_rank(source_type)
                           - epistemic_type_rank(target_type));
    double type_compat = 1.0 - 0.15 * static_cast<double>(rank_diff);

    return 0.6 * trust_compat + 0.4 * type_compat;
}

// =============================================================================
// edge_confidence_from_model
// =============================================================================

double GraphReasoner::edge_confidence_from_model(ConceptId source_id) const {
    const ConceptModel* model = registry_.get_model(source_id);
    if (!model) return 0.3;  // No model: low confidence

    if (model->is_converged()) {
        return 0.7 + 0.3 * (1.0 - model->final_loss());
    }

    double sample_frac = std::min(1.0,
        static_cast<double>(model->sample_count()) / 100.0);
    return 0.3 + 0.4 * sample_frac;
}

// =============================================================================
// is_causal_relation / relation_reasoning_weight
// =============================================================================

bool GraphReasoner::is_causal_relation(RelationType type) {
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

double GraphReasoner::relation_reasoning_weight(RelationType type) {
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

// =============================================================================
// build_composition_features --- 90D input for ConvergencePort
// =============================================================================

std::array<double, convergence::QUERY_DIM> GraphReasoner::build_composition_features(
    ConceptId current, ConceptId target, RelationType rel,
    const FlexEmbedding& ctx, const PredictFeatures& pf) const
{
    std::array<double, convergence::QUERY_DIM> features{};

    auto emb_current = embeddings_.concept_embeddings().get_or_default(current);
    auto emb_target = embeddings_.concept_embeddings().get_or_default(target);
    auto rel_emb = embeddings_.get_relation_embedding(rel);

    // [0..CORE_DIM-1] concept_emb(current).core
    for (size_t i = 0; i < CORE_DIM; ++i)
        features[i] = emb_current.core[i];
    // [CORE_DIM..2*CORE_DIM-1] concept_emb(target).core
    for (size_t i = 0; i < CORE_DIM; ++i)
        features[CORE_DIM + i] = emb_target.core[i];
    // [2*CORE_DIM..3*CORE_DIM-1] relation_emb.core
    for (size_t i = 0; i < CORE_DIM; ++i)
        features[2 * CORE_DIM + i] = rel_emb.core[i];
    // [3*CORE_DIM..4*CORE_DIM-1] ctx_emb.core
    for (size_t i = 0; i < CORE_DIM; ++i)
        features[3 * CORE_DIM + i] = ctx.core[i];

    // [64..69] local CM intermediate features
    features[64] = pf.multihead_scores[0];
    features[65] = pf.multihead_scores[1];
    features[66] = pf.multihead_scores[2];
    features[67] = pf.multihead_scores[3];
    features[68] = pf.bilinear_score;
    features[69] = pf.dim_fraction;

    // [70..73] structural features
    features[70] = relation_reasoning_weight(rel);
    double epistemic_trust = 0.5;
    auto cinfo = ltm_.retrieve_concept(target);
    if (cinfo) epistemic_trust = cinfo->epistemic.trust;
    features[71] = epistemic_trust;

    auto out_rels = ltm_.get_outgoing_relations(target);
    features[72] = std::log(1.0 + static_cast<double>(out_rels.size())) / 5.0;
    features[73] = is_causal_relation(rel) ? 1.0 : 0.0;

    // [4*CORE_DIM+10..] dimensional score from bilinear intermediate
    for (size_t i = 0; i < CORE_DIM; ++i)
        features[(4 * CORE_DIM + 10) + i] = pf.dimensional_score[i];

    return features;
}

// =============================================================================
// forward_chain_state
// =============================================================================

std::array<double, convergence::OUTPUT_DIM> GraphReasoner::forward_chain_state(
    ConceptId concept_id,
    const std::array<double, convergence::QUERY_DIM>& features,
    const std::array<double, convergence::OUTPUT_DIM>& prev_state) const
{
    double input[convergence::CM_INPUT_DIM];
    for (size_t i = 0; i < convergence::QUERY_DIM; ++i)
        input[i] = features[i];
    for (size_t i = 0; i < convergence::OUTPUT_DIM; ++i)
        input[convergence::QUERY_DIM + i] = prev_state[i];

    std::array<double, convergence::OUTPUT_DIM> output{};
    ConceptModel* cm = registry_.get_model(concept_id);
    if (cm)
        cm->forward_convergence(input, output.data());

    return output;
}

// =============================================================================
// update_context --- EMA with chain state feedback and dimensional weighting
// =============================================================================

FlexEmbedding GraphReasoner::update_context(
    const FlexEmbedding& ctx, ConceptId new_concept,
    const std::array<double, convergence::OUTPUT_DIM>& chain_state,
    const CoreVec& dimensional_score) const
{
    FlexEmbedding emb = embeddings_.concept_embeddings().get_or_default(new_concept);
    FlexEmbedding result;

    // Check if dimensional scoring is active
    double dim_norm = 0.0;
    for (size_t i = 0; i < CORE_DIM; ++i)
        dim_norm += dimensional_score[i] * dimensional_score[i];
    bool use_dimensional = dim_norm > 1e-10;

    if (config_.enable_composition && config_.chain_ctx_blend > 0.0) {
        CoreVec chain_proj = project_chain_to_core(chain_state);
        double beta = config_.chain_ctx_blend;
        double alpha = config_.context_alpha;
        double ctx_weight = 1.0 - alpha - beta;
        if (ctx_weight < 0.0) {
            alpha = alpha / (alpha + beta);
            beta = 1.0 - alpha;
            ctx_weight = 0.0;
        }
        for (size_t i = 0; i < CORE_DIM; ++i) {
            double alpha_i = alpha;
            if (use_dimensional) {
                double dim_w = std::abs(dimensional_score[i]) / std::sqrt(dim_norm);
                alpha_i = alpha * (0.5 + 0.5 * dim_w);
            }
            result.core[i] = alpha_i * emb.core[i]
                            + (1.0 - alpha_i - beta) * ctx.core[i]
                            + beta * chain_proj[i];
        }
    } else {
        for (size_t i = 0; i < CORE_DIM; ++i) {
            double alpha_i = config_.context_alpha;
            if (use_dimensional) {
                double dim_w = std::abs(dimensional_score[i]) / std::sqrt(dim_norm);
                alpha_i = config_.context_alpha * (0.5 + 0.5 * dim_w);
            }
            result.core[i] = alpha_i * emb.core[i]
                            + (1.0 - alpha_i) * ctx.core[i];
        }
    }
    return result;
}

// =============================================================================
// evaluate_candidates --- Full-vector candidate evaluation
// =============================================================================
//
// For each candidate:
//   1. forward_edge() -> full EdgeResult with activation vectors
//   2. composite_score = weighted sum of transform_quality, coherence,
//      epistemic_alignment, relation_weight
//   3. Focus gate, ChainKAN coherence, seed anchor
//   4. Each rejected candidate gets a rejection_reason
//

std::vector<GraphReasoner::ScoredCandidate> GraphReasoner::evaluate_candidates(
    ConceptId current,
    const Activation& activation,
    const FlexEmbedding& ctx,
    const std::vector<FocusEntry>& focus_stack,
    const std::unordered_set<ConceptId>& visited,
    const std::unordered_set<uint16_t>& used_rels,
    const std::array<double, convergence::OUTPUT_DIM>& chain_state,
    ConceptId seed_id, size_t step_index,
    const FlexEmbedding& topic_centroid) const
{
    std::vector<ScoredCandidate> candidates;
    std::vector<ScoredCandidate> rejected;

    const double fgw = config_.focus_gate_weight;
    const double ccw = config_.chain_coherence_weight;
    const bool compose = config_.enable_composition;

    // Seed anchor embedding
    FlexEmbedding seed_emb;
    double anchor_strength = 0.0;
    if (config_.seed_anchor_weight > 0.0 && seed_id != 0) {
        seed_emb = embeddings_.concept_embeddings().get_or_default(seed_id);
        anchor_strength = config_.seed_anchor_weight *
            std::exp(-config_.seed_anchor_decay * static_cast<double>(step_index));
    }

    auto process_edge = [&](ConceptId target, RelationType type,
                             double rel_weight, bool outgoing) {
        ScoredCandidate cand;
        cand.target = target;
        cand.relation = type;
        cand.outgoing = outgoing;
        cand.is_causal = is_causal_relation(type);

        // Check if target is invalidated
        auto tgt_info = ltm_.retrieve_concept(target);
        if (!tgt_info || tgt_info->epistemic.is_invalidated()) {
            cand.rejection_reason = "INVALIDATED target";
            cand.composite_score = -1.0;
            rejected.push_back(cand);
            return;
        }

        // === Semantic similarity gate: embedding cosine(source, target) ===
        FlexEmbedding src_emb = embeddings_.concept_embeddings().get_or_default(current);
        FlexEmbedding tgt_emb_check = embeddings_.concept_embeddings().get_or_default(target);
        double emb_sim = core_similarity(src_emb, tgt_emb_check);
        cand.embedding_similarity = emb_sim;

        if (emb_sim < config_.min_embedding_similarity) {
            cand.rejection_reason = "below embedding similarity gate (" +
                std::to_string(emb_sim) + " < " +
                std::to_string(config_.min_embedding_similarity) + ")";
            cand.composite_score = -1.0;
            rejected.push_back(cand);
            return;
        }

        // === Topic centroid gate: check drift from chain topic ===
        if (config_.min_topic_similarity > 0.0) {
            double topic_sim = core_similarity(topic_centroid, tgt_emb_check);
            if (topic_sim < config_.min_topic_similarity) {
                cand.rejection_reason = "below topic centroid gate (" +
                    std::to_string(topic_sim) + " < " +
                    std::to_string(config_.min_topic_similarity) + ")";
                cand.composite_score = -1.0;
                rejected.push_back(cand);
                return;
            }
        }

        // === Core: forward_edge for full vector transformation ===
        cand.edge_result = forward_edge(current, target, type, activation, ctx);

        // Composite score: weighted combination
        double rel_reasoning_w = relation_reasoning_weight(type);
        double relation_score = rel_reasoning_w * std::pow(rel_weight, config_.relation_weight_power);
        if (!outgoing) relation_score *= config_.incoming_discount;

        // Include embedding similarity as a scoring factor
        double base_score =
            config_.weight_transform_quality * cand.edge_result.transform_quality +
            config_.weight_coherence * cand.edge_result.coherence +
            config_.weight_epistemic_alignment * cand.edge_result.epistemic_alignment +
            config_.weight_relation * relation_score;

        // Blend embedding similarity into composite (rewards semantically related targets)
        cand.composite_score = (1.0 - config_.embedding_sim_weight) * base_score
                             + config_.embedding_sim_weight * std::max(0.0, emb_sim);

        // Diversity bonus
        if (!used_rels.count(static_cast<uint16_t>(type)))
            cand.composite_score += config_.diversity_bonus;

        // Focus gate for non-causal edges
        if (!cand.is_causal && !focus_stack.empty()) {
            const FlexEmbedding& rel_emb_focus = embeddings_.get_relation_embedding(type);
            FlexEmbedding t_emb = embeddings_.concept_embeddings().get_or_default(target);

            double best_fscore = 0.0;
            for (const auto& fe : focus_stack) {
                ConceptModel* fcm = registry_.get_model(fe.id);
                if (!fcm) continue;
                FlexEmbedding f_emb = embeddings_.concept_embeddings().get_or_default(fe.id);
                double fs = fcm->predict_refined(rel_emb_focus, fe.emb, f_emb, t_emb);
                if (fs > best_fscore) best_fscore = fs;
            }

            if (best_fscore < config_.focus_min_gate) {
                cand.rejection_reason = "below focus gate (" +
                    std::to_string(best_fscore) + " < " +
                    std::to_string(config_.focus_min_gate) + ")";
                rejected.push_back(cand);
                return;
            }
            cand.composite_score = (1.0 - fgw) * cand.composite_score + fgw * best_fscore;
        }

        // Seed anchor: penalize drift
        if (anchor_strength > 0.0) {
            FlexEmbedding t_emb = embeddings_.concept_embeddings().get_or_default(target);
            double weighted_penalty = 0.0;
            double total_seed_weight = 0.0;
            for (size_t i = 0; i < CORE_DIM; ++i) {
                double sw = seed_emb.core[i] * seed_emb.core[i];
                double diff = seed_emb.core[i] - t_emb.core[i];
                weighted_penalty += sw * diff * diff;
                total_seed_weight += sw;
            }
            if (total_seed_weight > 1e-10)
                weighted_penalty /= total_seed_weight;
            cand.composite_score -= anchor_strength * weighted_penalty;
        }

        // ChainKAN composition: simulate chain state + coherence
        if (compose) {
            ConceptModel* cm_model = registry_.get_model(current);
            PredictFeatures pf;
            if (cm_model) {
                const FlexEmbedding& rel_emb_comp = embeddings_.get_relation_embedding(type);
                FlexEmbedding ctx_mixed = ctx;
                FlexEmbedding tgt_emb = embeddings_.concept_embeddings().get_or_default(target);
                for (size_t i = 0; i < CORE_DIM; ++i)
                    ctx_mixed.core[i] = (1.0 - config_.context_alpha) * ctx.core[i]
                                      + config_.context_alpha * tgt_emb.core[i];
                FlexEmbedding from_emb = embeddings_.concept_embeddings().get_or_default(current);
                pf = cm_model->predict_refined_with_features(
                    rel_emb_comp, ctx_mixed, from_emb, tgt_emb);
            } else {
                pf.bilinear_score = 0.5;
            }

            auto features = build_composition_features(current, target, type, ctx, pf);
            cand.simulated_state = forward_chain_state(target, features, chain_state);
            cand.chain_coherence = chain_kan_.evaluate(
                chain_state.data(), cand.simulated_state.data());

            // Blend with chain coherence
            cand.composite_score = (1.0 - ccw) * cand.composite_score
                                 + ccw * cand.chain_coherence;
        }

        if (cand.composite_score < config_.min_composite_score) {
            cand.rejection_reason = "below min composite score (" +
                std::to_string(cand.composite_score) + " < " +
                std::to_string(config_.min_composite_score) + ")";
            rejected.push_back(cand);
            return;
        }

        candidates.push_back(cand);
    };

    // Outgoing relations
    auto outgoing = ltm_.get_outgoing_relations(current);
    for (const auto& rel : outgoing) {
        if (visited.count(rel.target)) continue;
        process_edge(rel.target, rel.type, rel.weight, true);
    }

    // Incoming relations
    auto incoming = ltm_.get_incoming_relations(current);
    for (const auto& rel : incoming) {
        if (visited.count(rel.source)) continue;
        process_edge(rel.source, rel.type, rel.weight, false);
    }

    // Sort by composite score
    std::sort(candidates.begin(), candidates.end(),
        [](const ScoredCandidate& a, const ScoredCandidate& b) {
            return a.composite_score > b.composite_score;
        });

    if (candidates.size() > config_.max_candidates)
        candidates.resize(config_.max_candidates);

    // Mark rejected candidates with "lower composite score" reason
    // (those that passed all gates but weren't the winner)
    for (size_t i = 1; i < candidates.size(); ++i) {
        candidates[i].rejection_reason = "lower composite score";
    }

    // Append truly rejected candidates
    for (auto& r : rejected)
        candidates.push_back(std::move(r));

    return candidates;
}

// =============================================================================
// build_trace_step
// =============================================================================

TraceStep GraphReasoner::build_trace_step(
    ConceptId source, const ScoredCandidate& winner,
    const Activation& input_act,
    const std::vector<ScoredCandidate>& all_candidates,
    size_t step_index,
    const std::array<double, convergence::OUTPUT_DIM>& chain_state) const
{
    TraceStep step;
    step.source_id = source;
    step.target_id = winner.target;
    step.relation = winner.relation;
    step.is_outgoing = winner.outgoing;
    step.focus_shifted = winner.is_causal;
    step.step_index = step_index;

    // Activations
    step.input_activation = input_act;
    step.output_activation = winner.edge_result.output;
    step.dimensional_contribution = winner.edge_result.dimensional_contrib;

    // Epistemic metadata snapshots
    auto src_info = ltm_.retrieve_concept(source);
    auto tgt_info = ltm_.retrieve_concept(winner.target);
    if (src_info) {
        step.source_epistemic_type = src_info->epistemic.type;
        step.source_epistemic_status = src_info->epistemic.status;
        step.source_trust = src_info->epistemic.trust;
    }
    if (tgt_info) {
        step.target_epistemic_type = tgt_info->epistemic.type;
        step.target_epistemic_status = tgt_info->epistemic.status;
        step.target_trust = tgt_info->epistemic.trust;
    }

    // Edge quality
    step.edge_confidence = edge_confidence_from_model(source);
    step.transform_quality = winner.edge_result.transform_quality;
    step.coherence = winner.edge_result.coherence;
    step.epistemic_alignment = winner.edge_result.epistemic_alignment;

    // Dual neuron quality
    step.nn_quality = winner.edge_result.nn_quality;
    step.kan_quality = winner.edge_result.kan_quality;
    step.kan_gate = winner.edge_result.kan_gate;

    // Step trust
    step.compute_step_trust(
        config_.step_trust_source_weight,
        config_.step_trust_edge_weight,
        config_.step_trust_target_weight,
        config_.step_trust_transform_weight);

    // ChainKAN state
    step.chain_state = chain_state;
    step.chain_coherence = winner.chain_coherence;

    // Seed similarity
    // (computed by caller and set after)

    // Composite score
    step.composite_score = winner.composite_score;

    // Top contributing dimensions
    auto top_k = step.output_activation.top_k_dims(config_.trace_top_k_dims);
    step.top_dims = top_k;
    step.top_dim_values.reserve(top_k.size());
    for (size_t idx : top_k)
        step.top_dim_values.push_back(step.output_activation.core()[idx]);

    // Alternatives with rejection reasons
    for (size_t i = 1; i < all_candidates.size(); ++i) {
        const auto& c = all_candidates[i];
        TraceAlternative alt;
        alt.target_id = c.target;
        alt.relation = c.relation;
        alt.composite_score = c.composite_score;
        alt.rejection_reason = c.rejection_reason.empty()
            ? "lower composite score" : c.rejection_reason;
        step.alternatives.push_back(alt);
    }

    return step;
}

// =============================================================================
// reason_from(seed) --- Delegates to reason_from_internal (backward-compatible)
// =============================================================================

GraphChain GraphReasoner::reason_from(ConceptId seed) const {
    return reason_from_internal(seed, nullptr);
}

// =============================================================================
// reason_from_internal(seed, feedback) --- Main reasoning loop with activations
// =============================================================================
//
// Core reasoning engine. When feedback is null, behaves identically to the
// original reason_from(). When feedback is provided, primes the chain with
// enriched context, refined topic direction, and pre-populated visited set
// (forcing exploration of different paths each round).
//

GraphChain GraphReasoner::reason_from_internal(ConceptId seed,
                                                const FeedbackState* feedback) const {
    GraphChain chain;

    // Initialize activation from seed embedding
    FlexEmbedding seed_emb_flex = embeddings_.concept_embeddings().get_or_default(seed);
    Activation activation = Activation::from_embedding(seed_emb_flex);
    FlexEmbedding ctx = seed_emb_flex;

    // Feedback priming: blend seed embedding with enriched context
    if (feedback) {
        double alpha = config_.feedback.context_blend_alpha;
        for (size_t i = 0; i < CORE_DIM; ++i)
            ctx.core[i] = (1.0 - alpha) * seed_emb_flex.core[i]
                         + alpha * feedback->enriched_context[i];
    }

    // Focus stack
    std::vector<FocusEntry> focus_stack;
    focus_stack.push_back({seed, ctx});

    // Visited tracking
    std::unordered_set<ConceptId> visited;
    std::unordered_set<uint16_t> used_rels;

    // Feedback priming: pre-populate visited set (force different paths)
    if (feedback) {
        visited = feedback->explored;
    }
    visited.insert(seed);  // Seed itself is always allowed

    // Chain state
    std::array<double, convergence::OUTPUT_DIM> chain_state{};
    if (config_.enable_composition) {
        PredictFeatures seed_pf;
        seed_pf.bilinear_score = 0.5;
        auto seed_features = build_composition_features(
            seed, seed, RelationType::CUSTOM, ctx, seed_pf);
        chain_state = forward_chain_state(seed, seed_features, chain_state);
    }

    // Seed step (step 0)
    TraceStep seed_step;
    seed_step.source_id = seed;
    seed_step.target_id = seed;
    seed_step.step_index = 0;
    seed_step.input_activation = activation;
    seed_step.output_activation = activation;
    seed_step.chain_state = chain_state;
    seed_step.seed_similarity = 1.0;
    seed_step.composite_score = 1.0;

    auto seed_info = ltm_.retrieve_concept(seed);
    if (seed_info) {
        seed_step.source_epistemic_type = seed_info->epistemic.type;
        seed_step.source_trust = seed_info->epistemic.trust;
        seed_step.target_epistemic_type = seed_info->epistemic.type;
        seed_step.target_trust = seed_info->epistemic.trust;
        seed_step.source_epistemic_status = seed_info->epistemic.status;
        seed_step.target_epistemic_status = seed_info->epistemic.status;
    }
    seed_step.edge_confidence = edge_confidence_from_model(seed);
    seed_step.step_trust = seed_step.source_trust;

    chain.steps.push_back(seed_step);
    chain.initial_activation = activation;

    ConceptId current = seed;

    // Topic centroid: running EMA of visited concept embeddings (anti-drift)
    // Feedback priming: start from refined topic direction
    FlexEmbedding topic_centroid = seed_emb_flex;
    if (feedback) {
        for (size_t i = 0; i < CORE_DIM; ++i)
            topic_centroid.core[i] = feedback->refined_topic[i];
    }

    for (size_t step = 0; step < config_.max_steps; ++step) {
        // Check activation decay
        if (activation.core_magnitude() < config_.min_activation_magnitude) {
            chain.termination = TerminationReason::ACTIVATION_DECAY;
            break;
        }

        // Evaluate all candidates with full activation vectors
        auto all_candidates = evaluate_candidates(
            current, activation, ctx, focus_stack, visited, used_rels,
            chain_state, seed, step, topic_centroid);

        // Filter to viable candidates (positive composite score, no rejection)
        std::vector<ScoredCandidate> viable;
        for (const auto& c : all_candidates) {
            if (c.composite_score >= config_.min_composite_score &&
                c.rejection_reason.empty()) {
                viable.push_back(c);
            }
        }

        // Fallback: retry without focus gate
        if (viable.empty() && !focus_stack.empty()) {
            std::vector<FocusEntry> empty_stack;
            all_candidates = evaluate_candidates(
                current, activation, ctx, empty_stack, visited, used_rels,
                chain_state, seed, step, topic_centroid);
            for (const auto& c : all_candidates) {
                if (c.composite_score >= config_.min_composite_score &&
                    c.rejection_reason.empty()) {
                    viable.push_back(c);
                }
            }
        }

        if (viable.empty()) {
            chain.termination = TerminationReason::NO_VIABLE_CANDIDATES;
            break;
        }

        const auto& best = viable[0];

        // Coherence-gated termination
        if (config_.enable_composition && step > 0 &&
            best.chain_coherence < config_.min_coherence_gate) {
            chain.termination = TerminationReason::COHERENCE_GATE;
            break;
        }

        // Update chain state
        if (config_.enable_composition)
            chain_state = best.simulated_state;

        // Build full trace step
        TraceStep trace = build_trace_step(
            current, best, activation, all_candidates, step + 1, chain_state);

        // Seed similarity
        FlexEmbedding step_emb = embeddings_.concept_embeddings().get_or_default(best.target);
        trace.seed_similarity = core_similarity(seed_emb_flex, step_emb);

        chain.steps.push_back(trace);

        // Update activation to the transformed output
        activation = best.edge_result.output;

        // Update context
        ctx = update_context(ctx, best.target, chain_state,
                              best.edge_result.dimensional_contrib);

        // Update topic centroid (EMA of visited concept embeddings)
        {
            FlexEmbedding step_emb_tc = embeddings_.concept_embeddings().get_or_default(best.target);
            double alpha_tc = config_.topic_centroid_alpha;
            for (size_t i = 0; i < CORE_DIM; ++i)
                topic_centroid.core[i] = alpha_tc * step_emb_tc.core[i]
                                       + (1.0 - alpha_tc) * topic_centroid.core[i];
        }

        // Focus shift
        if (best.is_causal) {
            FlexEmbedding new_emb = embeddings_.concept_embeddings().get_or_default(best.target);
            focus_stack.push_back({best.target, new_emb});
            if (focus_stack.size() > 3)
                focus_stack.erase(focus_stack.begin());
        }

        // Check chain trust
        if (chain.steps.size() >= 3) {
            double log_sum = 0.0;
            size_t count = 0;
            for (size_t i = 1; i < chain.steps.size(); ++i) {
                double t = std::max(1e-10, chain.steps[i].step_trust);
                log_sum += std::log(t);
                ++count;
            }
            double geo_mean = count > 0 ? std::exp(log_sum / static_cast<double>(count)) : 0.0;
            if (geo_mean < config_.min_chain_trust) {
                chain.termination = TerminationReason::TRUST_TOO_LOW;
                break;
            }
        }

        // Seed drift check
        if (chain.steps.size() >= 3) {
            size_t consecutive_low = 0;
            for (size_t s = chain.steps.size(); s >= 2; --s) {
                if (chain.steps[s - 1].seed_similarity < config_.min_seed_similarity)
                    ++consecutive_low;
                else
                    break;
            }
            if (consecutive_low >= config_.max_consecutive_seed_drops) {
                chain.termination = TerminationReason::SEED_DRIFT;
                break;
            }
        }

        current = best.target;
        visited.insert(current);
        used_rels.insert(static_cast<uint16_t>(best.relation));
    }

    // If we exhausted max_steps without other termination
    if (chain.termination == TerminationReason::STILL_RUNNING)
        chain.termination = TerminationReason::MAX_STEPS_REACHED;

    // Best-prefix truncation (remove drifted tail)
    if (chain.steps.size() >= 3) {
        size_t best_prefix = chain.steps.size() - 1;
        for (size_t s = chain.steps.size() - 1; s >= 2; --s) {
            if (chain.steps[s].seed_similarity < config_.min_seed_similarity)
                best_prefix = s - 1;
            else
                break;
        }
        if (best_prefix < chain.steps.size() - 1 && best_prefix >= 1)
            chain.steps.resize(best_prefix + 1);
    }

    // Compute chain-level metrics
    chain.compute_chain_metrics();

    // Log for orchestrator training data
    if (logger_) {
        double quality = compute_chain_quality(chain);
        int fb_round = feedback ? static_cast<int>(feedback->round) : -1;
        logger_->log_chain(seed, chain, quality, embeddings_, ltm_, fb_round);
    }

    return chain;
}

// =============================================================================
// extract_feedback --- Build feedback state from a completed chain
// =============================================================================
//
// Extracts context, topic direction, and visited concepts from a chain
// to prime the next feedback round. Only high-quality steps (composite_score
// > 0.3) contribute to the enriched context centroid.
//

FeedbackState GraphReasoner::extract_feedback(const GraphChain& chain,
                                               const FeedbackState* prior) const {
    FeedbackState state;

    // Accumulate explored concepts from prior rounds
    if (prior) {
        state.explored = prior->explored;
        state.best_quality = prior->best_quality;
        state.round = prior->round + 1;
    }

    // Add all concepts visited in this chain to explored set
    for (const auto& step : chain.steps) {
        state.explored.insert(step.source_id);
        if (step.target_id != step.source_id)
            state.explored.insert(step.target_id);
    }

    // Enriched context: weighted centroid of high-quality step embeddings
    state.enriched_context.fill(0.0);
    double weight_sum = 0.0;
    for (size_t i = 1; i < chain.steps.size(); ++i) {
        const auto& step = chain.steps[i];
        if (step.composite_score > 0.3) {
            FlexEmbedding emb = embeddings_.concept_embeddings().get_or_default(step.target_id);
            double w = step.composite_score;
            for (size_t d = 0; d < CORE_DIM; ++d)
                state.enriched_context[d] += w * emb.core[d];
            weight_sum += w;
        }
    }
    if (weight_sum > 1e-12) {
        for (size_t d = 0; d < CORE_DIM; ++d)
            state.enriched_context[d] /= weight_sum;
    } else {
        // Fallback: use seed embedding
        FlexEmbedding seed_emb = embeddings_.concept_embeddings().get_or_default(
            chain.steps.empty() ? 0 : chain.steps[0].source_id);
        state.enriched_context = seed_emb.core;
    }

    // Refined topic: approximate from last step's embedding with EMA blending
    if (chain.steps.size() >= 2) {
        FlexEmbedding last_emb = embeddings_.concept_embeddings().get_or_default(
            chain.steps.back().target_id);
        FlexEmbedding seed_emb = embeddings_.concept_embeddings().get_or_default(
            chain.steps[0].source_id);
        // Blend: 70% last step direction + 30% seed anchor
        for (size_t d = 0; d < CORE_DIM; ++d)
            state.refined_topic[d] = 0.7 * last_emb.core[d] + 0.3 * seed_emb.core[d];
    } else {
        FlexEmbedding seed_emb = embeddings_.concept_embeddings().get_or_default(
            chain.steps.empty() ? 0 : chain.steps[0].source_id);
        state.refined_topic = seed_emb.core;
    }

    // Update best quality
    double q = compute_chain_quality(chain);
    if (q > state.best_quality)
        state.best_quality = q;

    return state;
}

// =============================================================================
// reason_with_feedback(seed) --- Adaptive multi-round reasoning
// =============================================================================
//
// Outer feedback loop. After each chain completes, extracts discoveries and
// re-reasons from the same seed with enriched context and accumulated visited
// set. Stops early if quality is already high or improvement is diminishing.
//

GraphChain GraphReasoner::reason_with_feedback(ConceptId seed) const {
    // Round 0: no priming
    GraphChain best_chain = reason_from_internal(seed, nullptr);
    double best_quality = compute_chain_quality(best_chain);

    // Skip feedback if first chain is already great
    if (best_quality >= config_.feedback.quality_skip_threshold)
        return best_chain;

    FeedbackState feedback = extract_feedback(best_chain, nullptr);
    feedback.best_quality = best_quality;
    double prev_quality = best_quality;

    for (size_t round = 1; round <= config_.feedback.max_rounds; ++round) {
        GraphChain chain_r = reason_from_internal(seed, &feedback);
        double quality_r = compute_chain_quality(chain_r);

        if (quality_r > best_quality) {
            best_chain = std::move(chain_r);
            best_quality = quality_r;
        }

        // Diminishing returns check
        if (quality_r - prev_quality < config_.feedback.improvement_threshold)
            break;

        prev_quality = quality_r;
        feedback = extract_feedback(
            best_quality == quality_r ? best_chain : chain_r, &feedback);
        feedback.best_quality = best_quality;
    }

    return best_chain;
}

// =============================================================================
// reason_with_feedback(seeds) --- Multi-seed with feedback
// =============================================================================

GraphChain GraphReasoner::reason_with_feedback(const std::vector<ConceptId>& seeds) const {
    GraphChain best;
    double best_quality = -1.0;
    for (ConceptId seed : seeds) {
        auto chain = reason_with_feedback(seed);
        double q = compute_chain_quality(chain);
        if (q > best_quality) {
            best = std::move(chain);
            best_quality = q;
        }
    }
    return best;
}

// =============================================================================
// reason_from(seeds) --- Multi-seed: return best chain
// =============================================================================

GraphChain GraphReasoner::reason_from(const std::vector<ConceptId>& seeds) const {
    GraphChain best;
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

// =============================================================================
// compute_chain_quality
// =============================================================================

double GraphReasoner::compute_chain_quality(const GraphChain& chain) const {
    if (chain.steps.size() < 2) return 0.0;

    // Factor 1: chain trust (geometric mean of step trusts)
    double trust = std::max(0.01, chain.chain_trust);

    // Factor 2: length reward (log scale, diminishing returns)
    double length_reward = std::log(1.0 + static_cast<double>(chain.steps.size())) / 3.0;

    // Factor 3: topical coherence (mean pairwise core similarity)
    double coherence_sum = 0.0;
    size_t pair_count = 0;
    for (size_t i = 0; i < chain.steps.size(); ++i) {
        auto emb_i = embeddings_.concept_embeddings().get_or_default(
            chain.steps[i].step_index == 0 ? chain.steps[i].source_id : chain.steps[i].target_id);
        for (size_t j = i + 1; j < chain.steps.size(); ++j) {
            auto emb_j = embeddings_.concept_embeddings().get_or_default(
                chain.steps[j].step_index == 0 ? chain.steps[j].source_id : chain.steps[j].target_id);
            coherence_sum += core_similarity(emb_i, emb_j);
            ++pair_count;
        }
    }
    double topical = pair_count > 0
        ? (coherence_sum / static_cast<double>(pair_count) + 1.0) / 2.0
        : 0.5;

    // Factor 4: average transform quality
    double avg_tq = 0.0;
    size_t tq_count = 0;
    for (size_t i = 1; i < chain.steps.size(); ++i) {
        avg_tq += chain.steps[i].transform_quality;
        ++tq_count;
    }
    avg_tq = tq_count > 0 ? avg_tq / static_cast<double>(tq_count) : 0.5;

    return std::pow(trust * length_reward * topical * avg_tq, 0.25);
}

// =============================================================================
// Co-Learning API: extract_signals
// =============================================================================
//
// Extracts learning signals from a completed reasoning chain.
// For each traversed edge: positive/negative classification based on quality.
// For each rejected alternative: negative signal (pain) with reason.
// Chain-level suggestions: strengthen good edges, weaken bad ones.
//

ChainSignal GraphReasoner::extract_signals(const GraphChain& chain) const {
    ChainSignal signal;
    signal.seed = chain.steps.empty() ? 0 : chain.steps[0].source_id;
    signal.termination = chain.termination;
    signal.chain_quality = compute_chain_quality(chain);

    // Adaptive threshold = chain mean composite score.
    // Within each chain, above-mean edges get strengthened, below-mean get weakened.
    // This is zero-sum: total positive deviation ≈ total negative deviation.
    // Prevents uniform inflation (absolute 0.5 → 9:1 strengthen:weaken).
    double score_sum = 0.0;
    size_t score_count = 0;
    for (size_t i = 1; i < chain.steps.size(); ++i) {
        score_sum += chain.steps[i].composite_score;
        ++score_count;
    }
    const double threshold = score_count > 0 ? score_sum / static_cast<double>(score_count) : 0.5;

    for (size_t i = 1; i < chain.steps.size(); ++i) {
        const auto& step = chain.steps[i];

        // Traversed edge signal
        EdgeSignal es;
        es.source = step.source_id;
        es.target = step.target_id;
        es.relation = step.relation;
        es.transform_quality = step.transform_quality;
        es.coherence = step.coherence;
        es.epistemic_alignment = step.epistemic_alignment;
        es.composite_score = step.composite_score;
        es.nn_quality = step.nn_quality;
        es.kan_quality = step.kan_quality;
        es.kan_gate = step.kan_gate;
        es.was_traversed = true;

        // Compute embedding similarity
        FlexEmbedding src_emb = embeddings_.concept_embeddings().get_or_default(step.source_id);
        FlexEmbedding tgt_emb = embeddings_.concept_embeddings().get_or_default(step.target_id);
        es.embedding_similarity = core_similarity(src_emb, tgt_emb);

        es.is_positive = (es.composite_score >= threshold);
        signal.traversed_edges.push_back(es);

        // Generate edge suggestion with proportional delta
        // Deviation from threshold: positive = strengthen, negative = weaken
        ChainSignal::EdgeSuggestion suggestion;
        suggestion.source = step.source_id;
        suggestion.target = step.target_id;
        suggestion.relation = step.relation;
        double deviation = es.composite_score - threshold;
        suggestion.delta_weight = deviation;  // proportional, zero-centered at absolute threshold
        if (es.is_positive) {
            suggestion.reason = "quality " + std::to_string(es.composite_score)
                              + " above threshold " + std::to_string(threshold);
        } else {
            suggestion.reason = "quality " + std::to_string(es.composite_score)
                              + " below threshold " + std::to_string(threshold);
        }
        signal.suggestions.push_back(suggestion);

        // Top rejected alternatives (up to 3 per step)
        size_t alt_count = std::min(step.alternatives.size(), size_t(3));
        for (size_t j = 0; j < alt_count; ++j) {
            const auto& alt = step.alternatives[j];
            EdgeSignal aes;
            aes.source = step.source_id;
            aes.target = alt.target_id;
            aes.relation = alt.relation;
            aes.composite_score = alt.composite_score;
            aes.was_traversed = false;
            aes.is_positive = false;
            aes.rejection_reason = alt.rejection_reason;

            FlexEmbedding alt_emb = embeddings_.concept_embeddings().get_or_default(alt.target_id);
            aes.embedding_similarity = core_similarity(src_emb, alt_emb);

            signal.rejected_edges.push_back(aes);
        }
    }

    return signal;
}

// =============================================================================
// Co-Learning API: evaluate_edge
// =============================================================================
//
// Evaluates a specific edge's quality without doing full reasoning.
// Useful for the Graph -> CM direction: "is this edge worth keeping?"
//

EdgeSignal GraphReasoner::evaluate_edge(
    ConceptId source, ConceptId target,
    RelationType relation) const
{
    EdgeSignal es;
    es.source = source;
    es.target = target;
    es.relation = relation;

    // Get source embedding as activation
    FlexEmbedding src_emb = embeddings_.concept_embeddings().get_or_default(source);
    Activation act = Activation::from_embedding(src_emb);

    // Forward pass
    EdgeResult result = forward_edge(source, target, relation, act, src_emb);

    es.transform_quality = result.transform_quality;
    es.coherence = result.coherence;
    es.epistemic_alignment = result.epistemic_alignment;
    es.composite_score = result.composite_score;
    es.nn_quality = result.nn_quality;
    es.kan_quality = result.kan_quality;
    es.kan_gate = result.kan_gate;

    // Embedding similarity
    FlexEmbedding tgt_emb = embeddings_.concept_embeddings().get_or_default(target);
    es.embedding_similarity = core_similarity(src_emb, tgt_emb);

    // Classify
    double quality = 0.35 * es.transform_quality + 0.35 * es.coherence
                   + 0.15 * es.epistemic_alignment + 0.15 * es.embedding_similarity;
    es.is_positive = (quality >= 0.5);

    return es;
}

} // namespace brain19
