#include "colearn_loop.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <random>
#include <thread>

namespace brain19 {

CoLearnLoop::CoLearnLoop(
    LongTermMemory& ltm, ConceptModelRegistry& registry,
    EmbeddingManager& embeddings, GraphReasoner& reasoner,
    const CoLearnConfig& config)
    : ltm_(ltm)
    , registry_(registry)
    , embeddings_(embeddings)
    , reasoner_(reasoner)
    , config_(config)
    , episodic_memory_(config.max_episodes)
    , extractor_(ltm, reasoner, config)
    , error_collector_(config.error_correction)
{
    if (config_.thread_count > 1) {
        thread_pool_ = std::make_unique<ThreadPool>(config_.thread_count);
    }
}

CoLearnLoop::~CoLearnLoop() {
    stop_continuous();
}

// =============================================================================
// SuperpositionTracker — Context-dependent quality tracking
// =============================================================================

void CoLearnLoop::SuperpositionTracker::record(
    ConceptId cid, RelationType rel, double quality)
{
    auto& cs = stats[cid];
    cs.quality_by_relation[static_cast<uint16_t>(rel)].push_back(quality);
    ++cs.total_observations;
}

bool CoLearnLoop::SuperpositionTracker::should_enable(
    ConceptId cid, size_t min_observations, double min_std) const
{
    auto it = stats.find(cid);
    if (it == stats.end()) return false;
    const auto& cs = it->second;
    if (cs.total_observations < min_observations) return false;
    if (cs.quality_by_relation.size() < 2) return false;  // need diverse contexts

    // Compute per-relation mean qualities
    std::vector<double> means;
    for (const auto& [rel, qualities] : cs.quality_by_relation) {
        if (qualities.empty()) continue;
        double sum = 0.0;
        for (double q : qualities) sum += q;
        means.push_back(sum / static_cast<double>(qualities.size()));
    }
    if (means.size() < 2) return false;

    // Compute std across relation means
    double mean_of_means = 0.0;
    for (double m : means) mean_of_means += m;
    mean_of_means /= static_cast<double>(means.size());

    double variance = 0.0;
    for (double m : means) {
        double d = m - mean_of_means;
        variance += d * d;
    }
    variance /= static_cast<double>(means.size());

    return std::sqrt(variance) > min_std;
}

// =============================================================================
// select_wake_seeds — Diverse 4-way seed selection
// =============================================================================

std::vector<ConceptId> CoLearnLoop::select_wake_seeds() {
    // If CuriosityEngine is wired in, delegate to it
    if (curiosity_) {
        curiosity_->refresh(ltm_, registry_, episodic_memory_, error_collector_,
                            seed_pain_scores_, nullptr, nullptr);
        auto seeds = curiosity_->select_seeds(config_.wake_chains_per_cycle);
        if (!seeds.empty()) return seeds;
        // Fall through to legacy if curiosity returned nothing
    }

    auto all_ids = ltm_.get_all_concept_ids();
    if (all_ids.empty()) return {};

    size_t count = std::min(config_.wake_chains_per_cycle, all_ids.size());
    std::vector<ConceptId> seeds;
    seeds.reserve(count);

    // 4-way split: random + high-connectivity + low-trust + high-pain
    // If no pain data yet (first cycle), redistribute to the other 3 buckets
    bool have_pain = !seed_pain_scores_.empty();
    size_t n_random, n_connected, n_low_trust, n_high_pain;
    if (have_pain) {
        n_random = count / 4;
        n_connected = count / 4;
        n_low_trust = count / 4;
        n_high_pain = count - n_random - n_connected - n_low_trust;
    } else {
        n_random = count / 3;
        n_connected = count / 3;
        n_low_trust = count - n_random - n_connected;
        n_high_pain = 0;
    }

    // Random seeds
    {
        std::vector<ConceptId> shuffled = all_ids;
        std::mt19937 rng(static_cast<unsigned>(cycle_count_ * 12345 + 67));
        std::shuffle(shuffled.begin(), shuffled.end(), rng);
        for (size_t i = 0; i < n_random && i < shuffled.size(); ++i) {
            seeds.push_back(shuffled[i]);
        }
    }

    // High connectivity seeds
    {
        struct Scored { ConceptId id; size_t degree; };
        std::vector<Scored> scored;
        for (ConceptId cid : all_ids) {
            scored.push_back({cid, ltm_.get_relation_count(cid)});
        }
        std::sort(scored.begin(), scored.end(),
            [](const Scored& a, const Scored& b) { return a.degree > b.degree; });
        for (size_t i = 0; i < n_connected && i < scored.size(); ++i) {
            if (std::find(seeds.begin(), seeds.end(), scored[i].id) == seeds.end()) {
                seeds.push_back(scored[i].id);
            }
        }
    }

    // Low trust seeds (explore uncertain regions)
    {
        struct Scored { ConceptId id; double trust; };
        std::vector<Scored> scored;
        for (ConceptId cid : all_ids) {
            auto info = ltm_.retrieve_concept(cid);
            if (info && !info->epistemic.is_invalidated()) {
                scored.push_back({cid, info->epistemic.trust});
            }
        }
        std::sort(scored.begin(), scored.end(),
            [](const Scored& a, const Scored& b) { return a.trust < b.trust; });
        for (size_t i = 0; i < n_low_trust && i < scored.size(); ++i) {
            if (std::find(seeds.begin(), seeds.end(), scored[i].id) == seeds.end()) {
                seeds.push_back(scored[i].id);
            }
        }
    }

    // High pain seeds (revisit failures)
    if (n_high_pain > 0) {
        struct Scored { ConceptId id; double pain; };
        std::vector<Scored> scored;
        for (const auto& [cid, pain] : seed_pain_scores_) {
            scored.push_back({cid, pain});
        }
        std::sort(scored.begin(), scored.end(),
            [](const Scored& a, const Scored& b) { return a.pain > b.pain; });
        for (size_t i = 0; i < n_high_pain && i < scored.size(); ++i) {
            if (std::find(seeds.begin(), seeds.end(), scored[i].id) == seeds.end()) {
                seeds.push_back(scored[i].id);
            }
        }
    }

    return seeds;
}

// =============================================================================
// Wake Phase — Reason from seeds, store episodes
// =============================================================================

void CoLearnLoop::wake_phase() {
    last_chains_produced_ = 0;
    last_episodes_stored_ = 0;
    last_quality_sum_ = 0.0;

    auto seeds = select_wake_seeds();
    if (seeds.empty()) return;

    if (!thread_pool_ || config_.thread_count <= 1) {
        wake_phase_serial(seeds);
    } else {
        wake_phase_parallel(seeds);
    }
}

// Serial wake — byte-identical to original code path
void CoLearnLoop::wake_phase_serial(const std::vector<ConceptId>& seeds) {
    for (ConceptId seed : seeds) {
        GraphChain chain = reasoner_.reason_from(seed);
        if (chain.empty()) continue;

        ++last_chains_produced_;

        double quality = reasoner_.compute_chain_quality(chain);
        last_quality_sum_ += quality;

        // Track pain per seed (EMA)
        double term_pain = 0.0;
        if (chain.termination == TerminationReason::NO_VIABLE_CANDIDATES) term_pain = 0.7;
        else if (chain.termination == TerminationReason::TRUST_TOO_LOW) term_pain = 0.6;
        else if (chain.termination == TerminationReason::ACTIVATION_DECAY) term_pain = 0.5;
        else if (chain.termination == TerminationReason::SEED_DRIFT) term_pain = 0.4;
        else if (chain.termination == TerminationReason::COHERENCE_GATE) term_pain = 0.3;

        double quality_pain = std::max(0.0, 1.0 - quality);
        double pain_score = 0.5 * term_pain + 0.5 * quality_pain;

        auto it = seed_pain_scores_.find(seed);
        if (it != seed_pain_scores_.end()) {
            it->second = 0.7 * it->second + 0.3 * pain_score;
        } else {
            seed_pain_scores_[seed] = pain_score;
        }

        // Track context-dependent quality for superposition heuristic
        // + record activation for FlexEmbedding growth tracking
        auto& concept_store = embeddings_.concept_embeddings();
        concept_store.record_activation(seed, tick_);
        for (size_t si = 1; si < chain.steps.size(); ++si) {
            superposition_tracker_.record(
                chain.steps[si].target_id,
                chain.steps[si].relation,
                quality);
            concept_store.record_activation(chain.steps[si].target_id, tick_);
        }

        // Extract signals and collect prediction errors for corrective training
        ChainSignal signal = reasoner_.extract_signals(chain);
        error_collector_.collect_from_chain(chain, signal);

        // Convert chain to episode and store
        Episode ep = episodic_memory_.from_chain(chain, seed);
        ep.quality = quality;
        episodic_memory_.store(ep);
        ++last_episodes_stored_;
    }
}

// Parallel wake — dispatch chains to thread pool, gather + merge single-threaded
void CoLearnLoop::wake_phase_parallel(const std::vector<ConceptId>& seeds) {
    struct LocalResult {
        GraphChain chain;
        double quality = 0.0;
        ConceptId seed = 0;
        double pain_score = 0.0;
        ChainSignal signal;
    };

    // Phase 1: Dispatch chains to thread pool
    std::vector<std::future<LocalResult>> futures;
    futures.reserve(seeds.size());

    for (ConceptId seed : seeds) {
        futures.push_back(thread_pool_->submit([this, seed]() -> LocalResult {
            LocalResult r;
            r.seed = seed;
            r.chain = reasoner_.reason_from(seed);  // const, thread-safe reads
            if (r.chain.empty()) return r;

            r.quality = reasoner_.compute_chain_quality(r.chain);

            // Compute pain locally
            double term_pain = 0.0;
            if (r.chain.termination == TerminationReason::NO_VIABLE_CANDIDATES) term_pain = 0.7;
            else if (r.chain.termination == TerminationReason::TRUST_TOO_LOW) term_pain = 0.6;
            else if (r.chain.termination == TerminationReason::ACTIVATION_DECAY) term_pain = 0.5;
            else if (r.chain.termination == TerminationReason::SEED_DRIFT) term_pain = 0.4;
            else if (r.chain.termination == TerminationReason::COHERENCE_GATE) term_pain = 0.3;

            double quality_pain = std::max(0.0, 1.0 - r.quality);
            r.pain_score = 0.5 * term_pain + 0.5 * quality_pain;

            r.signal = reasoner_.extract_signals(r.chain);
            return r;
        }));
    }

    // Phase 2: Gather + merge (single-threaded, no locks needed on our state)
    for (auto& f : futures) {
        LocalResult r = f.get();
        if (r.chain.empty()) continue;

        ++last_chains_produced_;
        last_quality_sum_ += r.quality;

        // Merge pain (no lock — single thread)
        auto it = seed_pain_scores_.find(r.seed);
        if (it != seed_pain_scores_.end()) {
            it->second = 0.7 * it->second + 0.3 * r.pain_score;
        } else {
            seed_pain_scores_[r.seed] = r.pain_score;
        }

        // Track context-dependent quality for superposition heuristic
        // + record activation for FlexEmbedding growth tracking
        auto& concept_store = embeddings_.concept_embeddings();
        concept_store.record_activation(r.seed, tick_);
        for (size_t si = 1; si < r.chain.steps.size(); ++si) {
            superposition_tracker_.record(
                r.chain.steps[si].target_id,
                r.chain.steps[si].relation,
                r.quality);
            concept_store.record_activation(r.chain.steps[si].target_id, tick_);
        }

        // Collect errors and store episodes (single-threaded merge)
        error_collector_.collect_from_chain(r.chain, r.signal);

        Episode ep = episodic_memory_.from_chain(r.chain, r.seed);
        ep.quality = r.quality;
        episodic_memory_.store(ep);
        ++last_episodes_stored_;
    }
}

// =============================================================================
// Sleep Phase — Replay and consolidate episodes
// =============================================================================

void CoLearnLoop::sleep_phase() {
    last_consolidation_ = ConsolidationResult{};

    auto replay_episodes = episodic_memory_.select_for_replay(
        config_.sleep_replay_count,
        config_.replay_weight_quality,
        config_.replay_weight_recency,
        config_.replay_weight_novelty);

    if (replay_episodes.empty()) return;

    // Consolidate
    last_consolidation_ = extractor_.consolidate_batch(replay_episodes);

    // Mark episodes as replayed and consolidated
    for (const auto* ep : replay_episodes) {
        episodic_memory_.mark_replayed(ep->id);
        // Consolidation strength increases with each replay
        double new_strength = std::min(1.0,
            ep->consolidation_strength + 0.2);
        episodic_memory_.mark_consolidated(ep->id, new_strength);
    }
}

// =============================================================================
// Train Phase — Retrain CMs on updated graph
// =============================================================================

void CoLearnLoop::train_phase() {
    last_train_stats_ = ConceptTrainerStats{};
    pre_train_avg_loss_ = 0.0;
    last_correction_stats_ = {};
    last_superposition_enabled_ = 0;
    last_superposition_trained_ = 0;

    // --- Superposition enabling heuristic (Step 7) ---
    // Check which concepts show context-dependent behavior and enable superposition
    for (const auto& [cid, cs] : superposition_tracker_.stats) {
        ConceptModel* m = registry_.get_model(cid);
        if (!m) continue;
        if (m->superposition().enabled > 0.5) continue;  // already enabled

        if (superposition_tracker_.should_enable(
                cid, config_.superposition_min_observations,
                config_.superposition_min_std)) {
            // Enable superposition with small random initialization
            auto& sp = m->superposition();
            sp.enabled = 1.0;
            double scale = 0.01;
            for (size_t i = 0; i < sp.u.size(); ++i) {
                double x = std::sin(static_cast<double>(i * 71 + cid * 37)) * 43758.5453;
                x = x - std::floor(x);
                sp.u[i] = (x * 2.0 - 1.0) * scale;
            }
            for (size_t i = 0; i < sp.v.size(); ++i) {
                double x = std::sin(static_cast<double>(i * 53 + cid * 19)) * 43758.5453;
                x = x - std::floor(x);
                sp.v[i] = (x * 2.0 - 1.0) * scale;
            }
            for (size_t i = 0; i < sp.keys.size(); ++i) {
                double x = std::sin(static_cast<double>(i * 97 + cid * 41)) * 43758.5453;
                x = x - std::floor(x);
                sp.keys[i] = (x * 2.0 - 1.0) * scale;
            }
            ++last_superposition_enabled_;
        }
    }

    // Also retrain concepts that have correction samples (error-driven)
    const auto& changes = extractor_.cumulative_changes();
    std::vector<ConceptId> to_retrain;
    for (const auto& [cid, cum_delta] : changes) {
        if (cum_delta >= config_.retrain_threshold) {
            to_retrain.push_back(cid);
        }
    }

    // Add concepts with correction samples even if they didn't hit retrain threshold
    for (ConceptId cid : registry_.get_model_ids()) {
        if (!error_collector_.get_corrections(cid).empty()) {
            if (std::find(to_retrain.begin(), to_retrain.end(), cid) == to_retrain.end()) {
                to_retrain.push_back(cid);
            }
        }
    }

    if (to_retrain.empty()) {
        error_collector_.clear();
        return;
    }

    // Compute average loss before training
    size_t model_count = 0;
    for (ConceptId cid : to_retrain) {
        const ConceptModel* m = registry_.get_model(cid);
        if (m && m->is_converged()) {
            pre_train_avg_loss_ += m->final_loss();
            ++model_count;
        }
    }
    if (model_count > 0) {
        pre_train_avg_loss_ /= static_cast<double>(model_count);
    }

    registry_.ensure_models_for(ltm_);

    // Fine-tuning: small LR to avoid catastrophic forgetting
    ConceptTrainerConfig trainer_config;
    trainer_config.model_config.max_epochs = config_.retrain_epochs;
    trainer_config.model_config.learning_rate = config_.retrain_learning_rate;
    trainer_config.refined_epochs = config_.retrain_refined_epochs;
    trainer_config.kan_learning_rate = config_.retrain_kan_lr;
    ConceptTrainer trainer(trainer_config);

    static const size_t RECALL_HASH = std::hash<std::string>{}("recall");
    const auto& concept_store = embeddings_.concept_embeddings();

    double loss_sum = 0.0;
    for (ConceptId cid : to_retrain) {
        ConceptModel* m = registry_.get_model(cid);
        if (!m) continue;

        auto samples = trainer.generate_samples(cid, embeddings_, ltm_);

        // Inject correction samples from error-driven learning
        const auto& corrections = error_collector_.get_corrections(cid);
        for (const auto& corr : corrections) {
            TrainingSample ts;
            ts.relation_embedding = embeddings_.get_relation_embedding(corr.relation);
            ts.context_embedding = embeddings_.make_target_embedding(RECALL_HASH, cid, corr.target_concept);
            ts.target = corr.corrected_target;
            ts.weight = corr.sample_weight;
            samples.push_back(ts);
            ++last_correction_stats_.samples_injected;
        }

        if (samples.empty()) continue;

        // Hold-out validation: 80% train, 20% validate
        // This is a real validation gate — model must generalize beyond training data
        size_t val_count = samples.size() / 5;
        if (val_count < 1 && samples.size() > 2) val_count = 1;

        std::vector<TrainingSample> val_samples;
        if (val_count > 0) {
            val_samples.assign(samples.end() - static_cast<ptrdiff_t>(val_count), samples.end());
            samples.resize(samples.size() - val_count);
        }

        // Pre-retrain: compute MSE on validation set (or full set if no holdout)
        const auto& eval_samples = val_samples.empty() ? samples : val_samples;
        double pre_mse = 0.0;
        for (const auto& sample : eval_samples) {
            double pred = m->predict(sample.relation_embedding, sample.context_embedding);
            double err = pred - sample.target;
            pre_mse += err * err * sample.weight;
        }
        pre_mse /= static_cast<double>(eval_samples.size());

        // Save model state for potential rollback
        std::array<double, CM_FLAT_SIZE> saved_params;
        m->to_flat(saved_params);

        // Base bilinear training (on train set only)
        auto result = m->train(samples, trainer_config.model_config);

        // Refined KAN training (like train_all Phase 2)
        FlexEmbedding concept_from = concept_store.get_or_default(cid);
        auto outgoing = ltm_.get_outgoing_relations(cid);
        RefinedAdamState adam_state;
        for (size_t epoch = 0; epoch < trainer_config.refined_epochs; ++epoch) {
            for (const auto& rel : outgoing) {
                FlexEmbedding concept_to = concept_store.get_or_default(rel.target);
                FlexEmbedding rel_emb = embeddings_.get_relation_embedding(rel.type);
                FlexEmbedding ctx_emb = embeddings_.make_target_embedding(RECALL_HASH, cid, rel.target);
                m->train_refined(rel_emb, ctx_emb, concept_from, concept_to,
                                 rel.weight, trainer_config.kan_learning_rate, adam_state);
            }
        }

        // Superposition training (Step 6): train u/v/keys if enabled
        if (m->superposition().enabled > 0.5) {
            for (size_t epoch = 0; epoch < config_.superposition_epochs; ++epoch) {
                for (const auto& rel : outgoing) {
                    FlexEmbedding rel_emb = embeddings_.get_relation_embedding(rel.type);
                    FlexEmbedding ctx_emb = embeddings_.make_target_embedding(RECALL_HASH, cid, rel.target);
                    auto ctx_query = reasoner_.project_context_for_superposition(ctx_emb);
                    m->train_superposition_step(
                        rel_emb, ctx_emb, rel.weight,
                        config_.superposition_learning_rate, ctx_query);
                }
            }
            ++last_superposition_trained_;
        }

        // Post-retrain: compute MSE on validation set
        double post_mse = 0.0;
        for (const auto& sample : eval_samples) {
            double pred = m->predict(sample.relation_embedding, sample.context_embedding);
            double err = pred - sample.target;
            post_mse += err * err * sample.weight;
        }
        post_mse /= static_cast<double>(eval_samples.size());

        // Validation: only keep if MSE improved (or at least didn't get worse)
        if (post_mse > pre_mse) {
            // Retraining made predictions worse — rollback
            m->from_flat(saved_params);
            ++last_train_stats_.models_rolled_back;
        } else {
            // Keep retrained model
            ++last_train_stats_.models_trained;
            last_train_stats_.total_epochs += result.epochs_run;
            loss_sum += result.final_loss;
            if (result.converged) ++last_train_stats_.models_converged;
        }
    }

    if (last_train_stats_.models_trained > 0) {
        last_train_stats_.avg_final_loss = loss_sum / static_cast<double>(last_train_stats_.models_trained);
    }

    // Capture error correction stats before clearing
    last_correction_stats_.terminal = error_collector_.terminal_count();
    last_correction_stats_.quality_drop = error_collector_.quality_drop_count();
    last_correction_stats_.success = error_collector_.success_count();

    // Clear cumulative changes only for retrained concepts
    extractor_.clear_retrained(to_retrain);
    error_collector_.clear();

    // FlexEmbedding growth/shrink evaluation (every EVAL_INTERVAL ticks)
    last_embeddings_grown_ = 0;
    last_embeddings_shrunk_ = 0;
    if (tick_ > 0 && tick_ % FlexConfig::EVAL_INTERVAL == 0) {
        auto resize_result = embeddings_.concept_embeddings().evaluate_and_resize(tick_);
        last_embeddings_grown_ = resize_result.grown;
        last_embeddings_shrunk_ = resize_result.shrunk;
    }
}

// =============================================================================
// run_cycle — Full wake/sleep/train cycle
// =============================================================================

CoLearnLoop::CycleResult CoLearnLoop::run_cycle() {
    ++cycle_count_;
    ++tick_;

    wake_phase();
    extractor_.set_cycle(cycle_count_);
    sleep_phase();
    train_phase();

    // Build result
    CycleResult result;
    result.cycle_number = cycle_count_;
    result.chains_produced = last_chains_produced_;
    result.episodes_stored = last_episodes_stored_;

    if (last_chains_produced_ > 0) {
        result.avg_chain_quality = last_quality_sum_ / static_cast<double>(last_chains_produced_);
    }

    result.consolidation = last_consolidation_;
    result.models_retrained = last_train_stats_.models_trained;
    result.models_converged = last_train_stats_.models_converged;
    result.models_rolled_back = last_train_stats_.models_rolled_back;
    result.avg_loss_before = pre_train_avg_loss_;
    result.avg_loss_after = last_train_stats_.avg_final_loss;

    // Error-driven learning stats
    result.correction_samples_injected = last_correction_stats_.samples_injected;
    result.terminal_corrections = last_correction_stats_.terminal;
    result.quality_drop_corrections = last_correction_stats_.quality_drop;
    result.success_reinforcements = last_correction_stats_.success;

    // Superposition stats
    result.superposition_enabled_count = last_superposition_enabled_;
    result.superposition_trained_count = last_superposition_trained_;

    // FlexEmbedding growth/shrink stats
    result.embeddings_grown = last_embeddings_grown_;
    result.embeddings_shrunk = last_embeddings_shrunk_;

    // Track quality delta
    result.quality_delta = result.avg_chain_quality - last_avg_quality_;
    last_avg_quality_ = result.avg_chain_quality;

    return result;
}

std::vector<CoLearnLoop::CycleResult> CoLearnLoop::run_cycles(size_t n) {
    std::vector<CycleResult> results;
    results.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        results.push_back(run_cycle());
    }
    return results;
}

// =============================================================================
// Continuous mode — background thread running wake/sleep/train in a loop
// =============================================================================

void CoLearnLoop::start_continuous() {
    if (continuous_running_.load()) return;
    continuous_stop_.store(false);
    continuous_running_.store(true);
    continuous_thread_ = std::thread([this]() {
        while (!continuous_stop_.load()) {
            CycleResult result = run_cycle();
            if (cycle_callback_) {
                cycle_callback_(result);
            }
            if (config_.continuous_interval_ms > 0) {
                std::this_thread::sleep_for(
                    std::chrono::milliseconds(config_.continuous_interval_ms));
            }
        }
        continuous_running_.store(false);
    });
}

void CoLearnLoop::stop_continuous() {
    if (!continuous_running_.load()) return;
    continuous_stop_.store(true);
    if (continuous_thread_.joinable()) {
        continuous_thread_.join();
    }
}

bool CoLearnLoop::is_running() const {
    return continuous_running_.load();
}

void CoLearnLoop::set_cycle_callback(CycleCallback cb) {
    cycle_callback_ = std::move(cb);
}

} // namespace brain19
