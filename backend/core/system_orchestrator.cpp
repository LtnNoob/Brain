#include "system_orchestrator.hpp"
#include "../bootstrap/foundation_concepts.hpp"
#include "../cmodel/concept_pattern_engine.hpp"
#include "../language/language_training.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <unordered_set>

namespace brain19 {

// ─── Construction / Destruction ──────────────────────────────────────────────

SystemOrchestrator::SystemOrchestrator()
    : config_()
{}

SystemOrchestrator::SystemOrchestrator(Config config)
    : config_(std::move(config))
{}

SystemOrchestrator::~SystemOrchestrator() {
    if (running_) {
        shutdown();
    }
}

// ─── Logging ─────────────────────────────────────────────────────────────────

void SystemOrchestrator::log(const std::string& msg) const {
    std::cerr << "[Brain19] " << msg << "\n";
}

// ─── Initialize ──────────────────────────────────────────────────────────────

bool SystemOrchestrator::initialize() {
    if (running_) {
        log("Already running");
        return false;
    }

    log("Initializing Brain19...");

    try {
        // ── Stage 1: LTM ────────────────────────────────────────────────
        log("  [1/15] LTM...");
        ltm_ = std::make_unique<LongTermMemory>();
        init_stage_ = 1;

        // ── Stage 2: WAL + Persistence ────────────────────────────────
        if (config_.enable_persistence) {
            log("  [2/15] Persistence...");
            std::filesystem::create_directories(config_.data_dir);

            if (config_.enable_wal) {
                auto wal_path = config_.data_dir + "/brain19.wal";
                wal_ = std::make_unique<persistent::WALWriter>(wal_path);
                if (wal_->is_open()) {
                    log("    WAL opened: " + wal_path);
                } else {
                    log("    WAL failed to open (continuing without WAL)");
                    wal_.reset();
                }
            }
        }
        init_stage_ = 2;

        // ── Stage 3: BrainController + STM ──────────────────────────────
        log("  [3/15] BrainController...");
        brain_ = std::make_unique<BrainController>();
        brain_->initialize();
        init_stage_ = 3;

        // ── Stage 4: ConceptModels ──────────────────────────────────────
        log("  [4/15] ConceptModels...");
        embeddings_ = std::make_unique<EmbeddingManager>();
        registry_ = std::make_unique<ConceptModelRegistry>();
        trainer_ = std::make_unique<ConceptTrainer>();
        init_stage_ = 4;

        // ── Stage 5: CognitiveDynamics ──────────────────────────────────
        log("  [5/15] CognitiveDynamics...");
        cognitive_ = std::make_unique<CognitiveDynamics>();
        if (config_.enable_gdo) {
            gdo_ = std::make_unique<GlobalDynamicsOperator>(config_.gdo_config);
        }
        init_stage_ = 5;

        // ── Stage 6: CuriosityEngine ────────────────────────────────────
        log("  [6/15] CuriosityEngine...");
        curiosity_ = std::make_unique<CuriosityEngine>();
        goal_queue_ = std::make_unique<GoalQueue>(20);
        init_stage_ = 6;

        // ── Stage 7: KANAdapter ─────────────────────────────────────────
        log("  [7/15] KANAdapter...");
        kan_adapter_ = std::make_unique<KANAdapter>();
        init_stage_ = 7;

        // ── Stage 8: UnderstandingLayer + MiniLLMs ──────────────────────
        log("  [8/15] UnderstandingLayer...");
        understanding_ = std::make_unique<UnderstandingLayer>();
        // Register ConceptPatternEngine (per-concept pattern weights + ConceptModel predictions)
        understanding_->register_mini_llm(
            std::make_unique<ConceptPatternEngine>(*registry_, *embeddings_));
        init_stage_ = 8;

        // ── Stage 9: KAN-LLM Hybrid ────────────────────────────────────
        log("  [9/15] KAN-LLM Hybrid...");
        kan_validator_ = std::make_unique<KanValidator>();
        domain_manager_ = std::make_unique<DomainManager>();
        refinement_loop_ = std::make_unique<RefinementLoop>(*kan_validator_);
        init_stage_ = 9;

        // ── Stage 10: IngestionPipeline ─────────────────────────────────
        log("  [10/15] IngestionPipeline...");
        ingestion_ = std::make_unique<IngestionPipeline>(*ltm_);
        wiki_importer_ = std::make_unique<WikipediaImporter>();
        init_stage_ = 10;

        // ── Stage 11: ChatInterface (Template-Engine, kein LLM) ─────────
        log("  [11/15] ChatInterface...");
        chat_ = std::make_unique<ChatInterface>();
        log("    Knowledge-only mode (Template-Engine)");

        // ── Stage 11b: KAN Language Engine ───────────────────────────────
        log("  [11b] KAN Language Engine...");
        {
            LanguageConfig lang_config;
            language_engine_ = std::make_unique<KANLanguageEngine>(
                lang_config, *ltm_, *registry_, *embeddings_);
            language_engine_->initialize();
            log("    Language engine initialized (" +
                std::to_string(language_engine_->tokenizer().vocab_size()) + " tokens)");
        }
        init_stage_ = 11;

        // ── Stage 12: Shared Wrappers ───────────────────────────────────
        log("  [12/15] Shared wrappers...");
        shared_ltm_ = std::make_unique<SharedLTM>(*ltm_);
        shared_stm_ = std::make_unique<SharedSTM>(*brain_->get_stm_mutable());
        shared_registry_ = std::make_unique<SharedRegistry>(*registry_);
        shared_embeddings_ = std::make_unique<SharedEmbeddings>(*embeddings_);
        init_stage_ = 12;

        // ── Stage 13: Streams ───────────────────────────────────────────
        log("  [13/15] Streams...");
        {
            StreamConfig scfg;
            if (config_.max_streams > 0) {
                scfg.max_streams = static_cast<uint32_t>(config_.max_streams);
            }
            stream_orch_ = std::make_unique<StreamOrchestrator>(
                *shared_ltm_, *shared_stm_, *shared_registry_, *shared_embeddings_, scfg);

            SchedulerConfig sched_cfg;
            if (config_.max_streams > 0) {
                sched_cfg.total_max_streams = static_cast<uint32_t>(config_.max_streams);
            }
            stream_sched_ = std::make_unique<StreamScheduler>(*stream_orch_, sched_cfg);

            if (config_.enable_monitoring) {
                stream_monitor_ = std::make_unique<StreamMonitor>(*stream_orch_, *stream_sched_);
            }
        }
        init_stage_ = 13;

        // ── Stage 14: Evolution ─────────────────────────────────────────
        log("  [14/15] Evolution...");
        pattern_discovery_ = std::make_unique<PatternDiscovery>(*ltm_);
        epistemic_promotion_ = std::make_unique<EpistemicPromotion>(*ltm_);
        concept_proposer_ = std::make_unique<ConceptProposer>(*ltm_);
        init_stage_ = 14;

        // ── Stage 15: Bootstrap Foundation ──────────────────────────────
        if (config_.seed_foundation && ltm_->get_all_concept_ids().empty()) {
            log("  [15/15] Seeding foundation concepts...");
            seed_foundation();
        } else {
            log("  [15/15] Foundation already present or disabled");
        }
        init_stage_ = 15;

        // ── ThinkingPipeline ────────────────────────────────────────────
        thinking_ = std::make_unique<ThinkingPipeline>(config_.thinking_config);

        // Create active context
        active_context_ = brain_->create_context();

        // Ensure ConceptModels for all concepts
        registry_->ensure_models_for(*ltm_);

        // Rebuild dimensional context now that LTM is populated
        if (language_engine_ && language_engine_->is_ready()) {
            language_engine_->rebuild_dimensional_context();
        }

        // Initial ConceptModel training from KG structure
        // Without this, all predictions are ~0.50 (untrained) and KAN validation
        // always refutes. Train once at startup so existing KG data produces
        // meaningful predictions immediately.
        if (trainer_ && registry_->size() > 0) {
            log("    Training ConceptModels from KG...");
            auto stats = trainer_->train_all(*registry_, *embeddings_, *ltm_);
            log("    Trained " + std::to_string(stats.models_trained) + " models ("
                + std::to_string(stats.models_converged) + " converged, avg loss "
                + std::to_string(stats.avg_final_loss) + ")");
        }

        // Graph densification — only for small graphs where topology-based inference
        // is reliable. Large graphs with noisy wave data amplify errors through
        // transitive closure, so densification is skipped.
        {
            size_t concept_count = ltm_->get_all_concept_ids().size();
            if (concept_count < 1000) {
                log("    Graph densification (" + std::to_string(concept_count) + " concepts)...");
                GraphDensifier densifier(*ltm_);
                auto dens_result = densifier.densify();
                log("    Densified: +" + std::to_string(dens_result.relations_added)
                    + " relations (density " + std::to_string(dens_result.density_before)
                    + " -> " + std::to_string(dens_result.density_after) + ")");
                for (const auto& [phase, count] : dens_result.phase_counts) {
                    log("      " + phase + ": +" + std::to_string(count));
                }
                for (const auto& [type, count] : dens_result.type_distribution) {
                    log("      type " + type + ": " + std::to_string(count));
                }
                auto samples = densifier.sample_generated(100);
                log("    Quality sample (" + std::to_string(samples.size()) + " relations):");
                for (size_t i = 0; i < samples.size(); ++i) {
                    log("      [" + std::to_string(i+1) + "] "
                        + samples[i].source_label + " --["
                        + samples[i].type_name + "]--> "
                        + samples[i].target_label
                        + " (w=" + std::to_string(samples[i].weight) + ")");
                }
            } else {
                log("    Skipping densification (large graph: "
                    + std::to_string(concept_count) + " concepts, density "
                    + std::to_string(double(ltm_->total_relation_count()) / concept_count)
                    + " — existing data sufficient)");
            }
        }

        // Rebuild dimensional context with denser graph
        if (language_engine_ && language_engine_->is_ready()) {
            language_engine_->rebuild_dimensional_context();
        }

        // KAN decoder training from KG relations
        if (language_engine_ && language_engine_->is_ready()) {
            log("    Training KAN decoder from KG relations...");
            LanguageTraining lang_trainer(*language_engine_, *ltm_);
            LanguageConfig lang_config;
            auto lang_result = lang_trainer.train_stage1(lang_config);
            log("    KAN decoder: " + std::to_string(lang_result.epochs_run) + " epochs, loss="
                + std::to_string(lang_result.final_loss));
        }

        // Start GDO
        if (gdo_) {
            gdo_->set_thinking_callback([this](const std::vector<ConceptId>& seeds) {
                std::lock_guard<std::recursive_mutex> lock(subsystem_mtx_);
                if (running_ && thinking_) {
                    size_t concepts_before = ltm_->get_all_concept_ids().size();

                    auto result = thinking_->execute(
                        seeds, active_context_,
                        *ltm_, *brain_->get_stm_mutable(), *brain_,
                        *cognitive_, *curiosity_,
                        *registry_, *embeddings_,
                        understanding_.get(),
                        kan_validator_.get(),
                        gdo_.get(),
                        refinement_loop_.get()
                    );
                    run_evolution_after_thinking(result);

                    // Store results for surfacing in next ask() call
                    size_t concepts_after = ltm_->get_all_concept_ids().size();
                    size_t new_count = (concepts_after > concepts_before)
                        ? (concepts_after - concepts_before) : 0;

                    GDOThinkingResult gdo_result;
                    gdo_result.seeds = seeds;
                    gdo_result.proposals_generated =
                        result.understanding.total_proposals_generated;
                    gdo_result.new_concepts_created = new_count;
                    gdo_result.duration_ms = result.total_duration_ms;

                    // Collect labels of newly created concepts
                    if (new_count > 0) {
                        auto all_ids = ltm_->get_all_concept_ids();
                        // New concepts are at the end (sequential IDs)
                        for (size_t i = all_ids.size() - new_count; i < all_ids.size(); ++i) {
                            auto cinfo = ltm_->retrieve_concept(all_ids[i]);
                            if (cinfo) {
                                gdo_result.new_concept_labels.push_back(cinfo->label);
                            }
                        }
                    }

                    {
                        std::lock_guard<std::mutex> glock(gdo_results_mtx_);
                        gdo_thinking_results_.push_back(std::move(gdo_result));
                        // Cap buffer to avoid unbounded growth
                        if (gdo_thinking_results_.size() > 10) {
                            gdo_thinking_results_.erase(
                                gdo_thinking_results_.begin());
                        }
                    }
                }
            });
            gdo_->start();
            log("    GDO started");
        }

        // Start streams
        stream_sched_->start();
        if (stream_monitor_) {
            stream_monitor_->start();
        }

        // Wire stream alert callback
        if (stream_orch_) {
            stream_orch_->set_alert_callback([this](const std::string& msg) {
                std::lock_guard<std::mutex> lock(alert_log_mtx_);
                stream_alerts_.push_back(msg);
                log("  [Stream Alert] " + msg);
            });
        }

        running_ = true;

        // Start periodic task thread
        periodic_running_ = true;
        periodic_thread_ = std::thread([this]() { periodic_task_loop(); });

        log("Brain19 initialized successfully!");
        log("  Concepts: " + std::to_string(concept_count()));
        log("  Relations: " + std::to_string(relation_count()));

        return true;

    } catch (const std::exception& e) {
        log("Initialization failed at stage " + std::to_string(init_stage_) + ": " + e.what());
        cleanup_from_stage(init_stage_);
        return false;
    }
}

// ─── Shutdown ────────────────────────────────────────────────────────────────

void SystemOrchestrator::shutdown() {
    if (!running_) return;
    log("Shutting down Brain19...");

    // Stop periodic task thread first (while running_ is still true)
    periodic_running_ = false;
    if (periodic_thread_.joinable()) {
        log("  Stopping periodic tasks...");
        periodic_thread_.join();
    }

    // Auto-checkpoint on shutdown (before setting running_ = false)
    if (config_.enable_persistence && wal_) {
        log("  Final checkpoint on shutdown...");
        create_checkpoint("shutdown");
        wal_->checkpoint();
    }

    running_ = false;

    // Stop GDO before clearing callbacks
    if (gdo_) {
        log("  Stopping GDO...");
        gdo_->set_thinking_callback(nullptr);
        gdo_->stop();
    }

    // Clear stream alert callback before stopping (prevents use-after-free)
    if (stream_orch_) {
        stream_orch_->set_alert_callback(nullptr);
    }

    // Reverse order of initialization
    // 13: Streams
    if (stream_monitor_) {
        log("  Stopping stream monitor...");
        stream_monitor_->stop();
    }
    if (stream_sched_) {
        log("  Stopping stream scheduler...");
        stream_sched_->shutdown(std::chrono::milliseconds{3000});
    }
    if (stream_orch_) {
        log("  Stopping stream orchestrator...");
        stream_orch_->shutdown(std::chrono::milliseconds{3000});
    }

    // Destroy stream objects BEFORE brain controller frees STM.
    // ThinkStream destructors call stm_.destroy_context(), so the STM must be alive.
    stream_monitor_.reset();
    stream_sched_.reset();
    stream_orch_.reset();

    // Destroy context
    if (brain_ && active_context_ != 0) {
        brain_->destroy_context(active_context_);
        active_context_ = 0;
    }

    // Shutdown brain controller
    if (brain_) {
        brain_->shutdown();
    }

    // Reset all in reverse order
    thinking_.reset();
    concept_proposer_.reset();
    epistemic_promotion_.reset();
    pattern_discovery_.reset();
    shared_embeddings_.reset();
    shared_registry_.reset();
    shared_stm_.reset();
    shared_ltm_.reset();
    chat_.reset();
    ingestion_.reset();
    wiki_importer_.reset();
    refinement_loop_.reset();
    domain_manager_.reset();
    kan_validator_.reset();
    understanding_.reset();
    kan_adapter_.reset();
    goal_queue_.reset();
    curiosity_.reset();
    gdo_.reset();
    cognitive_.reset();
    trainer_.reset();
    registry_.reset();
    embeddings_.reset();
    brain_.reset();
    wal_.reset();
    persistent_ltm_.reset();
    ltm_.reset();

    init_stage_ = 0;
    log("Brain19 shut down.");
}

// ─── Cleanup on partial init failure ─────────────────────────────────────────

void SystemOrchestrator::cleanup_from_stage(int stage) {
    // Clean up in reverse from the stage that succeeded
    thinking_.reset();  // Created after stage 15, always safe to reset
    if (stage >= 14) { concept_proposer_.reset(); epistemic_promotion_.reset(); pattern_discovery_.reset(); }
    if (stage >= 13) { stream_monitor_.reset(); stream_sched_.reset(); stream_orch_.reset(); }
    if (stage >= 12) { shared_embeddings_.reset(); shared_registry_.reset(); shared_stm_.reset(); shared_ltm_.reset(); }
    if (stage >= 11) { language_engine_.reset(); chat_.reset(); }
    if (stage >= 10) { ingestion_.reset(); wiki_importer_.reset(); }
    if (stage >= 9)  { refinement_loop_.reset(); domain_manager_.reset(); kan_validator_.reset(); }
    if (stage >= 8)  { understanding_.reset(); }
    if (stage >= 7)  { kan_adapter_.reset(); }
    if (stage >= 6)  { goal_queue_.reset(); curiosity_.reset(); }
    if (stage >= 5)  { if (gdo_) { gdo_->stop(); gdo_.reset(); } cognitive_.reset(); }
    if (stage >= 4)  { trainer_.reset(); registry_.reset(); embeddings_.reset(); }
    if (stage >= 3)  { if (brain_) brain_->shutdown(); brain_.reset(); }
    if (stage >= 2)  { wal_.reset(); }
    if (stage >= 1)  { persistent_ltm_.reset(); ltm_.reset(); }
    init_stage_ = 0;
}

// ─── Foundation Seeding ──────────────────────────────────────────────────────

void SystemOrchestrator::seed_foundation() {
    // Try configured path, then common alternatives
    std::vector<std::string> paths;
    if (!config_.foundation_file.empty())
        paths.push_back(config_.foundation_file);
    paths.push_back("data/foundation_full.json");       // from project root
    paths.push_back("../data/foundation_full.json");     // from backend/
    paths.push_back("data/foundation.json");
    paths.push_back("../data/foundation.json");

    for (const auto& path : paths) {
        if (FoundationConcepts::seed_from_file(*ltm_, path)) {
            log("    Seeded from file: " + path);
            log("    Concepts: " + std::to_string(ltm_->get_all_concept_ids().size()) +
                ", Relations: " + std::to_string(ltm_->total_relation_count()));
            return;
        }
    }
    log("    No foundation file found, falling back to hardcoded");
    FoundationConcepts::seed_all(*ltm_);
    log("    Seeded " + std::to_string(FoundationConcepts::concept_count()) +
        " concepts, " + std::to_string(FoundationConcepts::relation_count()) + " relations");
}

// ─── Chat ────────────────────────────────────────────────────────────────────

// ─── Seed-Finding Helpers (file-local) ────────────────────────────────────────

namespace {

static const std::unordered_set<std::string> SEED_STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "must",
    "i", "me", "my", "we", "our", "you", "your", "he", "she", "it",
    "they", "them", "their", "his", "her", "its",
    "what", "which", "who", "whom", "this", "that", "these", "those",
    "am", "if", "or", "and", "but", "not", "no", "so", "too", "very",
    "of", "in", "on", "at", "to", "for", "with", "by", "from", "about",
    "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "each",
    "every", "both", "few", "more", "most", "other", "some", "such",
    "only", "own", "same", "than", "just", "also", "now",
    "ein", "eine", "der", "die", "das", "ist", "sind", "war", "und",
    "oder", "nicht", "ich", "du", "er", "sie", "es", "wir", "ihr",
    "von", "zu", "mit", "auf", "aus", "fuer", "ueber", "nach",
    "wie", "was", "wer", "wo", "warum",
};

std::string seed_to_lower(const std::string& s) {
    std::string result = s;
    std::transform(result.begin(), result.end(), result.begin(), ::tolower);
    return result;
}

std::vector<std::string> seed_extract_keywords(const std::string& text) {
    std::string lower = seed_to_lower(text);
    std::vector<std::string> keywords;
    std::string word;
    for (char c : lower) {
        if (std::isalnum(static_cast<unsigned char>(c))) {
            word += c;
        } else {
            if (word.size() >= 2 && SEED_STOP_WORDS.find(word) == SEED_STOP_WORDS.end()) {
                keywords.push_back(word);
            }
            word.clear();
        }
    }
    if (word.size() >= 2 && SEED_STOP_WORDS.find(word) == SEED_STOP_WORDS.end()) {
        keywords.push_back(word);
    }
    return keywords;
}

size_t seed_levenshtein(const std::string& a, const std::string& b) {
    size_t m = a.size(), n = b.size();
    std::vector<size_t> prev(n + 1), curr(n + 1);
    for (size_t j = 0; j <= n; ++j) prev[j] = j;
    for (size_t i = 1; i <= m; ++i) {
        curr[0] = i;
        for (size_t j = 1; j <= n; ++j) {
            size_t cost = (a[i-1] == b[j-1]) ? 0 : 1;
            curr[j] = std::min({prev[j] + 1, curr[j-1] + 1, prev[j-1] + cost});
        }
        std::swap(prev, curr);
    }
    return prev[n];
}

struct ScoredSeed {
    ConceptId id;
    double score;
};

} // anonymous namespace

// ─── Ask (Multi-Strategy Semantic Matching) ──────────────────────────────────

ChatResponse SystemOrchestrator::ask(const std::string& question) {
    if (!running_ || !chat_) {
        ChatResponse resp{};
        resp.answer = "[Brain19 not running]";
        resp.used_llm = false;
        resp.contains_speculation = false;
        resp.llm_time_ms = 0.0;
        return resp;
    }

    // Update chat with total counts
    chat_->set_totals(ltm_->get_all_concept_ids().size(), ltm_->total_relation_count());

    std::lock_guard<std::recursive_mutex> lock(subsystem_mtx_);

    // Classify intent first
    auto intent = ChatInterface::classify_intent(question);

    // Multi-strategy seed finding
    std::string lower_q = seed_to_lower(question);
    auto keywords = seed_extract_keywords(question);
    std::vector<ScoredSeed> scored_seeds;

    for (auto cid : ltm_->get_all_concept_ids()) {
        auto info = ltm_->retrieve_concept(cid);
        if (!info) continue;

        std::string lower_label = seed_to_lower(info->label);
        std::string lower_def = seed_to_lower(info->definition);
        double score = 0.0;

        // Strategy 1: Exact label match
        if (lower_q == lower_label) {
            score += 10.0;
        }

        // Strategy 2: Label substring in query (weighted by label length)
        if (lower_label.size() >= 3 && lower_q.find(lower_label) != std::string::npos) {
            score += 5.0 * (static_cast<double>(lower_label.size()) / lower_q.size());
        }

        // Strategy 3: Keyword match on labels
        for (const auto& kw : keywords) {
            if (lower_label.find(kw) != std::string::npos) {
                score += 3.0;
            }
        }

        // Strategy 4: Keyword match on definitions
        for (const auto& kw : keywords) {
            if (lower_def.find(kw) != std::string::npos) {
                score += 1.5;
            }
        }

        // Strategy 5: Fuzzy match (Levenshtein) + prefix match
        for (const auto& kw : keywords) {
            if (kw.size() >= 3 && lower_label.size() >= 3) {
                size_t dist = seed_levenshtein(kw, lower_label);
                size_t max_len = std::max(kw.size(), lower_label.size());
                double similarity = 1.0 - (static_cast<double>(dist) / max_len);
                if (similarity >= 0.7) {
                    score += 2.0 * similarity;
                }
            }
            // Prefix match
            if (kw.size() >= 3 && lower_label.size() >= kw.size() &&
                lower_label.substr(0, kw.size()) == kw) {
                score += 2.5;
            }
        }

        // Strategy 6: Multi-word label decomposition
        {
            std::vector<std::string> label_words;
            std::string w;
            for (char c : lower_label) {
                if (std::isalnum(static_cast<unsigned char>(c))) {
                    w += c;
                } else if (!w.empty()) {
                    label_words.push_back(w);
                    w.clear();
                }
            }
            if (!w.empty()) label_words.push_back(w);

            for (const auto& lw : label_words) {
                if (lw.size() < 3) continue;
                for (const auto& kw : keywords) {
                    if (lw == kw) {
                        score += 3.5;
                    } else if (lw.size() >= 4 && kw.size() >= 4 &&
                               lw.substr(0, 4) == kw.substr(0, 4)) {
                        score += 2.0;
                    }
                }
            }
        }

        // Epistemic trust boost
        score *= (0.8 + 0.2 * info->epistemic.trust);

        if (score > 0.0) {
            scored_seeds.push_back({cid, score});
        }
    }

    // Sort by score, take top text-matched seeds
    std::sort(scored_seeds.begin(), scored_seeds.end(),
        [](const ScoredSeed& a, const ScoredSeed& b) { return a.score > b.score; });

    std::vector<ConceptId> seeds;
    size_t max_seeds = 8;
    for (size_t i = 0; i < std::min(scored_seeds.size(), max_seeds); ++i) {
        seeds.push_back(scored_seeds[i].id);
    }

    // ── Strategy 7: Embedding similarity expansion ──
    // For top text-matched seeds, find embedding-similar concepts via
    // ConceptEmbeddingStore::most_similar() (cosine similarity on 10D
    // embeddings trained by MicroModels). This catches semantically
    // related concepts that text matching misses.
    std::vector<ThinkingContext::EmbeddingSeed> embedding_discoveries;
    if (!seeds.empty() && embeddings_) {
        auto& emb_store = embeddings_->concept_embeddings();
        std::unordered_set<ConceptId> seed_set(seeds.begin(), seeds.end());
        size_t expand_from = std::min(seeds.size(), size_t(3));  // top 3 text matches

        for (size_t i = 0; i < expand_from; ++i) {
            auto similar = emb_store.most_similar(seeds[i], 5);
            for (const auto& [sim_cid, sim_score] : similar) {
                if (seed_set.count(sim_cid)) continue;
                if (sim_score < 0.3) continue;  // min similarity threshold

                // Add to scored_seeds with embedding-derived score
                double emb_score = 1.5 + (sim_score - 0.3) * 5.0;
                scored_seeds.push_back({sim_cid, emb_score});

                // Track for display
                auto cinfo = ltm_->retrieve_concept(sim_cid);
                ThinkingContext::EmbeddingSeed es;
                es.concept_id = sim_cid;
                es.similar_to = seeds[i];
                es.similarity = sim_score;
                es.label = cinfo ? cinfo->label : "?";
                embedding_discoveries.push_back(es);

                // Add to seeds if room
                if (seeds.size() < 12) {
                    seeds.push_back(sim_cid);
                    seed_set.insert(sim_cid);
                }
            }
        }
    }

    // Inject energy into GDO from user query
    if (gdo_ && !seeds.empty()) {
        gdo_->inject_energy(config_.gdo_config.injection_boost);
        gdo_->inject_seeds(seeds, 0.8);
    }

    // Run full cognitive pipeline and build ThinkingContext for response fusion
    if (!seeds.empty()) {
        auto thinking_result = run_thinking_cycle(seeds);

        // ── Build ThinkingContext from ThinkingResult ──
        ThinkingContext ctx;

        // Salient concepts: start with seeds (query-matched), then add
        // pipeline-discovered salient concepts that aren't already included
        std::unordered_set<ConceptId> salient_set;
        for (auto sid : seeds) {
            ctx.salient_concepts.push_back(sid);
            salient_set.insert(sid);
        }
        for (const auto& s : thinking_result.top_salient) {
            if (!salient_set.count(s.concept_id)) {
                ctx.salient_concepts.push_back(s.concept_id);
                salient_set.insert(s.concept_id);
            }
        }

        // Thought path summaries (top 5)
        for (const auto& path : thinking_result.best_paths) {
            if (ctx.thought_path_summaries.size() >= 5) break;
            std::string summary;
            for (size_t i = 0; i < path.nodes.size(); ++i) {
                if (i > 0) summary += " -> ";
                auto info = ltm_->retrieve_concept(path.nodes[i].concept_id);
                summary += info ? info->label : ("?" + std::to_string(path.nodes[i].concept_id));
            }
            ctx.thought_path_summaries.push_back(summary);
        }

        // ── Domain detection: keyword match + cognitive salience ──
        // Combines TWO signals:
        // 1. Keyword-match score: concepts directly matching the query text
        //    (these ARE what the user asked about — must be prioritized)
        // 2. Cognitive salience: concepts discovered by spreading activation
        //    (these provide enrichment/context — secondary signal)
        // This implements "KAN-Relations + Pattern Matching → Topic Detection"
        {
            // Build combined relevance score per concept
            // Normalize keyword scores to [0,1], then combine
            double max_keyword_score = 0.0;
            for (const auto& ss : scored_seeds) {
                if (ss.score > max_keyword_score) max_keyword_score = ss.score;
            }

            std::unordered_map<ConceptId, double> combined_score;

            // Keyword match signal (normalized, high weight)
            constexpr double KEYWORD_WEIGHT = 3.0;
            for (const auto& ss : scored_seeds) {
                double norm_kw = (max_keyword_score > 0.0)
                    ? (ss.score / max_keyword_score) : 0.0;
                combined_score[ss.id] = norm_kw * KEYWORD_WEIGHT;
            }

            // Cognitive salience signal (already [0,1])
            constexpr double SALIENCE_WEIGHT = 1.0;
            for (const auto& s : thinking_result.top_salient) {
                combined_score[s.concept_id] += s.salience * SALIENCE_WEIGHT;
            }

            // Activated but not salient/keyword: low signal
            for (auto cid : thinking_result.activated_concepts) {
                if (!combined_score.count(cid)) {
                    combined_score[cid] = 0.05;
                }
            }

            // Select top concepts for domain clustering (by combined score)
            std::vector<std::pair<ConceptId, double>> domain_candidates;
            domain_candidates.reserve(combined_score.size());
            for (auto& [cid, score] : combined_score) {
                domain_candidates.push_back({cid, score});
            }
            std::sort(domain_candidates.begin(), domain_candidates.end(),
                [](const auto& a, const auto& b) { return a.second > b.second; });
            if (domain_candidates.size() > 30) domain_candidates.resize(30);

            // Map concept → domain ancestor (walk up IS_A, max 2 hops)
            // Walk up 1 IS_A hop: "Physics" → "Science" (not all the way to "Knowledge")
            std::unordered_map<ConceptId, std::vector<ConceptId>> root_to_members;
            std::unordered_map<ConceptId, double> root_score;
            for (auto& [cid, score] : domain_candidates) {
                ConceptId current = cid;
                std::unordered_set<ConceptId> visited;
                for (size_t depth = 0; depth < 1; ++depth) {
                    if (visited.count(current)) break;
                    visited.insert(current);
                    auto rels = ltm_->get_outgoing_relations(current);
                    bool found = false;
                    for (const auto& r : rels) {
                        if (r.type == RelationType::IS_A && !visited.count(r.target)) {
                            current = r.target;
                            found = true;
                            break;
                        }
                    }
                    if (!found) break;
                }
                root_to_members[current].push_back(cid);
                root_score[current] += score;
            }

            // Convert to DomainInsight
            std::vector<ThinkingContext::DomainInsight> domains;
            for (auto& [root_id, members] : root_to_members) {
                if (members.empty()) continue;
                ThinkingContext::DomainInsight di;
                auto root_info = ltm_->retrieve_concept(root_id);
                di.domain_name = root_info ? root_info->label : "Domain-" + std::to_string(root_id);
                for (auto mid : members) {
                    di.concepts.push_back(mid);
                }
                // Relevance = combined score (normalized to [0,1] using max possible)
                double max_possible = KEYWORD_WEIGHT + SALIENCE_WEIGHT;
                di.relevance = std::min(1.0, root_score[root_id] / max_possible);
                domains.push_back(std::move(di));
            }
            // Sort by relevance descending
            std::sort(domains.begin(), domains.end(),
                [](const ThinkingContext::DomainInsight& a,
                   const ThinkingContext::DomainInsight& b) {
                    return a.relevance > b.relevance;
                });
            // Keep top domains (floor 0.25 to filter spreading-activation noise)
            for (auto& d : domains) {
                if (d.relevance < 0.25 && !ctx.detected_domains.empty()) break;
                ctx.detected_domains.push_back(std::move(d));
                if (ctx.detected_domains.size() >= 5) break;
            }
        }

        // ── KAN-Relations analysis ──
        // Priority: relations FROM seed concepts (query-matched), then
        // between salient concepts. Cap at 10 links total.
        {
            auto add_relation = [&](const RelationInfo& r) {
                if (ctx.relation_links.size() >= 10) return;
                auto src_info = ltm_->retrieve_concept(r.source);
                auto tgt_info = ltm_->retrieve_concept(r.target);
                if (!src_info || !tgt_info) return;

                ThinkingContext::RelationLink rl;
                rl.source = r.source;
                rl.target = r.target;
                rl.relation_name = relation_type_to_string(r.type);
                rl.weight = r.weight;
                rl.source_label = src_info->label;
                rl.target_label = tgt_info->label;
                ctx.relation_links.push_back(std::move(rl));
            };

            // First: relations FROM seed concepts (what the user asked about)
            for (auto sid : seeds) {
                if (ctx.relation_links.size() >= 10) break;
                auto rels = ltm_->get_outgoing_relations(sid);
                for (const auto& r : rels) {
                    add_relation(r);
                }
            }

            // Then: relations between non-seed salient concepts
            for (size_t i = seeds.size(); i < ctx.salient_concepts.size() && i < 8; ++i) {
                if (ctx.relation_links.size() >= 10) break;
                auto rels = ltm_->get_outgoing_relations(ctx.salient_concepts[i]);
                for (const auto& r : rels) {
                    // Only salient-to-salient for non-seed concepts
                    bool target_salient = false;
                    for (auto scid : ctx.salient_concepts) {
                        if (scid == r.target) { target_salient = true; break; }
                    }
                    if (target_salient) add_relation(r);
                }
            }
        }

        // ── Understanding Layer results → ThinkingContext ──
        const auto& uresult = thinking_result.understanding;

        // Meaning insights from MiniLLMs
        for (const auto& mp : uresult.meaning_proposals) {
            ThinkingContext::MeaningInsight mi;
            mi.interpretation = mp.interpretation;
            mi.confidence = mp.model_confidence;
            mi.source_model = mp.source_model;
            mi.source_concepts = mp.source_concepts;
            ctx.meaning_insights.push_back(std::move(mi));
        }

        // Hypothesis insights with KAN validation
        for (size_t i = 0; i < uresult.hypothesis_proposals.size(); ++i) {
            const auto& hp = uresult.hypothesis_proposals[i];
            ThinkingContext::HypothesisInsight hi;
            hi.statement = hp.hypothesis_statement;
            hi.confidence = hp.model_confidence;
            hi.source_model = hp.source_model;

            // Check if this hypothesis was KAN-validated
            if (i < thinking_result.validated_hypotheses.size()) {
                const auto& vr = thinking_result.validated_hypotheses[i];
                hi.kan_validated = true;
                if (vr.validated && vr.assessment.converged) {
                    hi.validation_status = "validated";
                } else if (!vr.validated) {
                    hi.validation_status = "refuted";
                } else {
                    hi.validation_status = "inconclusive";
                }
            }
            ctx.hypothesis_insights.push_back(std::move(hi));
        }

        // Topology A hypothesis insights (marked distinctly)
        for (const auto& ah : thinking_result.topology_a_hypotheses) {
            ThinkingContext::HypothesisInsight hi;
            hi.statement = ah.hypothesis_statement;
            hi.confidence = ah.model_confidence;
            hi.source_model = ah.source_model;
            ctx.hypothesis_insights.push_back(std::move(hi));
        }

        // Contradiction notes
        for (const auto& cp : uresult.contradiction_proposals) {
            ThinkingContext::ContradictionNote cn;
            cn.concept_a = cp.concept_a;
            cn.concept_b = cp.concept_b;
            cn.description = cp.contradiction_description;
            cn.severity = cp.severity;
            ctx.contradiction_notes.push_back(std::move(cn));
        }

        // Pipeline statistics
        ctx.steps_completed = thinking_result.steps_completed;
        ctx.thinking_duration_ms = thinking_result.total_duration_ms;
        ctx.total_proposals = uresult.total_proposals_generated;

        // Embedding discoveries from Strategy 7
        ctx.embedding_discoveries = std::move(embedding_discoveries);

        // ── Drain GDO autonomous thinking results ──
        {
            std::lock_guard<std::mutex> glock(gdo_results_mtx_);
            for (auto& gr : gdo_thinking_results_) {
                ThinkingContext::AutonomousInsight ai;
                ai.seed_concepts = std::move(gr.seeds);
                ai.discovered_labels = std::move(gr.new_concept_labels);
                ai.proposals_generated = gr.proposals_generated;
                ai.duration_ms = gr.duration_ms;
                ctx.autonomous_insights.push_back(std::move(ai));
            }
            gdo_thinking_results_.clear();
        }

        // ── Try KAN Language Engine first ──
        if (language_engine_ && language_engine_->is_ready()) {
            auto lang_result = language_engine_->generate(question);
            if (!lang_result.used_template && !lang_result.text.empty()) {
                ChatResponse resp;
                resp.answer = lang_result.text;
                resp.referenced_concepts = lang_result.activated_concepts;
                resp.contains_speculation = false;
                resp.used_llm = false;
                resp.intent = intent;
                return resp;
            }
            // Fall through to ChatInterface template-based response
        }

        return chat_->ask_with_thinking(question, *ltm_, ctx, intent);
    }

    return chat_->ask(question, *ltm_);
}

// ─── Ingestion ───────────────────────────────────────────────────────────────

IngestionResult SystemOrchestrator::ingest_text(const std::string& text, bool auto_approve) {
    if (!running_ || !ingestion_) {
        IngestionResult r;
        r.success = false;
        r.error_message = "Not running";
        return r;
    }

    std::lock_guard<std::recursive_mutex> lock(subsystem_mtx_);

    auto result = ingestion_->ingest_text(text, "", auto_approve);
    if (result.success && !result.stored_concept_ids.empty()) {
        registry_->ensure_models_for(result.stored_concept_ids);
    }
    return result;
}

IngestionResult SystemOrchestrator::ingest_wikipedia(const std::string& url) {
    if (!running_ || !wiki_importer_ || !ingestion_) {
        IngestionResult r;
        r.success = false;
        r.error_message = "Not running";
        return r;
    }

    // HTTP fetch outside lock (network I/O)
    auto proposal = wiki_importer_->import_from_url(url);
    if (!proposal) {
        IngestionResult r;
        r.success = false;
        r.error_message = "Failed to import from URL";
        return r;
    }

    std::lock_guard<std::recursive_mutex> lock(subsystem_mtx_);

    auto result = ingestion_->ingest_text(proposal->extracted_text, url, true);
    if (result.success && !result.stored_concept_ids.empty()) {
        registry_->ensure_models_for(result.stored_concept_ids);
    }
    return result;
}

// ─── Checkpoint ──────────────────────────────────────────────────────────────

void SystemOrchestrator::create_checkpoint(const std::string& tag) {
    if (!running_) return;

    std::lock_guard<std::recursive_mutex> lock(subsystem_mtx_);

    persistent::CheckpointManager::Options opts;
    opts.base_dir = config_.data_dir + "/checkpoints";
    opts.max_keep = config_.max_checkpoints;
    opts.tag = tag;

    persistent::CheckpointManager mgr(opts);

    auto stm_data = brain_->get_stm()->export_state();

    auto path = mgr.save(
        nullptr,    // PersistentLTM (not used with in-memory LTM)
        &stm_data,
        registry_.get(),
        nullptr,    // KAN modules
        nullptr,    // Cognitive state
        nullptr     // Config
    );

    mgr.rotate();
    log("Checkpoint saved: " + path);
}

bool SystemOrchestrator::restore_checkpoint(const std::string& checkpoint_dir) {
    if (!running_) return false;

    std::lock_guard<std::recursive_mutex> lock(subsystem_mtx_);

    STMSnapshotData stm_data;
    auto result = persistent::CheckpointRestore::restore(
        checkpoint_dir,
        static_cast<uint8_t>(persistent::Component::ALL),
        nullptr,     // PersistentLTM
        &stm_data,
        registry_.get(),
        nullptr,     // KAN modules
        nullptr,     // Cognitive state
        nullptr      // Config
    );

    if (result.success) {
        brain_->get_stm_mutable()->import_state(stm_data);
        log("Restored from checkpoint: " + checkpoint_dir);
    }
    return result.success;
}

// ─── Monitoring ──────────────────────────────────────────────────────────────

std::string SystemOrchestrator::get_status() const {
    std::lock_guard<std::recursive_mutex> lock(subsystem_mtx_);

    std::ostringstream ss;
    ss << "=== Brain19 Status ===\n";
    ss << "Running: " << (running_ ? "yes" : "no") << "\n";
    ss << "Concepts: " << concept_count() << "\n";
    ss << "Relations: " << relation_count() << "\n";
    ss << "ConceptModels: " << (registry_ ? registry_->size() : 0) << "\n";
    ss << "MiniLLMs: " << (understanding_ ? understanding_->get_mini_llm_count() : 0) << "\n";
    ss << "Chat LLM: " << (chat_ && chat_->is_llm_available() ? "available" : "unavailable") << "\n";
    ss << "Active context: " << active_context_ << "\n";
    if (stream_orch_) {
        ss << "Streams: " << stream_orch_->running_count() << "/" << stream_orch_->stream_count() << "\n";
    }
    return ss.str();
}

std::string SystemOrchestrator::get_stream_status() const {
    std::lock_guard<std::recursive_mutex> lock(subsystem_mtx_);
    if (!stream_monitor_) return "Stream monitoring disabled\n";

    auto snapshots = stream_monitor_->stream_snapshots();
    auto global = stream_monitor_->global_snapshot();
    std::ostringstream ss;
    ss << "=== Stream Status ===\n";
    ss << "Total throughput: " << global.total_throughput << " ticks/sec\n";
    ss << "Active streams: " << global.active_streams << "/" << global.total_streams << "\n";
    for (auto& snap : snapshots) {
        ss << "  Stream " << snap.stream_id
           << " [" << category_name(snap.category) << "]"
           << " tps=" << snap.ticks_per_sec
           << " idle=" << snap.idle_pct << "%"
           << " err=" << snap.errors << "\n";
    }
    return ss.str();
}

// ─── Access ──────────────────────────────────────────────────────────────────

size_t SystemOrchestrator::concept_count() const {
    std::lock_guard<std::recursive_mutex> lock(subsystem_mtx_);
    return ltm_ ? ltm_->get_all_concept_ids().size() : 0;
}

size_t SystemOrchestrator::relation_count() const {
    std::lock_guard<std::recursive_mutex> lock(subsystem_mtx_);
    if (!ltm_) return 0;
    return ltm_->total_relation_count();
}

std::vector<ConceptId> SystemOrchestrator::get_all_concept_ids() const {
    std::lock_guard<std::recursive_mutex> lock(subsystem_mtx_);
    return ltm_ ? ltm_->get_all_concept_ids() : std::vector<ConceptId>{};
}

std::optional<ConceptInfo> SystemOrchestrator::get_concept(ConceptId cid) const {
    std::lock_guard<std::recursive_mutex> lock(subsystem_mtx_);
    return ltm_ ? ltm_->retrieve_concept(cid) : std::nullopt;
}

std::vector<RelationInfo> SystemOrchestrator::get_outgoing_relations(ConceptId cid) const {
    std::lock_guard<std::recursive_mutex> lock(subsystem_mtx_);
    return ltm_ ? ltm_->get_outgoing_relations(cid) : std::vector<RelationInfo>{};
}

// ─── Thinking ────────────────────────────────────────────────────────────────

ThinkingResult SystemOrchestrator::run_thinking_cycle(const std::vector<ConceptId>& seeds) {
    if (!running_ || !thinking_) return {};

    std::lock_guard<std::recursive_mutex> lock(subsystem_mtx_);

    auto result = thinking_->execute(
        seeds, active_context_,
        *ltm_, *brain_->get_stm_mutable(), *brain_,
        *cognitive_, *curiosity_,
        *registry_, *embeddings_,
        understanding_.get(),
        kan_validator_.get(),
        gdo_.get(),
        refinement_loop_.get()
    );

    // Feed thinking results into evolution pipeline
    run_evolution_after_thinking(result);

    return result;
}

ThinkingResult SystemOrchestrator::run_thinking_cycle(
    const std::vector<ConceptId>& seeds, GoalState goal)
{
    if (!running_ || !thinking_) return {};

    std::lock_guard<std::recursive_mutex> lock(subsystem_mtx_);

    auto result = thinking_->execute_with_goal(
        seeds, std::move(goal), active_context_,
        *ltm_, *brain_->get_stm_mutable(), *brain_,
        *cognitive_, *curiosity_,
        *registry_, *embeddings_,
        understanding_.get(),
        kan_validator_.get(),
        gdo_.get(),
        refinement_loop_.get()
    );

    run_evolution_after_thinking(result);

    return result;
}

// ─── Evolution ──────────────────────────────────────────────────────────────

void SystemOrchestrator::run_periodic_maintenance() {
    if (!running_ || !epistemic_promotion_ || !pattern_discovery_ || !curiosity_) return;

    std::lock_guard<std::recursive_mutex> lock(subsystem_mtx_);

    log("Running periodic maintenance...");
    
    // Run epistemic promotion maintenance
    auto maintenance_result = epistemic_promotion_->run_maintenance();
    
    if (maintenance_result.promotions > 0 || maintenance_result.demotions > 0) {
        log("  Epistemic maintenance: " + 
            std::to_string(maintenance_result.promotions) + " promotions, " +
            std::to_string(maintenance_result.demotions) + " demotions");
            
        // Update ConceptModels for promoted/demoted concepts
        if (registry_ && (maintenance_result.promotions > 0 || maintenance_result.demotions > 0)) {
            // For now, ensure all concepts have models (could be optimized)
            registry_->ensure_models_for(*ltm_);
        }
    }
    
    if (!maintenance_result.pending_human_review.empty()) {
        log("  " + std::to_string(maintenance_result.pending_human_review.size()) + 
            " concepts pending human review for FACT promotion");
    }
    
    // Run pattern discovery
    auto patterns = pattern_discovery_->discover_all();

    if (!patterns.empty()) {
        size_t gap_count = 0;
        for (const auto& pattern : patterns) {
            if (pattern.pattern_type == "gap") gap_count++;
        }
        log("  Pattern discovery: " + std::to_string(patterns.size()) + " patterns" +
            (gap_count > 0 ? ", " + std::to_string(gap_count) + " gaps" : ""));
    }
}

// ─── Periodic Task Loop ──────────────────────────────────────────────────────

void SystemOrchestrator::periodic_task_loop() {
    using clock = std::chrono::steady_clock;

    auto last_checkpoint = clock::now();
    auto last_maintenance = clock::now();
    const auto checkpoint_interval = std::chrono::minutes(config_.checkpoint_interval_minutes);
    const auto maintenance_interval = std::chrono::minutes(5);

    while (periodic_running_.load(std::memory_order_relaxed)) {
        // Sleep in 1-second intervals so we can respond to shutdown quickly
        std::this_thread::sleep_for(std::chrono::seconds(1));

        if (!periodic_running_.load(std::memory_order_relaxed)) break;

        auto now = clock::now();

        // Auto-checkpoint
        if (config_.enable_persistence && (now - last_checkpoint) >= checkpoint_interval) {
            try {
                create_checkpoint("auto");
                if (wal_) wal_->checkpoint();
            } catch (const std::exception& e) {
                log("  Auto-checkpoint failed: " + std::string(e.what()));
            }
            last_checkpoint = now;
        }

        // Epistemic promotion + pattern discovery
        if ((now - last_maintenance) >= maintenance_interval) {
            try {
                run_periodic_maintenance();
            } catch (const std::exception& e) {
                log("  Periodic maintenance failed: " + std::string(e.what()));
            }
            last_maintenance = now;
        }
    }
}

// ─── Evolution After Thinking ────────────────────────────────────────────────

void SystemOrchestrator::run_evolution_after_thinking(const ThinkingResult& result) {
    // Enqueue generated goals into the goal queue
    if (goal_queue_ && !result.generated_goals.empty()) {
        for (const auto& goal : result.generated_goals) {
            goal_queue_->push(goal);
        }
    }

    if (!concept_proposer_) return;

    // Generate proposals from curiosity triggers
    if (!result.curiosity_triggers.empty()) {
        auto proposals = concept_proposer_->from_curiosity(result.curiosity_triggers);
        auto ranked = concept_proposer_->rank_proposals(proposals, 3);

        for (const auto& proposal : ranked) {
            EpistemicMetadata meta(
                proposal.initial_type,
                EpistemicStatus::ACTIVE,
                proposal.initial_trust
            );
            auto new_id = ltm_->store_concept(
                proposal.label, proposal.description, meta
            );
            wal_log_store_concept(new_id, proposal.label, proposal.description, meta);
            if (registry_) registry_->create_model(new_id);
        }
    }

    // Targeted ConceptModel training for activated concepts
    // Only train models that were active in this thinking cycle, not train_all().
    // This keeps predictions fresh as the KG evolves without O(N) overhead per query.
    if (trainer_ && registry_ && embeddings_ && !result.activated_concepts.empty()) {
        size_t trained = 0;
        for (auto cid : result.activated_concepts) {
            ConceptModel* model = registry_->get_model(cid);
            if (!model) continue;
            auto samples = trainer_->generate_samples(cid, *embeddings_, *ltm_);
            if (samples.empty()) continue;
            trainer_->train_single(cid, *model, *embeddings_, *ltm_);
            ++trained;
        }
        (void)trained;  // Available for future logging
    }

    // Generate proposals from relevance anomalies
    if (result.combined_relevance.size() > 0) {
        auto proposals = concept_proposer_->from_relevance_anomalies(result.combined_relevance);
        auto ranked = concept_proposer_->rank_proposals(proposals, 2);

        for (const auto& proposal : ranked) {
            EpistemicMetadata meta(
                proposal.initial_type,
                EpistemicStatus::ACTIVE,
                proposal.initial_trust
            );
            auto new_id = ltm_->store_concept(
                proposal.label, proposal.description, meta
            );
            wal_log_store_concept(new_id, proposal.label, proposal.description, meta);
            if (registry_) registry_->create_model(new_id);
        }
    }
}

// ─── WAL Helpers ─────────────────────────────────────────────────────────────

void SystemOrchestrator::wal_log_store_concept(
    ConceptId cid, const std::string& label,
    const std::string& definition, const EpistemicMetadata& meta
) {
    if (!wal_) return;

    // Build payload: struct + string data appended
    persistent::WALStoreConceptPayload payload{};
    payload.concept_id = cid;
    payload.label_offset = sizeof(payload);
    payload.label_length = static_cast<uint32_t>(label.size());
    payload.definition_offset = static_cast<uint32_t>(sizeof(payload) + label.size());
    payload.definition_length = static_cast<uint32_t>(definition.size());
    payload.epistemic_type = static_cast<uint8_t>(meta.type);
    payload.epistemic_status = static_cast<uint8_t>(meta.status);
    payload.trust = meta.trust;
    payload.created_epoch_us = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch()
        ).count());

    // Assemble full payload: struct + label bytes + definition bytes
    std::vector<uint8_t> buf(sizeof(payload) + label.size() + definition.size());
    std::memcpy(buf.data(), &payload, sizeof(payload));
    std::memcpy(buf.data() + sizeof(payload), label.data(), label.size());
    std::memcpy(buf.data() + sizeof(payload) + label.size(), definition.data(), definition.size());

    wal_->append(persistent::WALOpType::STORE_CONCEPT, buf.data(), static_cast<uint32_t>(buf.size()));
}

} // namespace brain19
