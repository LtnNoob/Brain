#include "system_orchestrator.hpp"
#include "../bootstrap/foundation_concepts.hpp"

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

        // ── Stage 4: MicroModels ────────────────────────────────────────
        log("  [4/15] MicroModels...");
        embeddings_ = std::make_unique<EmbeddingManager>();
        registry_ = std::make_unique<MicroModelRegistry>();
        trainer_ = std::make_unique<MicroTrainer>();
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
        // Register a stub MiniLLM (always available)
        understanding_->register_mini_llm(std::make_unique<StubMiniLLM>());
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

        // Ensure MicroModels for all concepts
        registry_->ensure_models_for(*ltm_);

        // Start GDO
        if (gdo_) {
            gdo_->set_thinking_callback([this](const std::vector<ConceptId>& seeds) {
                std::lock_guard<std::recursive_mutex> lock(subsystem_mtx_);
                if (running_ && thinking_) {
                    auto result = thinking_->execute(
                        seeds, active_context_,
                        *ltm_, *brain_->get_stm_mutable(), *brain_,
                        *cognitive_, *curiosity_,
                        *registry_, *embeddings_,
                        understanding_.get(),
                        kan_validator_.get(),
                        gdo_.get()
                    );
                    run_evolution_after_thinking(result);
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
    stream_monitor_.reset();
    stream_sched_.reset();
    stream_orch_.reset();
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
    if (stage >= 11) { chat_.reset(); }
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
    if (!config_.foundation_file.empty()) {
        if (FoundationConcepts::seed_from_file(*ltm_, config_.foundation_file)) {
            log("    Seeded from file: " + config_.foundation_file);
            log("    Concepts: " + std::to_string(ltm_->get_all_concept_ids().size()) +
                ", Relations: " + std::to_string(ltm_->total_relation_count()));
            return;
        }
        log("    File not found or invalid, falling back to hardcoded");
    }
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

    // Sort by score, take top seeds
    std::sort(scored_seeds.begin(), scored_seeds.end(),
        [](const ScoredSeed& a, const ScoredSeed& b) { return a.score > b.score; });

    std::vector<ConceptId> seeds;
    size_t max_seeds = 8;
    for (size_t i = 0; i < std::min(scored_seeds.size(), max_seeds); ++i) {
        seeds.push_back(scored_seeds[i].id);
    }

    // Inject energy into GDO from user query
    if (gdo_ && !seeds.empty()) {
        gdo_->inject_energy(config_.gdo_config.injection_boost);
        gdo_->inject_seeds(seeds, 0.8);
    }

    // Run thinking cycle and pass results to ChatInterface
    if (!seeds.empty()) {
        auto thinking_result = run_thinking_cycle(seeds);

        // Collect salient concept IDs
        std::vector<ConceptId> salient_ids;
        for (const auto& s : thinking_result.top_salient) {
            salient_ids.push_back(s.concept_id);
        }

        // Build thought path summaries (top 3)
        std::vector<std::string> path_summaries;
        for (const auto& path : thinking_result.best_paths) {
            if (path_summaries.size() >= 3) break;
            std::string summary;
            for (size_t i = 0; i < path.nodes.size(); ++i) {
                if (i > 0) summary += " -> ";
                auto info = ltm_->retrieve_concept(path.nodes[i].concept_id);
                summary += info ? info->label : ("?" + std::to_string(path.nodes[i].concept_id));
            }
            path_summaries.push_back(summary);
        }

        return chat_->ask_with_context(question, *ltm_, salient_ids, path_summaries, intent);
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
    ss << "MicroModels: " << (registry_ ? registry_->size() : 0) << "\n";
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
        gdo_.get()
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
        gdo_.get()
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
            
        // Update MicroModels for promoted/demoted concepts
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
