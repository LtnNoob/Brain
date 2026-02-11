#include "system_orchestrator.hpp"
#include "../bootstrap/foundation_concepts.hpp"

#include <chrono>
#include <filesystem>
#include <iostream>
#include <sstream>

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
        log("  [1/14] LTM...");
        ltm_ = std::make_unique<LongTermMemory>();
        init_stage_ = 1;

        // ── Stage 2: WAL + Checkpoint Restore ───────────────────────────
        if (config_.enable_persistence) {
            log("  [2/14] Persistence...");
            std::filesystem::create_directories(config_.data_dir);
            // Note: PersistentLTM is an alternative backend.
            // For now we use in-memory LTM with checkpoint save/restore.
        }
        init_stage_ = 2;

        // ── Stage 3: BrainController + STM ──────────────────────────────
        log("  [3/14] BrainController...");
        brain_ = std::make_unique<BrainController>();
        brain_->initialize();
        init_stage_ = 3;

        // ── Stage 4: MicroModels ────────────────────────────────────────
        log("  [4/14] MicroModels...");
        embeddings_ = std::make_unique<EmbeddingManager>();
        registry_ = std::make_unique<MicroModelRegistry>();
        trainer_ = std::make_unique<MicroTrainer>();
        init_stage_ = 4;

        // ── Stage 5: CognitiveDynamics ──────────────────────────────────
        log("  [5/14] CognitiveDynamics...");
        cognitive_ = std::make_unique<CognitiveDynamics>();
        init_stage_ = 5;

        // ── Stage 6: CuriosityEngine ────────────────────────────────────
        log("  [6/14] CuriosityEngine...");
        curiosity_ = std::make_unique<CuriosityEngine>();
        init_stage_ = 6;

        // ── Stage 7: KANAdapter ─────────────────────────────────────────
        log("  [7/14] KANAdapter...");
        kan_adapter_ = std::make_unique<KANAdapter>();
        init_stage_ = 7;

        // ── Stage 8: UnderstandingLayer + MiniLLMs ──────────────────────
        log("  [8/14] UnderstandingLayer...");
        understanding_ = std::make_unique<UnderstandingLayer>();
        // Register a stub MiniLLM (always available)
        understanding_->register_mini_llm(std::make_unique<StubMiniLLM>());
        // Try to register Ollama-backed MiniLLM
        {
            OllamaConfig ollama_cfg;
            ollama_cfg.host = config_.ollama_host;
            ollama_cfg.model = config_.ollama_model;
            auto ollama_llm = std::make_unique<OllamaMiniLLM>(ollama_cfg);
            if (ollama_llm->is_available()) {
                log("    Ollama MiniLLM available");
                understanding_->register_mini_llm(std::move(ollama_llm));
            } else {
                log("    Ollama MiniLLM not available (degraded mode)");
            }
        }
        init_stage_ = 8;

        // ── Stage 9: KAN-LLM Hybrid ────────────────────────────────────
        log("  [9/14] KAN-LLM Hybrid...");
        kan_validator_ = std::make_unique<KanValidator>();
        domain_manager_ = std::make_unique<DomainManager>();
        refinement_loop_ = std::make_unique<RefinementLoop>(*kan_validator_);
        init_stage_ = 9;

        // ── Stage 10: IngestionPipeline ─────────────────────────────────
        log("  [10/14] IngestionPipeline...");
        ingestion_ = std::make_unique<IngestionPipeline>(*ltm_);
        wiki_importer_ = std::make_unique<WikipediaImporter>();
        init_stage_ = 10;

        // ── Stage 11: ChatInterface + OllamaClient ──────────────────────
        log("  [11/14] ChatInterface...");
        chat_ = std::make_unique<ChatInterface>();
        {
            OllamaConfig chat_cfg;
            chat_cfg.host = config_.ollama_host;
            chat_cfg.model = config_.ollama_model;
            if (chat_->initialize(chat_cfg)) {
                log("    Chat LLM available");
            } else {
                log("    Chat LLM not available (knowledge-only mode)");
            }
        }
        init_stage_ = 11;

        // ── Stage 12: Shared Wrappers ───────────────────────────────────
        log("  [12/14] Shared wrappers...");
        shared_ltm_ = std::make_unique<SharedLTM>(*ltm_);
        shared_stm_ = std::make_unique<SharedSTM>(*brain_->get_stm_mutable());
        shared_registry_ = std::make_unique<SharedRegistry>(*registry_);
        shared_embeddings_ = std::make_unique<SharedEmbeddings>(*embeddings_);
        init_stage_ = 12;

        // ── Stage 13: Streams ───────────────────────────────────────────
        log("  [13/14] Streams...");
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

        // ── Stage 14: Bootstrap Foundation ──────────────────────────────
        if (config_.seed_foundation && ltm_->get_all_concept_ids().empty()) {
            log("  [14/14] Seeding foundation concepts...");
            seed_foundation();
        } else {
            log("  [14/14] Foundation already present or disabled");
        }
        init_stage_ = 14;

        // ── ThinkingPipeline ────────────────────────────────────────────
        thinking_ = std::make_unique<ThinkingPipeline>();

        // Create active context
        active_context_ = brain_->create_context();

        // Ensure MicroModels for all concepts
        registry_->ensure_models_for(*ltm_);

        // Start streams
        stream_sched_->start();
        if (stream_monitor_) {
            stream_monitor_->start();
        }

        running_ = true;
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

    running_ = false;

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
    curiosity_.reset();
    cognitive_.reset();
    trainer_.reset();
    registry_.reset();
    embeddings_.reset();
    brain_.reset();
    persistent_ltm_.reset();
    ltm_.reset();
    thinking_.reset();

    init_stage_ = 0;
    log("Brain19 shut down.");
}

// ─── Cleanup on partial init failure ─────────────────────────────────────────

void SystemOrchestrator::cleanup_from_stage(int stage) {
    // Clean up in reverse from the stage that succeeded
    if (stage >= 13) { stream_monitor_.reset(); stream_sched_.reset(); stream_orch_.reset(); }
    if (stage >= 12) { shared_embeddings_.reset(); shared_registry_.reset(); shared_stm_.reset(); shared_ltm_.reset(); }
    if (stage >= 11) { chat_.reset(); }
    if (stage >= 10) { ingestion_.reset(); wiki_importer_.reset(); }
    if (stage >= 9)  { refinement_loop_.reset(); domain_manager_.reset(); kan_validator_.reset(); }
    if (stage >= 8)  { understanding_.reset(); }
    if (stage >= 7)  { kan_adapter_.reset(); }
    if (stage >= 6)  { curiosity_.reset(); }
    if (stage >= 5)  { cognitive_.reset(); }
    if (stage >= 4)  { trainer_.reset(); registry_.reset(); embeddings_.reset(); }
    if (stage >= 3)  { if (brain_) brain_->shutdown(); brain_.reset(); }
    if (stage >= 1)  { persistent_ltm_.reset(); ltm_.reset(); }
    init_stage_ = 0;
}

// ─── Foundation Seeding ──────────────────────────────────────────────────────

void SystemOrchestrator::seed_foundation() {
    FoundationConcepts::seed_all(*ltm_);
    log("    Seeded " + std::to_string(FoundationConcepts::concept_count()) +
        " concepts, " + std::to_string(FoundationConcepts::relation_count()) + " relations");
}

// ─── Chat ────────────────────────────────────────────────────────────────────

ChatResponse SystemOrchestrator::ask(const std::string& question) {
    if (!running_ || !chat_) {
        ChatResponse resp;
        resp.answer = "[Brain19 not running]";
        resp.used_llm = false;
        return resp;
    }

    // Run a thinking cycle first to activate relevant concepts
    // Find seed concepts by searching LTM labels
    std::vector<ConceptId> seeds;
    for (auto cid : ltm_->get_all_concept_ids()) {
        auto info = ltm_->retrieve_concept(cid);
        if (info && question.find(info->label) != std::string::npos) {
            seeds.push_back(cid);
            if (seeds.size() >= 5) break;
        }
    }

    // Run thinking cycle and pass results to ChatInterface
    if (!seeds.empty()) {
        auto thinking_result = run_thinking_cycle(seeds);

        // Collect salient concept IDs
        std::vector<ConceptId> salient_ids;
        for (const auto& s : thinking_result.top_salient) {
            salient_ids.push_back(s.concept_id);
        }

        // Build thought path summaries
        std::vector<std::string> path_summaries;
        for (const auto& path : thinking_result.best_paths) {
            std::string summary;
            for (size_t i = 0; i < path.nodes.size(); ++i) {
                if (i > 0) summary += " → ";
                auto info = ltm_->retrieve_concept(path.nodes[i].concept_id);
                summary += info ? info->label : ("?" + std::to_string(path.nodes[i].concept_id));
            }
            path_summaries.push_back(summary);
        }

        return chat_->ask_with_context(question, *ltm_, salient_ids, path_summaries);
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

    auto result = ingestion_->ingest_text(text, "", auto_approve);
    if (result.success && !result.stored_concept_ids.empty()) {
        // Ensure micromodels only for newly stored concepts
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

    auto proposal = wiki_importer_->import_from_url(url);
    if (!proposal) {
        IngestionResult r;
        r.success = false;
        r.error_message = "Failed to import from URL";
        return r;
    }

    // Ingest the extracted text
    auto result = ingestion_->ingest_text(proposal->extracted_text, url, true);
    if (result.success && !result.stored_concept_ids.empty()) {
        registry_->ensure_models_for(result.stored_concept_ids);
    }
    return result;
}

// ─── Checkpoint ──────────────────────────────────────────────────────────────

void SystemOrchestrator::create_checkpoint(const std::string& tag) {
    if (!running_) return;

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
    return ltm_ ? ltm_->get_all_concept_ids().size() : 0;
}

size_t SystemOrchestrator::relation_count() const {
    if (!ltm_) return 0;
    return ltm_->total_relation_count();
}

// ─── Thinking ────────────────────────────────────────────────────────────────

ThinkingResult SystemOrchestrator::run_thinking_cycle(const std::vector<ConceptId>& seeds) {
    if (!running_ || !thinking_) return {};

    return thinking_->execute(
        seeds, active_context_,
        *ltm_, *brain_->get_stm_mutable(), *brain_,
        *cognitive_, *curiosity_,
        *registry_, *embeddings_,
        understanding_.get(),
        kan_validator_.get()
    );
}

} // namespace brain19
