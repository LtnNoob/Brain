#pragma once

#include "../common/types.hpp"
#include "../ltm/long_term_memory.hpp"
#include "../memory/brain_controller.hpp"
#include "../memory/stm.hpp"
#include "../epistemic/epistemic_metadata.hpp"
#include "../cognitive/cognitive_dynamics.hpp"
#include "../curiosity/curiosity_engine.hpp"
#include "../micromodel/micro_model_registry.hpp"
#include "../micromodel/embedding_manager.hpp"
#include "../micromodel/micro_trainer.hpp"
#include "../micromodel/relevance_map.hpp"
#include "../adapter/kan_adapter.hpp"
#include "../understanding/understanding_layer.hpp"
#include "../understanding/mini_llm.hpp"
#include "../llm/chat_interface.hpp"
#include "../ingestor/ingestion_pipeline.hpp"
#include "../importers/wikipedia_importer.hpp"
#include "../importers/scholar_importer.hpp"
#include "../persistent/persistent_ltm.hpp"
#include "../persistent/wal.hpp"
#include "../persistent/stm_snapshot.hpp"
#include "../persistent/checkpoint_manager.hpp"
#include "../persistent/checkpoint_restore.hpp"
#include "../concurrent/shared_ltm.hpp"
#include "../concurrent/shared_stm.hpp"
#include "../concurrent/shared_registry.hpp"
#include "../concurrent/shared_embeddings.hpp"
#include "../streams/stream_orchestrator.hpp"
#include "../streams/stream_scheduler.hpp"
#include "../streams/stream_monitor.hpp"
#include "../hybrid/kan_validator.hpp"
#include "../hybrid/domain_manager.hpp"
#include "../hybrid/refinement_loop.hpp"
#include "../evolution/pattern_discovery.hpp"
#include "../evolution/epistemic_promotion.hpp"
#include "../evolution/concept_proposal.hpp"

#include "thinking_pipeline.hpp"

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace brain19 {

class SystemOrchestrator {
public:
    struct Config {
        // Persistence
        std::string data_dir = "brain19_data/";
        bool enable_persistence = true;
        bool enable_wal = true;
        size_t checkpoint_interval_minutes = 30;
        size_t max_checkpoints = 5;

        // Streams
        size_t max_streams = 0;  // 0 = auto-detect
        bool enable_monitoring = true;

        // Bootstrap
        bool seed_foundation = true;
    };

    SystemOrchestrator();
    explicit SystemOrchestrator(Config config);
    ~SystemOrchestrator();

    // Non-copyable, non-movable
    SystemOrchestrator(const SystemOrchestrator&) = delete;
    SystemOrchestrator& operator=(const SystemOrchestrator&) = delete;

    // ─── Lifecycle ───────────────────────────────────────────────────────────

    bool initialize();
    void shutdown();
    bool is_running() const { return running_.load(std::memory_order_acquire); }

    // ─── Chat Interface ──────────────────────────────────────────────────────

    ChatResponse ask(const std::string& question);

    // ─── Ingestion ───────────────────────────────────────────────────────────

    IngestionResult ingest_text(const std::string& text, bool auto_approve = false);
    IngestionResult ingest_wikipedia(const std::string& url);

    // ─── Knowledge Management ────────────────────────────────────────────────

    void create_checkpoint(const std::string& tag = "");
    bool restore_checkpoint(const std::string& checkpoint_dir);

    // ─── Monitoring ──────────────────────────────────────────────────────────

    std::string get_status() const;
    std::string get_stream_status() const;

    // ─── Access (const, locked) ────────────────────────────────────────────

    size_t concept_count() const;
    size_t relation_count() const;

    // Locked LTM access for REPL commands
    std::vector<ConceptId> get_all_concept_ids() const;
    std::optional<ConceptInfo> get_concept(ConceptId cid) const;
    std::vector<RelationInfo> get_outgoing_relations(ConceptId cid) const;

    // ─── Thinking ────────────────────────────────────────────────────────────

    ThinkingResult run_thinking_cycle(const std::vector<ConceptId>& seeds);

    // ─── Evolution ──────────────────────────────────────────────────────────

    void run_periodic_maintenance();

    // ─── Subsystem Access (for advanced use) ─────────────────────────────────

    BrainController* brain_controller() { return brain_.get(); }
    CognitiveDynamics* cognitive_dynamics() { return cognitive_.get(); }
    ChatInterface* chat_interface() { return chat_.get(); }
    IngestionPipeline* ingestion_pipeline() { return ingestion_.get(); }

private:
    Config config_;
    std::atomic<bool> running_{false};
    int init_stage_ = 0;  // tracks how far initialization got (for cleanup)

    // Orchestrator-level mutex: protects all subsystem access (LTM, STM, registry, etc.)
    // from concurrent use between ask()/ingest() on main thread and periodic_task_loop().
    // Streams use SharedLTM/SharedSTM/SharedRegistry/SharedEmbeddings independently.
    mutable std::recursive_mutex subsystem_mtx_;

    // ─── Subsystems (owned, initialization order) ────────────────────────────

    // 1. LTM
    std::unique_ptr<LongTermMemory> ltm_;
    std::unique_ptr<persistent::PersistentLTM> persistent_ltm_;

    // 2. WAL
    std::unique_ptr<persistent::WALWriter> wal_;

    // 3. BrainController + STM
    std::unique_ptr<BrainController> brain_;

    // 4. MicroModels
    std::unique_ptr<EmbeddingManager> embeddings_;
    std::unique_ptr<MicroModelRegistry> registry_;
    std::unique_ptr<MicroTrainer> trainer_;

    // 5. Cognitive
    std::unique_ptr<CognitiveDynamics> cognitive_;

    // 6. Curiosity
    std::unique_ptr<CuriosityEngine> curiosity_;

    // 7. KAN
    std::unique_ptr<KANAdapter> kan_adapter_;

    // 8. Understanding
    std::unique_ptr<UnderstandingLayer> understanding_;

    // 9. Hybrid
    std::unique_ptr<KanValidator> kan_validator_;
    std::unique_ptr<DomainManager> domain_manager_;
    std::unique_ptr<RefinementLoop> refinement_loop_;

    // 10. Ingestion
    std::unique_ptr<IngestionPipeline> ingestion_;
    std::unique_ptr<WikipediaImporter> wiki_importer_;

    // 11. Chat + LLM
    std::unique_ptr<ChatInterface> chat_;

    // 12. Shared Wrappers
    std::unique_ptr<SharedLTM> shared_ltm_;
    std::unique_ptr<SharedSTM> shared_stm_;
    std::unique_ptr<SharedRegistry> shared_registry_;
    std::unique_ptr<SharedEmbeddings> shared_embeddings_;

    // 13. Streams
    std::unique_ptr<StreamOrchestrator> stream_orch_;
    std::unique_ptr<StreamScheduler> stream_sched_;
    std::unique_ptr<StreamMonitor> stream_monitor_;

    // 14. Evolution
    std::unique_ptr<PatternDiscovery> pattern_discovery_;
    std::unique_ptr<EpistemicPromotion> epistemic_promotion_;
    std::unique_ptr<ConceptProposer> concept_proposer_;

    // Thinking pipeline
    std::unique_ptr<ThinkingPipeline> thinking_;

    // Active context for interactive use
    ContextId active_context_ = 0;

    // Periodic task thread
    std::thread periodic_thread_;
    std::atomic<bool> periodic_running_{false};

    // Stream alert log (thread-safe)
    mutable std::mutex alert_log_mtx_;
    std::vector<std::string> stream_alerts_;

    // ─── Helpers ─────────────────────────────────────────────────────────────

    void cleanup_from_stage(int stage);
    void seed_foundation();
    void log(const std::string& msg) const;

    // Evolution: process thinking result for concept proposals
    void run_evolution_after_thinking(const ThinkingResult& result);

    // Periodic task loop
    void periodic_task_loop();

    // WAL helper: log a store_concept operation
    void wal_log_store_concept(ConceptId cid, const std::string& label,
                               const std::string& definition,
                               const EpistemicMetadata& meta);
};

} // namespace brain19
