// brain19_monitor — Standalone CLI for stream monitoring

#include "streams/stream_monitor.hpp"
#include "streams/stream_monitor_cli.hpp"
#include "streams/stream_orchestrator.hpp"
#include "streams/stream_scheduler.hpp"
#include "ltm/long_term_memory.hpp"
#include "memory/stm.hpp"
#include "cmodel/concept_model_registry.hpp"
#include "micromodel/embedding_manager.hpp"
#include "concurrent/shared_ltm.hpp"
#include "concurrent/shared_stm.hpp"
#include "concurrent/shared_registry.hpp"
#include "concurrent/shared_embeddings.hpp"

#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << brain19::StreamMonitorCLI::usage();
        return 1;
    }

    std::string command = argv[1];
    std::vector<std::string> args;
    for (int i = 2; i < argc; ++i) {
        args.emplace_back(argv[i]);
    }

    // Bootstrap minimal runtime
    brain19::LongTermMemory ltm_raw;
    brain19::ShortTermMemory stm_raw;
    brain19::ConceptModelRegistry reg_raw;
    brain19::EmbeddingManager emb_raw;

    brain19::SharedLTM ltm(ltm_raw);
    brain19::SharedSTM stm(stm_raw);
    brain19::SharedRegistry registry(reg_raw);
    brain19::SharedEmbeddings embeddings(emb_raw);

    brain19::StreamConfig cfg;
    cfg.max_streams = 4;

    brain19::StreamOrchestrator orchestrator(ltm, stm, registry, embeddings, cfg);
    brain19::SchedulerConfig sched_cfg;
    sched_cfg.total_max_streams = 4;
    brain19::StreamScheduler scheduler(orchestrator, sched_cfg);

    brain19::AlertThresholds thresholds;
    brain19::StreamMonitor monitor(orchestrator, scheduler, thresholds);

    // Start system briefly to collect metrics
    scheduler.start();
    monitor.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(300));

    brain19::StreamMonitorCLI cli(monitor);
    std::cout << cli.dispatch(command, args);

    monitor.stop();
    scheduler.shutdown();
    return 0;
}
