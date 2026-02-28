#include "core/brain19_app.hpp"
#include <iostream>
#include <string>
#include <cstring>

static void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [options] [command] [args...]\n\n";
    std::cout << "Options:\n";
    std::cout << "  --data-dir <path>      Data directory (default: brain19_data/)\n";
    std::cout << "  --no-persistence       Disable persistence\n";
    std::cout << "  --no-foundation        Skip foundation seeding\n";
    std::cout << "  --max-streams <n>      Max thinking streams (0=auto)\n";
    std::cout << "  --no-monitor           Disable stream monitoring\n";
    std::cout << "  --proactive            Enable proactive Co-Learning on startup\n";
    std::cout << "  --help                 Show this help\n\n";
    std::cout << "Commands:\n";
    std::cout << "  ask <question>         Ask a question\n";
    std::cout << "  ingest <text>          Ingest knowledge\n";
    std::cout << "  load <file_or_dir>     Load JSON training data\n";
    std::cout << "  import <url>           Import from Wikipedia\n";
    std::cout << "  status                 Show system status\n";
    std::cout << "  (none)                 Interactive REPL\n";
}

int main(int argc, char* argv[]) {
    brain19::SystemOrchestrator::Config config;

    // Parse options
    int arg_start = 1;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--data-dir") == 0 && i + 1 < argc) {
            config.data_dir = argv[++i];
            arg_start = i + 1;
        } else if (std::strcmp(argv[i], "--no-persistence") == 0) {
            config.enable_persistence = false;
            arg_start = i + 1;
        } else if (std::strcmp(argv[i], "--no-foundation") == 0) {
            config.seed_foundation = false;
            arg_start = i + 1;
        } else if (std::strcmp(argv[i], "--max-streams") == 0 && i + 1 < argc) {
            try {
                config.max_streams = std::stoul(argv[++i]);
            } catch (const std::exception&) {
                std::cerr << "Invalid value for --max-streams\n";
                return 1;
            }
            arg_start = i + 1;
        } else if (std::strcmp(argv[i], "--no-monitor") == 0) {
            config.enable_monitoring = false;
            arg_start = i + 1;
        } else if (std::strcmp(argv[i], "--proactive") == 0) {
            config.enable_proactive_colearn = true;
            arg_start = i + 1;
        } else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            // Not an option — start of command
            arg_start = i;
            break;
        }
    }

    brain19::Brain19App app(config);

    // If command specified, run it
    if (arg_start < argc) {
        std::string command = argv[arg_start];

        // Collect remaining args as single string
        std::string arg;
        for (int i = arg_start + 1; i < argc; ++i) {
            if (!arg.empty()) arg += " ";
            arg += argv[i];
        }

        // Initialize orchestrator before running command
        if (!app.initialize()) {
            std::cerr << "Failed to initialize Brain19\n";
            return 1;
        }

        int result = app.run_command(command, arg);
        app.shutdown();
        return result;
    }

    // Interactive mode
    return app.run_interactive();
}
