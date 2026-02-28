#pragma once

#include "system_orchestrator.hpp"
#include <string>

namespace brain19 {

class Brain19App {
public:
    Brain19App();
    explicit Brain19App(SystemOrchestrator::Config config);
    ~Brain19App() = default;

    // Initialize orchestrator (for non-interactive use)
    bool initialize() { return orchestrator_.initialize(); }

    // Shutdown orchestrator
    void shutdown() { orchestrator_.shutdown(); }

    // Run interactive REPL
    int run_interactive();

    // Run single command
    int run_command(const std::string& command, const std::string& arg = "");

private:
    SystemOrchestrator orchestrator_;

    // Command handlers
    void cmd_ask(const std::string& question);
    void cmd_ingest(const std::string& text);
    void cmd_import(const std::string& url);
    void cmd_status();
    void cmd_streams();
    void cmd_checkpoint(const std::string& tag);
    void cmd_restore(const std::string& dir);
    void cmd_concepts();
    void cmd_explain(const std::string& id_str);
    void cmd_think(const std::string& concept_label);
    void cmd_colearn(const std::string& arg);
    void cmd_load(const std::string& path);
    void cmd_help();

    // REPL helpers
    std::string read_line(const std::string& prompt);
    std::pair<std::string, std::string> parse_command(const std::string& input);
};

} // namespace brain19
