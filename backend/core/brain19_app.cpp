#include "brain19_app.hpp"
#include <iostream>
#include <sstream>
#include <algorithm>

namespace brain19 {

Brain19App::Brain19App()
    : orchestrator_()
{}

Brain19App::Brain19App(SystemOrchestrator::Config config)
    : orchestrator_(std::move(config))
{}

// ─── Interactive REPL ────────────────────────────────────────────────────────

int Brain19App::run_interactive() {
    std::cout << "╔══════════════════════════════════════╗\n";
    std::cout << "║         Brain19 — Think Engine       ║\n";
    std::cout << "╚══════════════════════════════════════╝\n\n";

    if (!orchestrator_.initialize()) {
        std::cerr << "Failed to initialize Brain19\n";
        return 1;
    }

    std::cout << "\nType 'help' for commands, 'quit' to exit.\n\n";

    while (orchestrator_.is_running()) {
        auto line = read_line("brain19> ");
        if (line.empty()) continue;

        auto [cmd, arg] = parse_command(line);

        if (cmd == "quit" || cmd == "exit" || cmd == "q") {
            break;
        }

        if (run_command(cmd, arg) == -1) {
            break;  // quit signal
        }
    }

    orchestrator_.shutdown();
    std::cout << "Goodbye.\n";
    return 0;
}

// ─── Single Command ──────────────────────────────────────────────────────────

int Brain19App::run_command(const std::string& command, const std::string& arg) {
    if (command == "ask")        { cmd_ask(arg); }
    else if (command == "ingest")     { cmd_ingest(arg); }
    else if (command == "import")     { cmd_import(arg); }
    else if (command == "status")     { cmd_status(); }
    else if (command == "streams")    { cmd_streams(); }
    else if (command == "checkpoint") { cmd_checkpoint(arg); }
    else if (command == "restore")    { cmd_restore(arg); }
    else if (command == "concepts")   { cmd_concepts(); }
    else if (command == "explain")    { cmd_explain(arg); }
    else if (command == "think")      { cmd_think(arg); }
    else if (command == "help")       { cmd_help(); }
    else if (command == "quit" || command == "exit" || command == "q") {
        return -1;
    }
    else {
        // Treat as a question if no command matches
        if (!command.empty()) {
            cmd_ask(command + (arg.empty() ? "" : " " + arg));
        }
    }
    return 0;
}

// ─── Command Implementations ─────────────────────────────────────────────────

void Brain19App::cmd_ask(const std::string& question) {
    if (question.empty()) {
        std::cout << "Usage: ask <question>\n";
        return;
    }
    auto resp = orchestrator_.ask(question);
    std::cout << "\n" << resp.answer << "\n";
    if (!resp.epistemic_note.empty()) {
        std::cout << "  [Epistemic: " << resp.epistemic_note << "]\n";
    }
    if (resp.contains_speculation) {
        std::cout << "  ⚠ Contains speculation\n";
    }
    if (!resp.referenced_concepts.empty()) {
        std::cout << "  Referenced " << resp.referenced_concepts.size() << " concepts\n";
    }
    std::cout << "\n";
}

void Brain19App::cmd_ingest(const std::string& text) {
    if (text.empty()) {
        std::cout << "Usage: ingest <text>\n";
        return;
    }
    auto result = orchestrator_.ingest_text(text, true);
    if (result.success) {
        std::cout << "Ingested: " << result.concepts_stored << " concepts, "
                  << result.relations_stored << " relations\n";
    } else {
        std::cout << "Ingestion failed: " << result.error_message << "\n";
    }
}

void Brain19App::cmd_import(const std::string& url) {
    if (url.empty()) {
        std::cout << "Usage: import <url>\n";
        return;
    }
    auto result = orchestrator_.ingest_wikipedia(url);
    if (result.success) {
        std::cout << "Imported: " << result.concepts_stored << " concepts, "
                  << result.relations_stored << " relations\n";
    } else {
        std::cout << "Import failed: " << result.error_message << "\n";
    }
}

void Brain19App::cmd_status() {
    std::cout << orchestrator_.get_status();
}

void Brain19App::cmd_streams() {
    std::cout << orchestrator_.get_stream_status();
}

void Brain19App::cmd_checkpoint(const std::string& tag) {
    orchestrator_.create_checkpoint(tag);
    std::cout << "Checkpoint created" << (tag.empty() ? "" : " [" + tag + "]") << "\n";
}

void Brain19App::cmd_restore(const std::string& dir) {
    if (dir.empty()) {
        std::cout << "Usage: restore <checkpoint_dir>\n";
        return;
    }
    if (orchestrator_.restore_checkpoint(dir)) {
        std::cout << "Restored from " << dir << "\n";
    } else {
        std::cout << "Restore failed\n";
    }
}

void Brain19App::cmd_concepts() {
    auto& ltm = orchestrator_.ltm();
    auto ids = ltm.get_all_concept_ids();
    std::cout << "Total concepts: " << ids.size() << "\n";

    // Show first 20
    size_t shown = 0;
    for (auto cid : ids) {
        if (shown >= 20) {
            std::cout << "  ... and " << (ids.size() - 20) << " more\n";
            break;
        }
        auto info = ltm.retrieve_concept(cid);
        if (info) {
            std::cout << "  [" << info->id << "] " << info->label
                      << " (" << to_string(info->epistemic.type)
                      << ", trust=" << info->epistemic.trust << ")\n";
        }
        ++shown;
    }
}

void Brain19App::cmd_explain(const std::string& id_str) {
    if (id_str.empty()) {
        std::cout << "Usage: explain <concept_id>\n";
        return;
    }

    try {
        ConceptId cid = std::stoull(id_str);
        auto info = orchestrator_.ltm().retrieve_concept(cid);
        if (!info) {
            std::cout << "Concept " << cid << " not found\n";
            return;
        }
        std::cout << "Concept #" << info->id << ": " << info->label << "\n";
        std::cout << "  Definition: " << info->definition << "\n";
        std::cout << "  Type: " << to_string(info->epistemic.type) << "\n";
        std::cout << "  Status: " << to_string(info->epistemic.status) << "\n";
        std::cout << "  Trust: " << info->epistemic.trust << "\n";

        auto rels = orchestrator_.ltm().get_outgoing_relations(cid);
        if (!rels.empty()) {
            std::cout << "  Relations:\n";
            for (auto& r : rels) {
                auto target = orchestrator_.ltm().retrieve_concept(r.target);
                std::cout << "    → " << (target ? target->label : "?")
                          << " (weight=" << r.weight << ")\n";
            }
        }
    } catch (...) {
        std::cout << "Invalid concept ID\n";
    }
}

void Brain19App::cmd_think(const std::string& concept_label) {
    if (concept_label.empty()) {
        std::cout << "Usage: think <concept_label>\n";
        return;
    }

    // Find concept by label
    std::vector<ConceptId> seeds;
    for (auto cid : orchestrator_.ltm().get_all_concept_ids()) {
        auto info = orchestrator_.ltm().retrieve_concept(cid);
        if (info && info->label.find(concept_label) != std::string::npos) {
            seeds.push_back(cid);
            if (seeds.size() >= 3) break;
        }
    }

    if (seeds.empty()) {
        std::cout << "No concepts matching '" << concept_label << "'\n";
        return;
    }

    auto result = orchestrator_.run_thinking_cycle(seeds);
    std::cout << "Thinking complete (" << result.total_duration_ms << " ms)\n";
    std::cout << "  Steps: " << result.steps_completed << "/10\n";
    std::cout << "  Activated: " << result.activated_concepts.size() << " concepts\n";
    std::cout << "  Top salient: " << result.top_salient.size() << "\n";
    std::cout << "  Thought paths: " << result.best_paths.size() << "\n";
    std::cout << "  Curiosity triggers: " << result.curiosity_triggers.size() << "\n";
    std::cout << "  KAN validations: " << result.validated_hypotheses.size() << "\n";
}

void Brain19App::cmd_help() {
    std::cout << "Brain19 Commands:\n";
    std::cout << "  ask <question>     — Ask Brain19 a question\n";
    std::cout << "  ingest <text>      — Ingest knowledge from text\n";
    std::cout << "  import <url>       — Import from Wikipedia\n";
    std::cout << "  think <concept>    — Run thinking cycle on a concept\n";
    std::cout << "  concepts           — List all concepts\n";
    std::cout << "  explain <id>       — Explain a concept\n";
    std::cout << "  status             — System status\n";
    std::cout << "  streams            — Stream monitoring\n";
    std::cout << "  checkpoint [tag]   — Save checkpoint\n";
    std::cout << "  restore <dir>      — Restore from checkpoint\n";
    std::cout << "  help               — Show this help\n";
    std::cout << "  quit               — Shutdown\n";
    std::cout << "\nOr just type a question directly.\n";
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

std::string Brain19App::read_line(const std::string& prompt) {
    std::cout << prompt << std::flush;
    std::string line;
    if (!std::getline(std::cin, line)) {
        return "quit";
    }
    // Trim
    while (!line.empty() && (line.front() == ' ' || line.front() == '\t'))
        line.erase(line.begin());
    while (!line.empty() && (line.back() == ' ' || line.back() == '\t'))
        line.pop_back();
    return line;
}

std::pair<std::string, std::string> Brain19App::parse_command(const std::string& input) {
    auto pos = input.find(' ');
    if (pos == std::string::npos) {
        return {input, ""};
    }
    std::string cmd = input.substr(0, pos);
    std::string arg = input.substr(pos + 1);
    // Trim arg
    while (!arg.empty() && arg.front() == ' ') arg.erase(arg.begin());
    return {cmd, arg};
}

} // namespace brain19
