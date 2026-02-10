// Phase 4.2b: Checkpoint CLI — brain19_checkpoint
//
// Usage:
//   ./brain19_checkpoint save [--dir DIR] [--tag TAG]
//   ./brain19_checkpoint restore <checkpoint-dir> [--components ltm,stm,models]
//   ./brain19_checkpoint list [--dir DIR]
//   ./brain19_checkpoint verify <checkpoint-dir>
//   ./brain19_checkpoint diff <checkpoint-a> <checkpoint-b>

#include "persistent/checkpoint_manager.hpp"
#include "persistent/checkpoint_restore.hpp"

#include <iostream>
#include <string>
#include <cstring>

using namespace brain19;
using namespace brain19::persistent;

namespace {

void usage() {
    std::cerr << R"(brain19_checkpoint — Full Brain State Checkpoint Tool

Usage:
  brain19_checkpoint save    [--dir DIR] [--tag TAG] [--data-dir DIR]
  brain19_checkpoint restore <checkpoint-dir> [--components ltm,stm,models] [--data-dir DIR]
  brain19_checkpoint list    [--dir DIR]
  brain19_checkpoint verify  <checkpoint-dir>
  brain19_checkpoint diff    <checkpoint-a> <checkpoint-b>

Options:
  --dir DIR         Checkpoint base directory (default: checkpoints/)
  --tag TAG         Tag/label for this checkpoint
  --data-dir DIR    LTM data directory (default: data/)
  --components CSV  Comma-separated: ltm,stm,models,kan,cognitive,config,all
)";
}

std::string get_arg(int argc, char** argv, const std::string& flag, const std::string& def = "") {
    for (int i = 1; i < argc - 1; ++i) {
        if (flag == argv[i]) return argv[i + 1];
    }
    return def;
}

bool has_flag(int argc, char** argv, const std::string& flag) {
    for (int i = 1; i < argc; ++i) {
        if (flag == argv[i]) return true;
    }
    return false;
}

int cmd_save(int argc, char** argv) {
    std::string base_dir = get_arg(argc, argv, "--dir", "checkpoints");
    std::string tag = get_arg(argc, argv, "--tag");
    std::string data_dir = get_arg(argc, argv, "--data-dir", "data");
    
    CheckpointManager::Options opts;
    opts.base_dir = base_dir;
    opts.tag = tag;
    opts.max_keep = 5;
    
    CheckpointManager mgr(opts);
    
    // Open LTM
    std::unique_ptr<PersistentLTM> ltm;
    try {
        ltm = std::make_unique<PersistentLTM>(data_dir);
    } catch (const std::exception& e) {
        std::cerr << "Warning: Cannot open LTM at " << data_dir << ": " << e.what() << "\n";
    }
    
    // Save with whatever we have
    try {
        auto path = mgr.save(
            ltm.get(),
            nullptr,    // STM (would need running brain)
            nullptr,    // Models
            nullptr,    // KAN
            nullptr,    // Cognitive
            nullptr     // Config
        );
        std::cout << "Checkpoint saved: " << path << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}

int cmd_restore(int argc, char** argv) {
    if (argc < 3) { usage(); return 1; }
    
    std::string checkpoint_dir = argv[2];
    std::string comp_str = get_arg(argc, argv, "--components", "all");
    std::string data_dir = get_arg(argc, argv, "--data-dir", "data");
    uint8_t components = parse_components(comp_str);
    
    std::unique_ptr<PersistentLTM> ltm;
    try {
        ltm = std::make_unique<PersistentLTM>(data_dir);
    } catch (const std::exception& e) {
        std::cerr << "Warning: Cannot open LTM: " << e.what() << "\n";
    }
    
    STMSnapshotData stm_data;
    MicroModelRegistry models;
    CognitiveState cognitive;
    CheckpointConfig config;
    
    auto result = CheckpointRestore::restore(
        checkpoint_dir, components,
        ltm.get(), &stm_data, &models, nullptr, &cognitive, &config
    );
    
    if (result.success) {
        std::cout << "Restore successful:\n";
        std::cout << "  Concepts:  " << result.concepts_restored << "\n";
        std::cout << "  Relations: " << result.relations_restored << "\n";
        std::cout << "  Models:    " << result.models_restored << "\n";
        return 0;
    } else {
        std::cerr << "Restore failed: " << result.error << "\n";
        return 1;
    }
}

int cmd_list(int argc, char** argv) {
    std::string base_dir = get_arg(argc, argv, "--dir", "checkpoints");
    
    auto entries = CheckpointRestore::list(base_dir);
    if (entries.empty()) {
        std::cout << "No checkpoints found in " << base_dir << "\n";
        return 0;
    }
    
    std::cout << "Checkpoints in " << base_dir << ":\n";
    std::cout << std::string(72, '-') << "\n";
    for (auto& e : entries) {
        std::cout << "  " << e.timestamp;
        if (!e.tag.empty()) std::cout << " [" << e.tag << "]";
        std::cout << "\n    " << e.path;
        std::cout << "\n    Concepts: " << e.concept_count 
                  << "  Relations: " << e.relation_count
                  << "  Models: " << e.model_count << "\n";
    }
    return 0;
}

int cmd_verify(int argc, char** argv) {
    if (argc < 3) { usage(); return 1; }
    std::string checkpoint_dir = argv[2];
    
    auto result = CheckpointRestore::verify(checkpoint_dir);
    std::cout << result.to_string();
    return result.valid ? 0 : 1;
}

int cmd_diff(int argc, char** argv) {
    if (argc < 4) { usage(); return 1; }
    std::string dir_a = argv[2];
    std::string dir_b = argv[3];
    
    auto result = CheckpointRestore::diff(dir_a, dir_b);
    std::cout << result.to_string();
    return 0;
}

} // anon

int main(int argc, char** argv) {
    if (argc < 2) { usage(); return 1; }
    
    std::string cmd = argv[1];
    
    if (cmd == "save")    return cmd_save(argc, argv);
    if (cmd == "restore") return cmd_restore(argc, argv);
    if (cmd == "list")    return cmd_list(argc, argv);
    if (cmd == "verify")  return cmd_verify(argc, argv);
    if (cmd == "diff")    return cmd_diff(argc, argv);
    
    if (cmd == "--help" || cmd == "-h") { usage(); return 0; }
    
    std::cerr << "Unknown command: " << cmd << "\n";
    usage();
    return 1;
}
