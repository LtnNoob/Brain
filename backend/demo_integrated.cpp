#include "memory/brain_controller.hpp"
#include "adapter/kan_adapter.hpp"
#include "kan/kan_module.hpp"
#include "kan/function_hypothesis.hpp"
#include "curiosity/curiosity_engine.hpp"
#include "snapshot_generator.hpp"
#include <iostream>
#include <fstream>
#include <thread>
#include <chrono>

using namespace brain19;

void print_separator(const std::string& title = "") {
    std::cout << "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
    if (!title.empty()) {
        std::cout << title << "\n";
        std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
    }
}

void save_snapshot_to_file(const std::string& json, const std::string& filename) {
    std::ofstream file(filename);
    if (file.is_open()) {
        file << json;
        file.close();
        std::cout << "✓ Snapshot saved to: " << filename << "\n";
    } else {
        std::cerr << "✗ Failed to save snapshot\n";
    }
}

int main() {
    print_separator("Brain19 - Complete Integration Demo");
    
    std::cout << "\nInitializing Brain19 subsystems...\n";
    
    // Initialize all subsystems
    BrainController brain;
    KANAdapter kan_adapter;
    CuriosityEngine curiosity;
    SnapshotGenerator snapshot_gen;
    
    if (!brain.initialize()) {
        std::cerr << "Failed to initialize BrainController\n";
        return 1;
    }
    std::cout << "✓ BrainController initialized\n";
    
    // Create context
    ContextId ctx = brain.create_context();
    std::cout << "✓ Context created: " << ctx << "\n";
    
    print_separator("Phase 1: Activate Concepts in STM");
    
    brain.begin_thinking(ctx);
    
    // Activate some concepts
    brain.activate_concept_in_context(ctx, 100, 0.95, ActivationClass::CORE_KNOWLEDGE);
    brain.activate_concept_in_context(ctx, 200, 0.85, ActivationClass::CORE_KNOWLEDGE);
    brain.activate_concept_in_context(ctx, 300, 0.72, ActivationClass::CONTEXTUAL);
    brain.activate_concept_in_context(ctx, 400, 0.58, ActivationClass::CONTEXTUAL);
    brain.activate_concept_in_context(ctx, 500, 0.45, ActivationClass::CONTEXTUAL);
    
    std::cout << "✓ Activated 5 concepts\n";
    
    // Activate some relations
    brain.activate_relation_in_context(ctx, 100, 200, RelationType::IS_A, 0.88);
    brain.activate_relation_in_context(ctx, 200, 300, RelationType::HAS_PROPERTY, 0.75);
    brain.activate_relation_in_context(ctx, 300, 400, RelationType::CAUSES, 0.62);
    
    std::cout << "✓ Activated 3 relations\n";
    
    print_separator("Phase 2: Curiosity Engine Observes");
    
    // Create observation
    SystemObservation obs;
    obs.context_id = ctx;
    obs.active_concept_count = 5;
    obs.active_relation_count = 3;
    
    auto triggers = curiosity.observe_and_generate_triggers({obs});
    
    std::cout << "✓ Curiosity generated " << triggers.size() << " trigger(s)\n";
    for (const auto& trigger : triggers) {
        std::cout << "  - " << trigger.description << "\n";
    }
    
    print_separator("Phase 3: Generate Visualization Snapshot");
    
    std::string json_snapshot = snapshot_gen.generate_json_snapshot(&brain, nullptr, &curiosity, ctx);
    
    std::cout << "\nJSON Snapshot:\n";
    std::cout << "─────────────────────────────────────────\n";
    std::cout << json_snapshot;
    std::cout << "─────────────────────────────────────────\n";
    
    // Save to file for frontend
    save_snapshot_to_file(json_snapshot, "snapshot.json");
    
    print_separator("Phase 4: KAN Adapter Demo (Optional)");
    
    if (triggers.empty()) {
        std::cout << "No triggers - skipping KAN training\n";
    } else {
        std::cout << "Trigger detected: Creating KAN module...\n";
        
        uint64_t kan_id = kan_adapter.create_kan_module(1, 1, 8);
        std::cout << "✓ KAN module created: " << kan_id << "\n";
        
        // Generate simple training data
        std::vector<DataPoint> data;
        for (double x = 0.0; x <= 1.0; x += 0.2) {
            data.push_back(DataPoint({x}, {x * 2.0}));
        }
        
        KanTrainingConfig config;
        config.max_iterations = 100;
        config.learning_rate = 0.05;
        
        std::cout << "Training KAN module...\n";
        auto hypothesis = kan_adapter.train_kan_module(kan_id, data, config);
        
        if (hypothesis) {
            std::cout << "✓ Training complete\n";
            std::cout << "  Iterations: " << hypothesis->training_iterations << "\n";
            std::cout << "  Error: " << hypothesis->training_error << "\n";
        }
        
        kan_adapter.destroy_kan_module(kan_id);
        std::cout << "✓ KAN module destroyed\n";
    }
    
    print_separator("Phase 5: System Status");
    
    auto active = brain.query_active_concepts(ctx, 0.0);
    std::cout << "Active concepts: " << active.size() << "\n";
    for (auto id : active) {
        double act = brain.query_concept_activation(ctx, id);
        std::cout << "  Concept " << id << ": " << (act * 100.0) << "%\n";
    }
    
    print_separator("Phase 6: Cleanup");
    
    brain.end_thinking(ctx);
    brain.destroy_context(ctx);
    brain.shutdown();
    
    std::cout << "✓ All subsystems shut down\n";
    
    print_separator("Integration Demo Complete");
    
    std::cout << "\n✓ Backend: All subsystems working\n";
    std::cout << "✓ Snapshot: Generated and saved\n";
    std::cout << "✓ Frontend: Ready to load snapshot.json\n";
    std::cout << "\nTo visualize:\n";
    std::cout << "  1. cd ../frontend\n";
    std::cout << "  2. npm install\n";
    std::cout << "  3. Copy ../backend/snapshot.json to frontend/public/\n";
    std::cout << "  4. npm run dev\n";
    
    print_separator();
    
    return 0;
}
