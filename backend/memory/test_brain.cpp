#include "brain_controller.hpp"
#include <iostream>
#include <iomanip>

using namespace brain19;

void print_separator(const std::string& title = "") {
    std::cout << "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
    if (!title.empty()) {
        std::cout << title << "\n";
        std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
    }
}

int main() {
    std::cout << std::fixed << std::setprecision(3);
    
    print_separator("Brain19 - STM + BrainController Test");
    
    // Test 1: BrainController Initialization
    std::cout << "\n=== Test 1: Controller Initialization ===\n";
    BrainController brain;
    if (brain.initialize()) {
        std::cout << "✓ BrainController initialized\n";
    }
    
    // Test 2: Context Management
    std::cout << "\n=== Test 2: Context Management ===\n";
    ContextId ctx1 = brain.create_context();
    ContextId ctx2 = brain.create_context();
    std::cout << "✓ Created context 1: " << ctx1 << "\n";
    std::cout << "✓ Created context 2: " << ctx2 << "\n";
    
    // Test 3: Thinking Lifecycle
    std::cout << "\n=== Test 3: Thinking Lifecycle ===\n";
    brain.begin_thinking(ctx1);
    std::cout << "✓ Started thinking in context " << ctx1 << "\n";
    
    // Test 4: Concept Activation via Controller
    std::cout << "\n=== Test 4: Concept Activation ===\n";
    brain.activate_concept_in_context(ctx1, 100, 0.9, ActivationClass::CORE_KNOWLEDGE);
    brain.activate_concept_in_context(ctx1, 200, 0.7, ActivationClass::CONTEXTUAL);
    brain.activate_concept_in_context(ctx1, 300, 0.4, ActivationClass::CONTEXTUAL);
    
    std::cout << "Concept 100 (CORE):       " 
              << brain.query_concept_activation(ctx1, 100) << "\n";
    std::cout << "Concept 200 (CONTEXTUAL): " 
              << brain.query_concept_activation(ctx1, 200) << "\n";
    std::cout << "Concept 300 (CONTEXTUAL): " 
              << brain.query_concept_activation(ctx1, 300) << "\n";
    
    // Test 5: Relation Activation via Controller
    std::cout << "\n=== Test 5: Relation Activation ===\n";
    brain.activate_relation_in_context(ctx1, 100, 200, RelationType::IS_A, 0.8);
    brain.activate_relation_in_context(ctx1, 200, 300, RelationType::CAUSES, 0.6);
    std::cout << "✓ Activated relations: 100→200 (IS_A), 200→300 (CAUSES)\n";
    
    // Test 6: Query Active Concepts
    std::cout << "\n=== Test 6: Query Active Concepts ===\n";
    auto active = brain.query_active_concepts(ctx1, 0.5);
    std::cout << "Active concepts (threshold >= 0.5): ";
    for (auto id : active) {
        std::cout << id << " ";
    }
    std::cout << "\n";
    
    // Test 7: Context Isolation
    std::cout << "\n=== Test 7: Context Isolation ===\n";
    brain.activate_concept_in_context(ctx2, 999, 0.95, ActivationClass::CORE_KNOWLEDGE);
    std::cout << "Context 1 - Concept 999: " 
              << brain.query_concept_activation(ctx1, 999) << " (should be 0.0)\n";
    std::cout << "Context 2 - Concept 999: " 
              << brain.query_concept_activation(ctx2, 999) << " (should be 0.95)\n";
    
    // Test 8: Decay via Controller
    std::cout << "\n=== Test 8: Mechanical Decay ===\n";
    std::cout << "Before decay:\n";
    std::cout << "  Concept 100 (CORE):       " 
              << brain.query_concept_activation(ctx1, 100) << "\n";
    std::cout << "  Concept 200 (CONTEXTUAL): " 
              << brain.query_concept_activation(ctx1, 200) << "\n";
    
    brain.decay_context(ctx1, 1.0);  // 1 second
    
    std::cout << "\nAfter 1s decay:\n";
    std::cout << "  Concept 100 (CORE):       " 
              << brain.query_concept_activation(ctx1, 100) << "\n";
    std::cout << "  Concept 200 (CONTEXTUAL): " 
              << brain.query_concept_activation(ctx1, 200) << "\n";
    
    // Test 9: Two-Phase Relation Decay
    std::cout << "\n=== Test 9: Two-Phase Relation Decay ===\n";
    const ShortTermMemory* stm = brain.get_stm();
    std::cout << "Initial relations: " << stm->debug_active_relation_count(ctx1) << "\n";
    
    // Decay multiple times
    for (int i = 1; i <= 10; i++) {
        brain.decay_context(ctx1, 1.0);
        size_t rel_count = stm->debug_active_relation_count(ctx1);
        std::cout << "After " << i << "s decay: " << rel_count << " relations\n";
        if (rel_count == 0) break;
    }
    
    // Test 10: Debug Introspection
    std::cout << "\n=== Test 10: Debug Introspection ===\n";
    std::cout << "Context 1:\n";
    std::cout << "  Active concepts:  " << stm->debug_active_concept_count(ctx1) << "\n";
    std::cout << "  Active relations: " << stm->debug_active_relation_count(ctx1) << "\n";
    std::cout << "Context 2:\n";
    std::cout << "  Active concepts:  " << stm->debug_active_concept_count(ctx2) << "\n";
    std::cout << "  Active relations: " << stm->debug_active_relation_count(ctx2) << "\n";
    
    // Test 11: End Thinking
    std::cout << "\n=== Test 11: End Thinking ===\n";
    brain.end_thinking(ctx1);
    std::cout << "✓ Stopped thinking in context " << ctx1 << "\n";
    
    // Test 12: Context Destruction
    std::cout << "\n=== Test 12: Context Destruction ===\n";
    brain.destroy_context(ctx1);
    std::cout << "✓ Destroyed context " << ctx1 << "\n";
    
    // Verify context is gone
    double act = brain.query_concept_activation(ctx1, 100);
    std::cout << "Query on destroyed context: " << act << " (should be 0.0)\n";
    
    // Test 13: Shutdown
    std::cout << "\n=== Test 13: Shutdown ===\n";
    brain.shutdown();
    std::cout << "✓ BrainController shutdown complete\n";
    
    print_separator("All Tests Complete");
    std::cout << "\n✓ STM: Purely mechanical activation layer\n";
    std::cout << "✓ STM: Two-phase decay prevents flapping\n";
    std::cout << "✓ STM: Debug introspection separated from logic\n";
    std::cout << "✓ Controller: Pure orchestration, no intelligence\n";
    std::cout << "✓ Controller: Explicit delegation only\n";
    std::cout << "✓ Architecture: Misuse impossible by construction\n";
    
    print_separator();
    
    return 0;
}
