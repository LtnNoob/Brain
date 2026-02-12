#include "memory/brain_controller.hpp"
#include "ltm/long_term_memory.hpp"
#include "epistemic/epistemic_metadata.hpp"
#include "adapter/kan_adapter.hpp"
#include "kan/kan_module.hpp"
#include "curiosity/curiosity_engine.hpp"
#include "snapshot_generator.hpp"
#include "importers/wikipedia_importer.hpp"
#include <iostream>
#include <fstream>

using namespace brain19;

void print_separator(const std::string& title = "") {
    std::cout << "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
    if (!title.empty()) {
        std::cout << title << "\n";
        std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
    }
}

int main() {
    print_separator("Brain19 - Complete System Demo with Epistemic Enforcement");
    
    std::cout << "\nInitializing Brain19 subsystems...\n";
    
    // Initialize subsystems
    BrainController brain;
    LongTermMemory ltm;
    KANAdapter kan_adapter;
    CuriosityEngine curiosity;
    SnapshotGenerator snapshot_gen;
    
    if (!brain.initialize()) {
        std::cerr << "Failed to initialize BrainController\n";
        return 1;
    }
    std::cout << "✓ BrainController initialized\n";
    std::cout << "✓ LongTermMemory initialized\n";
    
    print_separator("Phase 1: Store Knowledge with Epistemic Metadata");
    
    // Store some concepts in LTM with EXPLICIT epistemic metadata
    std::cout << "\nStoring concepts (epistemic metadata REQUIRED):\n\n";
    
    // Concept 1: A verified fact
    std::cout << "1. Storing FACT (high certainty):\n";
    EpistemicMetadata fact_meta(
        EpistemicType::FACT,
        EpistemicStatus::ACTIVE,
        0.98
    );
    ConceptId cat_id = ltm.store_concept(
        "Cat",
        "A small carnivorous mammal (Felis catus)",
        fact_meta
    );
    std::cout << "   ✓ Concept ID: " << cat_id << "\n";
    std::cout << "   - Type: FACT\n";
    std::cout << "   - Trust: 0.98\n\n";
    
    // Concept 2: A well-supported theory
    std::cout << "2. Storing THEORY (well-supported):\n";
    EpistemicMetadata theory_meta(
        EpistemicType::THEORY,
        EpistemicStatus::ACTIVE,
        0.85
    );
    ConceptId evolution_id = ltm.store_concept(
        "Evolution",
        "Scientific theory of biological change over time",
        theory_meta
    );
    std::cout << "   ✓ Concept ID: " << evolution_id << "\n";
    std::cout << "   - Type: THEORY\n";
    std::cout << "   - Trust: 0.85\n\n";
    
    // Concept 3: A hypothesis
    std::cout << "3. Storing HYPOTHESIS (under investigation):\n";
    EpistemicMetadata hypothesis_meta(
        EpistemicType::HYPOTHESIS,
        EpistemicStatus::ACTIVE,
        0.65
    );
    ConceptId dark_matter_id = ltm.store_concept(
        "Dark Matter",
        "Hypothesized form of matter that does not interact with light",
        hypothesis_meta
    );
    std::cout << "   ✓ Concept ID: " << dark_matter_id << "\n";
    std::cout << "   - Type: HYPOTHESIS\n";
    std::cout << "   - Trust: 0.65\n\n";
    
    // Concept 4: Speculation
    std::cout << "4. Storing SPECULATION (low certainty):\n";
    EpistemicMetadata speculation_meta(
        EpistemicType::SPECULATION,
        EpistemicStatus::ACTIVE,
        0.30
    );
    ConceptId multiverse_id = ltm.store_concept(
        "Multiverse",
        "Speculative hypothesis of multiple universes",
        speculation_meta
    );
    std::cout << "   ✓ Concept ID: " << multiverse_id << "\n";
    std::cout << "   - Type: SPECULATION\n";
    std::cout << "   - Trust: 0.30\n\n";
    
    // Concept 5: An old theory that was invalidated
    std::cout << "5. Storing and then INVALIDATING outdated knowledge:\n";
    EpistemicMetadata phlogiston_meta(
        EpistemicType::THEORY,
        EpistemicStatus::ACTIVE,
        0.75
    );
    ConceptId phlogiston_id = ltm.store_concept(
        "Phlogiston Theory",
        "Historical theory of combustion (now known to be incorrect)",
        phlogiston_meta
    );
    std::cout << "   ✓ Initially stored as THEORY with trust 0.75\n";
    
    // Invalidate it
    ltm.invalidate_concept(phlogiston_id, 0.05);
    auto phlogiston_after = ltm.retrieve_concept(phlogiston_id);
    std::cout << "   ✓ INVALIDATED (knowledge preserved, not deleted)\n";
    std::cout << "   - Status: " << epistemic_status_to_string(phlogiston_after->epistemic.status) << "\n";
    std::cout << "   - Trust: " << phlogiston_after->epistemic.trust << " (very low)\n\n";
    
    print_separator("Phase 2: Query by Epistemic Type");
    
    std::cout << "\nQuerying LTM by epistemic type:\n\n";
    
    auto facts = ltm.get_concepts_by_type(EpistemicType::FACT);
    std::cout << "FACTS: " << facts.size() << " concept(s)\n";
    for (auto id : facts) {
        auto info = ltm.retrieve_concept(id);
        std::cout << "  - " << info->label << " (trust: " << info->epistemic.trust << ")\n";
    }
    
    auto theories = ltm.get_concepts_by_type(EpistemicType::THEORY);
    std::cout << "\nTHEORIES: " << theories.size() << " concept(s)\n";
    for (auto id : theories) {
        auto info = ltm.retrieve_concept(id);
        std::cout << "  - " << info->label << " (trust: " << info->epistemic.trust << ")\n";
    }
    
    auto hypotheses = ltm.get_concepts_by_type(EpistemicType::HYPOTHESIS);
    std::cout << "\nHYPOTHESES: " << hypotheses.size() << " concept(s)\n";
    for (auto id : hypotheses) {
        auto info = ltm.retrieve_concept(id);
        std::cout << "  - " << info->label << " (trust: " << info->epistemic.trust << ")\n";
    }
    
    auto speculations = ltm.get_concepts_by_type(EpistemicType::SPECULATION);
    std::cout << "\nSPECULATIONS: " << speculations.size() << " concept(s)\n";
    for (auto id : speculations) {
        auto info = ltm.retrieve_concept(id);
        std::cout << "  - " << info->label << " (trust: " << info->epistemic.trust << ")\n";
    }
    
    print_separator("Phase 3: Activate Concepts in STM");
    
    ContextId ctx = brain.create_context();
    brain.begin_thinking(ctx);
    
    // Activate concepts from LTM into STM
    brain.activate_concept_in_context(ctx, cat_id, 0.95, ActivationClass::CORE_KNOWLEDGE);
    brain.activate_concept_in_context(ctx, evolution_id, 0.85, ActivationClass::CORE_KNOWLEDGE);
    brain.activate_concept_in_context(ctx, dark_matter_id, 0.70, ActivationClass::CONTEXTUAL);
    brain.activate_concept_in_context(ctx, multiverse_id, 0.40, ActivationClass::CONTEXTUAL);
    
    std::cout << "✓ Activated 4 concepts in STM from LTM\n";
    
    print_separator("Phase 4: Import External Knowledge (Importer Demo)");
    
    std::cout << "\nDemonstrating Wikipedia Importer:\n";
    WikipediaImporter wiki_importer;
    auto wiki_proposal = wiki_importer.parse_wikipedia_text(
        "Quantum Mechanics",
        "Quantum mechanics is a fundamental theory in physics that describes "
        "the behavior of matter and energy at atomic and subatomic scales."
    );
    
    std::cout << "✓ Wikipedia proposal created\n";
    std::cout << "  Suggested type: " << 
        (wiki_proposal->suggested_epistemic_type == SuggestedEpistemicType::DEFINITION_CANDIDATE ? 
         "DEFINITION_CANDIDATE" : "OTHER") << " (SUGGESTION ONLY)\n";
    std::cout << "  ⚠ Importer does NOT assign epistemic metadata\n";
    std::cout << "  ⚠ Human must review and explicitly decide\n\n";
    
    std::cout << "Human reviews proposal and decides:\n";
    std::cout << "  Decision: THEORY (well-established physics)\n";
    std::cout << "  Trust: 0.95\n\n";
    
    EpistemicMetadata qm_meta(
        EpistemicType::THEORY,
        EpistemicStatus::ACTIVE,
        0.95
    );
    ConceptId qm_id = ltm.store_concept(
        wiki_proposal->title,
        wiki_proposal->extracted_text,
        qm_meta  // EXPLICIT human decision
    );
    
    std::cout << "✓ Stored in LTM with ID: " << qm_id << "\n";
    std::cout << "  Type: " << epistemic_type_to_string(qm_meta.type) << "\n";
    std::cout << "  Status: " << epistemic_status_to_string(qm_meta.status) << "\n";
    std::cout << "  Trust: " << qm_meta.trust << "\n";
    
    print_separator("Phase 5: Generate Snapshot with Epistemic Metadata");
    
    std::string json_snapshot = snapshot_gen.generate_json_snapshot(
        &brain,
        &ltm,  // LTM provides epistemic metadata
        &curiosity,
        ctx
    );
    
    std::cout << "\nSnapshot JSON (with epistemic metadata):\n";
    std::cout << "─────────────────────────────────────────\n";
    std::cout << json_snapshot;
    std::cout << "─────────────────────────────────────────\n";
    
    // Save to file
    std::ofstream file("snapshot.json");
    if (file.is_open()) {
        file << json_snapshot;
        file.close();
        std::cout << "\n✓ Snapshot saved to: snapshot.json\n";
    }
    
    print_separator("Phase 6: System Status Summary");
    
    std::cout << "\nLTM Statistics:\n";
    std::cout << "  Total concepts: 6\n";
    std::cout << "  Active: " << ltm.get_active_concepts().size() << "\n";
    std::cout << "  Invalidated: " << ltm.get_concepts_by_status(EpistemicStatus::INVALIDATED).size() << "\n";
    std::cout << "\nEpistemic Distribution:\n";
    std::cout << "  Facts: " << ltm.get_concepts_by_type(EpistemicType::FACT).size() << "\n";
    std::cout << "  Theories: " << ltm.get_concepts_by_type(EpistemicType::THEORY).size() << "\n";
    std::cout << "  Hypotheses: " << ltm.get_concepts_by_type(EpistemicType::HYPOTHESIS).size() << "\n";
    std::cout << "  Speculations: " << ltm.get_concepts_by_type(EpistemicType::SPECULATION).size() << "\n";
    
    print_separator("Phase 7: Cleanup");
    
    brain.end_thinking(ctx);
    brain.destroy_context(ctx);
    brain.shutdown();
    
    std::cout << "✓ All subsystems shut down\n";
    
    print_separator("Demo Complete");
    
    std::cout << "\n";
    std::cout << "═══════════════════════════════════════════════════════\n";
    std::cout << "  EPISTEMIC ENFORCEMENT DEMONSTRATION COMPLETE\n";
    std::cout << "═══════════════════════════════════════════════════════\n";
    std::cout << "\n";
    std::cout << "KEY POINTS DEMONSTRATED:\n";
    std::cout << "  ✓ All knowledge has explicit epistemic metadata\n";
    std::cout << "  ✓ Facts distinguishable from speculation\n";
    std::cout << "  ✓ Trust values differentiate certainty\n";
    std::cout << "  ✓ Invalidation preserves knowledge (no deletion)\n";
    std::cout << "  ✓ Importers only suggest, humans decide\n";
    std::cout << "  ✓ Snapshot exposes epistemic metadata\n";
    std::cout << "\n";
    std::cout << "BUG-001 STATUS: CLOSED ✅\n";
    std::cout << "  No implicit defaults\n";
    std::cout << "  No silent fallbacks\n";
    std::cout << "  No inferred epistemic state\n";
    std::cout << "\n";
    
    return 0;
}
