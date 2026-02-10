// Phase 1.1 Test: PersistentLTM — create 1000 concepts, persist, reload, verify
#include "persistent/persistent_ltm.hpp"
#include <iostream>
#include <cassert>
#include <filesystem>
#include <string>

using namespace brain19;
using namespace brain19::persistent;

int main() {
    const std::string test_dir = "/tmp/brain19_persistent_test";
    
    // Clean up from previous runs
    std::filesystem::remove_all(test_dir);
    
    std::cout << "=== Phase 1.1: PersistentLTM Test ===" << std::endl;
    
    // --- Phase A: Create and populate ---
    {
        PersistentLTM ltm(test_dir);
        
        // Store 1000 concepts
        for (int i = 1; i <= 1000; ++i) {
            std::string label = "concept_" + std::to_string(i);
            std::string def = "Definition of concept number " + std::to_string(i);
            
            EpistemicType type = static_cast<EpistemicType>(i % 6);
            double trust = 0.5 + (i % 50) * 0.01;
            
            ConceptId id = ltm.store_concept(label, def,
                EpistemicMetadata(type, EpistemicStatus::ACTIVE, trust));
            assert(id == static_cast<uint64_t>(i));
        }
        
        // Add some relations
        for (int i = 1; i <= 500; ++i) {
            auto rid = ltm.add_relation(
                i, i + 1,
                static_cast<RelationType>(i % 10),
                0.5 + (i % 5) * 0.1
            );
            assert(rid.has_value());
        }
        
        // Remove a few relations
        ltm.remove_relation(1);
        ltm.remove_relation(5);
        
        // Invalidate a concept
        ltm.invalidate_concept(42, 0.05);
        
        assert(ltm.concept_count() == 1000);
        assert(ltm.relation_count() == 498);  // 500 - 2 removed
        
        ltm.sync();
        std::cout << "Phase A: Created 1000 concepts, 498 relations. Synced." << std::endl;
    }
    // PersistentLTM destroyed — files on disk
    
    // --- Phase B: Reload and verify ---
    {
        PersistentLTM ltm(test_dir);
        
        assert(ltm.concept_count() == 1000);
        assert(ltm.relation_count() == 498);
        
        // Verify concept 1
        auto c1 = ltm.retrieve_concept(1);
        assert(c1.has_value());
        assert(c1->label == "concept_1");
        assert(c1->definition == "Definition of concept number 1");
        assert(c1->epistemic.type == static_cast<EpistemicType>(1 % 6));
        assert(c1->epistemic.status == EpistemicStatus::ACTIVE);
        
        // Verify concept 500
        auto c500 = ltm.retrieve_concept(500);
        assert(c500.has_value());
        assert(c500->label == "concept_500");
        
        // Verify concept 1000
        auto c1000 = ltm.retrieve_concept(1000);
        assert(c1000.has_value());
        assert(c1000->label == "concept_1000");
        assert(c1000->definition == "Definition of concept number 1000");
        
        // Verify invalidated concept 42
        auto c42 = ltm.retrieve_concept(42);
        assert(c42.has_value());
        assert(c42->epistemic.status == EpistemicStatus::INVALIDATED);
        assert(c42->epistemic.trust < 0.1);
        
        // Verify relation 2 exists (wasn't removed)
        auto r2 = ltm.get_relation(2);
        assert(r2.has_value());
        assert(r2->source == 2);
        assert(r2->target == 3);
        
        // Verify relation 1 removed
        auto r1 = ltm.get_relation(1);
        assert(!r1.has_value());
        
        // Verify outgoing relations
        auto out = ltm.get_outgoing_relations(2);
        assert(out.size() == 1);
        
        // Verify all concept IDs
        auto all_ids = ltm.get_all_concept_ids();
        assert(all_ids.size() == 1000);
        
        // Verify active concepts (should be 999 since one was invalidated)
        auto active = ltm.get_active_concepts();
        assert(active.size() == 999);
        
        // Can still add new concepts after reload
        ConceptId new_id = ltm.store_concept("new_concept", "After reload",
            EpistemicMetadata(EpistemicType::FACT, EpistemicStatus::ACTIVE, 0.9));
        assert(new_id == 1001);
        assert(ltm.concept_count() == 1001);
        
        std::cout << "Phase B: Reload verified! All 1000 concepts + 498 relations intact." << std::endl;
    }
    
    // Cleanup
    std::filesystem::remove_all(test_dir);
    
    std::cout << "\n✅ ALL TESTS PASSED — Phase 1.1 mmap persistence working!" << std::endl;
    return 0;
}
