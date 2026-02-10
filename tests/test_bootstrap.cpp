#include <cassert>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

#include "../backend/bootstrap/foundation_concepts.hpp"
#include "../backend/bootstrap/bootstrap_interface.hpp"
#include "../backend/bootstrap/context_accumulator.hpp"

using namespace brain19;

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) \
    static void test_##name(); \
    struct Register_##name { Register_##name() { test_##name(); } } reg_##name; \
    static void test_##name()

#define ASSERT_TRUE(cond) do { \
    if (!(cond)) { \
        std::cerr << "  FAIL: " #cond " (" << __FILE__ << ":" << __LINE__ << ")\n"; \
        tests_failed++; return; \
    } \
} while(0)

#define ASSERT_EQ(a, b) do { \
    if ((a) != (b)) { \
        std::cerr << "  FAIL: " #a " == " #b " (got " << (a) << " vs " << (b) \
                  << ") (" << __FILE__ << ":" << __LINE__ << ")\n"; \
        tests_failed++; return; \
    } \
} while(0)

#define PASS() do { tests_passed++; std::cout << "  PASS\n"; } while(0)

// ─── Test 1: Foundation seeding (all tiers) ────────────────────────────────

TEST(foundation_seeding) {
    std::cout << "[1] Foundation seeding (all tiers)...\n";
    LongTermMemory ltm;
    FoundationConcepts::seed_all(ltm);

    // Should have concepts stored
    auto all_ids = ltm.get_all_concept_ids();
    ASSERT_TRUE(all_ids.size() >= 180);  // T1(50)+T2(96)+T4(85) concepts

    // Check a Tier 1 concept exists
    bool found_entity = false;
    for (auto id : all_ids) {
        auto info = ltm.retrieve_concept(id);
        if (info && info->label == "Entity") {
            found_entity = true;
            ASSERT_TRUE(info->epistemic.type == EpistemicType::DEFINITION);
            ASSERT_TRUE(info->epistemic.status == EpistemicStatus::ACTIVE);
            ASSERT_TRUE(info->epistemic.trust >= 0.95);
            break;
        }
    }
    ASSERT_TRUE(found_entity);
    PASS();
}

// ─── Test 2: Foundation concept count + relation count ─────────────────────

TEST(foundation_counts) {
    std::cout << "[2] Foundation concept count + relation count...\n";
    size_t concepts = FoundationConcepts::concept_count();
    size_t relations = FoundationConcepts::relation_count();

    ASSERT_TRUE(concepts >= 180);    // Tier1(50) + Tier2(~96) + Tier4(~85)
    ASSERT_TRUE(relations >= 130);   // ~145 relations in tier 3

    std::cout << "    Concepts: " << concepts << ", Relations: " << relations << "\n";

    // Verify actual storage matches
    LongTermMemory ltm;
    FoundationConcepts::seed_all(ltm);
    auto all_ids = ltm.get_all_concept_ids();
    ASSERT_EQ(all_ids.size(), concepts);
    PASS();
}

// ─── Test 3: Guided proposal generation from text ──────────────────────────

TEST(proposal_generation) {
    std::cout << "[3] Guided proposal generation from text...\n";
    LongTermMemory ltm;
    BootstrapInterface bi(ltm);
    bi.initialize_foundation();

    std::string text = "Albert Einstein developed the theory of General Relativity "
                       "at the University of Berlin. His work on the Photoelectric "
                       "Effect earned him the Nobel Prize.";

    auto proposals = bi.process_text(text);

    // Should extract some candidates (capitalised words not in foundation)
    ASSERT_TRUE(!proposals.empty());

    // "Albert" or "Einstein" should be among candidates
    bool found_einstein = false;
    for (const auto& p : proposals) {
        if (p.entity_name == "Einstein" || p.entity_name == "Albert") {
            found_einstein = true;
            ASSERT_TRUE(!p.context_text.empty());
            ASSERT_TRUE(!p.auto_description.empty());
            ASSERT_TRUE(p.suggested_trust > 0.0);
        }
    }
    ASSERT_TRUE(found_einstein);
    PASS();
}

// ─── Test 4: Type suggestion based on existing knowledge ───────────────────

TEST(type_suggestion) {
    std::cout << "[4] Type suggestion based on existing knowledge...\n";
    ContextAccumulator acc;
    acc.record_concept("Cell", "biology");
    acc.record_concept("DNA", "biology");

    auto types = acc.suggest_types("Cell_Membrane");
    // Should suggest Organism (contains "cell")
    bool has_organism = false;
    for (const auto& t : types) {
        if (t == "Organism") has_organism = true;
    }
    ASSERT_TRUE(has_organism);

    // Unknown entity should get "Entity"
    auto generic = acc.suggest_types("Xyzzyplugh");
    ASSERT_TRUE(!generic.empty());
    ASSERT_EQ(generic[0], std::string("Entity"));
    PASS();
}

// ─── Test 5: Similar concept detection ─────────────────────────────────────

TEST(similar_concept_detection) {
    std::cout << "[5] Similar concept detection...\n";
    LongTermMemory ltm;
    BootstrapInterface bi(ltm);
    bi.initialize_foundation();

    // Process text with a concept similar to existing ones
    std::string text = "Thermodynamics describes how Heat transfers between systems.";
    auto proposals = bi.process_text(text);

    // "Thermodynamics" is already in foundation, so should NOT be proposed
    bool thermodynamics_proposed = false;
    for (const auto& p : proposals) {
        if (p.entity_name == "Thermodynamics") {
            thermodynamics_proposed = true;
        }
    }
    ASSERT_TRUE(!thermodynamics_proposed);

    // Known concept count should include foundation
    ASSERT_TRUE(bi.known_concepts() >= 200);
    PASS();
}

// ─── Test 6: Accept/reject workflow ────────────────────────────────────────

TEST(accept_reject_workflow) {
    std::cout << "[6] Accept/reject workflow...\n";
    LongTermMemory ltm;
    BootstrapInterface bi(ltm);
    bi.initialize_foundation();

    size_t initial_count = bi.known_concepts();

    std::string text = "Nikola Tesla invented Alternating Current systems.";
    auto proposals = bi.process_text(text);
    ASSERT_TRUE(!proposals.empty());

    size_t initial_pending = bi.pending_proposals();
    ASSERT_TRUE(initial_pending > 0);

    // Accept first proposal
    auto& first = proposals[0];
    bi.accept_proposal(first, "A concept from the test.", EpistemicType::FACT, 0.90);

    ASSERT_EQ(bi.known_concepts(), initial_count + 1);
    ASSERT_TRUE(bi.pending_proposals() < initial_pending);

    // Reject remaining if any
    if (proposals.size() > 1) {
        bi.reject_proposal(proposals[1], "Not relevant.");
        // Re-processing same text shouldn't re-propose rejected
        auto re = bi.process_text(text);
        bool found_rejected = false;
        for (const auto& p : re) {
            if (p.entity_name == proposals[1].entity_name) found_rejected = true;
        }
        ASSERT_TRUE(!found_rejected);
    }
    PASS();
}

// ─── Test 7: Context accumulation over multiple texts ──────────────────────

TEST(context_accumulation) {
    std::cout << "[7] Context accumulation over multiple texts...\n";
    ContextAccumulator acc;

    // Add concepts from different domains
    acc.record_concept("Cell", "biology");
    acc.record_concept("DNA", "biology");
    acc.record_concept("Gene", "biology");
    acc.record_concept("Atom", "physics");
    acc.record_concept("Force", "physics");

    acc.record_text_processed("Biology text 1");
    acc.record_text_processed("Physics text 1");

    ASSERT_EQ(acc.total_concepts(), size_t(5));
    ASSERT_EQ(acc.texts_processed(), size_t(2));
    ASSERT_EQ(acc.concept_frequency("Cell"), size_t(1));

    // Record again to increase frequency
    acc.record_concept("Cell", "biology");
    ASSERT_EQ(acc.concept_frequency("Cell"), size_t(2));

    // Domain stats
    auto stats = acc.get_domain_stats();
    ASSERT_TRUE(!stats.empty());

    // Biology should have highest count (3 concepts)
    bool bio_found = false;
    for (const auto& s : stats) {
        if (s.domain == "biology") {
            bio_found = true;
            ASSERT_EQ(s.concept_count, size_t(3));
        }
    }
    ASSERT_TRUE(bio_found);
    PASS();
}

// ─── Test 8: Progressive complexity suggestions ────────────────────────────

TEST(progressive_suggestions) {
    std::cout << "[8] Progressive complexity suggestions...\n";
    LongTermMemory ltm;
    BootstrapInterface bi(ltm);
    bi.initialize_foundation();

    auto suggestions = bi.suggest_next_topics();
    ASSERT_TRUE(!suggestions.empty());

    // Suggestions should be meaningful strings
    for (const auto& s : suggestions) {
        ASSERT_TRUE(!s.empty());
        ASSERT_TRUE(s.size() > 3);
    }

    std::cout << "    Suggested topics: ";
    for (size_t i = 0; i < std::min(suggestions.size(), size_t(3)); ++i) {
        std::cout << suggestions[i];
        if (i + 1 < std::min(suggestions.size(), size_t(3))) std::cout << ", ";
    }
    std::cout << "\n";
    PASS();
}

// ─── Test 9: Foundation epistemic integrity ────────────────────────────────

TEST(epistemic_integrity) {
    std::cout << "[9] Foundation epistemic integrity...\n";
    LongTermMemory ltm;
    FoundationConcepts::seed_all(ltm);

    auto all_ids = ltm.get_all_concept_ids();
    for (auto id : all_ids) {
        auto info = ltm.retrieve_concept(id);
        ASSERT_TRUE(info.has_value());
        // All foundation concepts must be ACTIVE
        ASSERT_TRUE(info->epistemic.status == EpistemicStatus::ACTIVE);
        // All must have high trust
        ASSERT_TRUE(info->epistemic.trust >= 0.95);
        // Must be DEFINITION or FACT
        ASSERT_TRUE(info->epistemic.type == EpistemicType::DEFINITION ||
                    info->epistemic.type == EpistemicType::FACT);
    }
    PASS();
}

// ─── Test 10: Knowledge gap detection ──────────────────────────────────────

TEST(knowledge_gap_detection) {
    std::cout << "[10] Knowledge gap detection...\n";
    ContextAccumulator acc;

    // Only add biology concepts — other domains should be gaps
    for (int i = 0; i < 20; ++i) {
        acc.record_concept("bio_" + std::to_string(i), "biology");
    }

    auto gaps = acc.find_knowledge_gaps();
    ASSERT_TRUE(!gaps.empty());

    // "physics" should be in gaps
    bool physics_gap = false;
    for (const auto& g : gaps) {
        if (g == "physics") physics_gap = true;
    }
    ASSERT_TRUE(physics_gap);
    PASS();
}

// ─── Main ──────────────────────────────────────────────────────────────────

int main() {
    std::cout << "=== Brain19 Bootstrap Tests ===\n\n";

    // Tests are auto-registered via static constructors above
    // (they already ran)

    std::cout << "\n=== Results: " << tests_passed << " passed, "
              << tests_failed << " failed ===\n";

    return tests_failed > 0 ? 1 : 0;
}
