#include <cassert>
#include <iostream>
#include <string>
#include <vector>

#include "../backend/ltm/long_term_memory.hpp"
#include "../backend/epistemic/epistemic_metadata.hpp"
#include "../backend/curiosity/curiosity_trigger.hpp"
#include "../backend/understanding/understanding_proposals.hpp"
#include "../backend/evolution/concept_proposal.hpp"
#include "../backend/evolution/epistemic_promotion.hpp"
#include "../backend/evolution/pattern_discovery.hpp"

using namespace brain19;

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) \
    std::cout << "TEST: " << #name << " ... "; \
    try { test_##name(); tests_passed++; std::cout << "PASSED\n"; } \
    catch (const std::exception& e) { tests_failed++; std::cout << "FAILED: " << e.what() << "\n"; }

#define ASSERT(cond) \
    if (!(cond)) throw std::runtime_error("Assertion failed: " #cond " at line " + std::to_string(__LINE__))

// ─── Helper: Build a small knowledge graph ────────────────────────────────

static LongTermMemory build_test_ltm() {
    LongTermMemory ltm;

    // Create some base concepts
    auto c1 = ltm.store_concept("animal", "Living organism",
        EpistemicMetadata(EpistemicType::FACT, EpistemicStatus::ACTIVE, 0.95));
    auto c2 = ltm.store_concept("dog", "Domestic canine",
        EpistemicMetadata(EpistemicType::FACT, EpistemicStatus::ACTIVE, 0.9));
    auto c3 = ltm.store_concept("cat", "Domestic feline",
        EpistemicMetadata(EpistemicType::FACT, EpistemicStatus::ACTIVE, 0.9));
    auto c4 = ltm.store_concept("mammal", "Warm-blooded vertebrate",
        EpistemicMetadata(EpistemicType::THEORY, EpistemicStatus::ACTIVE, 0.7));

    // IS_A relations
    ltm.add_relation(c2, c1, RelationType::IS_A, 1.0);    // dog IS_A animal
    ltm.add_relation(c3, c1, RelationType::IS_A, 1.0);    // cat IS_A animal
    ltm.add_relation(c2, c4, RelationType::IS_A, 1.0);    // dog IS_A mammal
    ltm.add_relation(c3, c4, RelationType::IS_A, 1.0);    // cat IS_A mammal

    // Some properties
    ltm.add_relation(c2, c3, RelationType::SIMILAR_TO, 0.7);

    return ltm;
}

// ─── Test 1: ConceptProposer from curiosity triggers ──────────────────────

void test_from_curiosity_triggers() {
    auto ltm = build_test_ltm();
    ConceptProposer proposer(ltm);

    std::vector<CuriosityTrigger> triggers;
    triggers.emplace_back(
        TriggerType::SHALLOW_RELATIONS, 1,
        std::vector<ConceptId>{1, 2},
        "Shallow relations between animal and dog"
    );
    triggers.emplace_back(
        TriggerType::MISSING_DEPTH, 2,
        std::vector<ConceptId>{3},
        "Missing depth for cat"
    );

    auto proposals = proposer.from_curiosity(triggers);
    ASSERT(proposals.size() == 2);
    ASSERT(proposals[0].initial_type == EpistemicType::SPECULATION);
    ASSERT(proposals[0].initial_trust <= 0.5);
    ASSERT(proposals[0].source.find("curiosity:") == 0);
}

// ─── Test 2: ConceptProposer from relevance anomalies ─────────────────────

void test_from_relevance_anomalies() {
    auto ltm = build_test_ltm();
    ConceptProposer proposer(ltm);

    // Create a relevance map with an anomalous high score between unconnected concepts
    // c1=animal, c4=mammal — they don't have direct relation in our test
    RelevanceMap map(1);  // source = concept 1 (animal)
    // We can't easily set scores directly, so test with empty map
    auto proposals = proposer.from_relevance_anomalies(map, 0.8);
    // Empty map → no anomalies
    ASSERT(proposals.empty());
}

// ─── Test 3: ConceptProposer deduplication + ranking ──────────────────────

void test_dedup_and_ranking() {
    auto ltm = build_test_ltm();
    ConceptProposer proposer(ltm);

    std::vector<ConceptProposal> proposals;
    proposals.emplace_back("concept_a", "desc a", EpistemicType::SPECULATION,
                           0.2, "src1", std::vector<ConceptId>{1, 2}, "reason1");
    proposals.emplace_back("concept_a", "desc a duplicate", EpistemicType::SPECULATION,
                           0.3, "src2", std::vector<ConceptId>{1}, "reason2");
    proposals.emplace_back("concept_b", "desc b", EpistemicType::HYPOTHESIS,
                           0.4, "src3", std::vector<ConceptId>{1, 2, 3}, "reason3");

    auto ranked = proposer.rank_proposals(proposals, 5);
    ASSERT(ranked.size() == 2);  // Deduplicated: concept_a appears once

    // concept_b should rank higher (HYPOTHESIS + more evidence)
    ASSERT(ranked[0].label == "concept_b");
}

// ─── Test 4: EpistemicPromotion SPECULATION → HYPOTHESIS ──────────────────

void test_speculation_to_hypothesis() {
    LongTermMemory ltm;

    // Create a speculation concept
    auto target = ltm.store_concept("new_idea", "A new speculation",
        EpistemicMetadata(EpistemicType::SPECULATION, EpistemicStatus::ACTIVE, 0.2));

    // Create 3+ supporting concepts with SUPPORTS relations
    auto s1 = ltm.store_concept("evidence1", "ev1",
        EpistemicMetadata(EpistemicType::FACT, EpistemicStatus::ACTIVE, 0.9));
    auto s2 = ltm.store_concept("evidence2", "ev2",
        EpistemicMetadata(EpistemicType::THEORY, EpistemicStatus::ACTIVE, 0.7));
    auto s3 = ltm.store_concept("evidence3", "ev3",
        EpistemicMetadata(EpistemicType::FACT, EpistemicStatus::ACTIVE, 0.85));

    ltm.add_relation(s1, target, RelationType::SUPPORTS, 0.8);
    ltm.add_relation(s2, target, RelationType::SUPPORTS, 0.7);
    ltm.add_relation(s3, target, RelationType::SUPPORTS, 0.9);

    EpistemicPromotion promo(ltm);
    auto candidate = promo.evaluate(target);
    ASSERT(candidate.has_value());
    ASSERT(candidate->proposed_type == EpistemicType::HYPOTHESIS);
    ASSERT(!candidate->requires_human_review);

    // Apply promotion
    bool ok = promo.promote(target, candidate->proposed_type, candidate->proposed_trust);
    ASSERT(ok);

    auto updated = ltm.retrieve_concept(target);
    ASSERT(updated->epistemic.type == EpistemicType::HYPOTHESIS);
}

// ─── Test 5: EpistemicPromotion HYPOTHESIS → THEORY ───────────────────────

void test_hypothesis_to_theory() {
    LongTermMemory ltm;

    auto target = ltm.store_concept("growing_idea", "Maturing hypothesis",
        EpistemicMetadata(EpistemicType::HYPOTHESIS, EpistemicStatus::ACTIVE, 0.4));

    // Need 5+ supporting relations from THEORY+ concepts
    for (int i = 0; i < 6; ++i) {
        auto s = ltm.store_concept("theory_ev_" + std::to_string(i), "evidence",
            EpistemicMetadata(EpistemicType::THEORY, EpistemicStatus::ACTIVE, 0.75));
        ltm.add_relation(s, target, RelationType::SUPPORTS, 0.8);
    }

    EpistemicPromotion promo(ltm);
    auto candidate = promo.evaluate(target);
    ASSERT(candidate.has_value());
    ASSERT(candidate->proposed_type == EpistemicType::THEORY);
    ASSERT(!candidate->requires_human_review);
}

// ─── Test 6: THEORY → FACT requires human review ─────────────────────────

void test_theory_to_fact_requires_human() {
    LongTermMemory ltm;

    auto target = ltm.store_concept("well_supported", "Strong theory",
        EpistemicMetadata(EpistemicType::THEORY, EpistemicStatus::ACTIVE, 0.7));

    // Add many strong supports
    for (int i = 0; i < 6; ++i) {
        auto s = ltm.store_concept("fact_ev_" + std::to_string(i), "strong evidence",
            EpistemicMetadata(EpistemicType::FACT, EpistemicStatus::ACTIVE, 0.95));
        ltm.add_relation(s, target, RelationType::SUPPORTS, 0.9);
    }

    EpistemicPromotion promo(ltm);
    auto candidate = promo.evaluate(target);
    ASSERT(candidate.has_value());
    ASSERT(candidate->proposed_type == EpistemicType::FACT);
    ASSERT(candidate->requires_human_review);  // MUST require human!

    // promote() should REFUSE FACT promotion
    bool refused = promo.promote(target, EpistemicType::FACT, 0.9);
    ASSERT(!refused);

    // Only confirm_as_fact works
    bool confirmed = promo.confirm_as_fact(target, 0.9, "Human verified");
    ASSERT(confirmed);

    auto updated = ltm.retrieve_concept(target);
    ASSERT(updated->epistemic.type == EpistemicType::FACT);
    ASSERT(updated->epistemic.trust >= 0.8);
}

// ─── Test 7: Demotion on contradiction ────────────────────────────────────

void test_demotion_on_contradiction() {
    LongTermMemory ltm;

    auto target = ltm.store_concept("shaky_theory", "Theory with issues",
        EpistemicMetadata(EpistemicType::THEORY, EpistemicStatus::ACTIVE, 0.7));

    auto contra = ltm.store_concept("contradicting", "Counter-evidence",
        EpistemicMetadata(EpistemicType::FACT, EpistemicStatus::ACTIVE, 0.9));

    ltm.add_relation(contra, target, RelationType::CONTRADICTS, 0.8);

    EpistemicPromotion promo(ltm);
    auto candidate = promo.evaluate(target);
    ASSERT(candidate.has_value());
    ASSERT(candidate->proposed_type == EpistemicType::HYPOTHESIS);  // Demoted
    ASSERT(candidate->proposed_trust < 0.7);
}

// ─── Test 8: PatternDiscovery find clusters ───────────────────────────────

void test_find_clusters() {
    auto ltm = build_test_ltm();
    PatternDiscovery discovery(ltm);

    auto clusters = discovery.find_clusters(2);
    ASSERT(!clusters.empty());
    ASSERT(clusters[0].pattern_type == "cluster");
    ASSERT(clusters[0].involved_concepts.size() >= 2);
}

// ─── Test 9: PatternDiscovery find bridges ────────────────────────────────

void test_find_bridges() {
    LongTermMemory ltm;

    // Cluster A
    auto a1 = ltm.store_concept("a1", "cluster a",
        EpistemicMetadata(EpistemicType::FACT, EpistemicStatus::ACTIVE, 0.9));
    auto a2 = ltm.store_concept("a2", "cluster a",
        EpistemicMetadata(EpistemicType::FACT, EpistemicStatus::ACTIVE, 0.9));
    ltm.add_relation(a1, a2, RelationType::SUPPORTS, 1.0);

    // Cluster B
    auto b1 = ltm.store_concept("b1", "cluster b",
        EpistemicMetadata(EpistemicType::FACT, EpistemicStatus::ACTIVE, 0.9));
    auto b2 = ltm.store_concept("b2", "cluster b",
        EpistemicMetadata(EpistemicType::FACT, EpistemicStatus::ACTIVE, 0.9));
    ltm.add_relation(b1, b2, RelationType::SUPPORTS, 1.0);

    // Bridge concept connecting both clusters
    auto bridge = ltm.store_concept("bridge", "connects clusters",
        EpistemicMetadata(EpistemicType::THEORY, EpistemicStatus::ACTIVE, 0.6));
    ltm.add_relation(a1, bridge, RelationType::SUPPORTS, 0.7);
    ltm.add_relation(bridge, b1, RelationType::SUPPORTS, 0.7);

    PatternDiscovery discovery(ltm);

    // With the bridge, everything is one component
    // The bridge concept links two otherwise separate groups
    auto all = discovery.discover_all();
    ASSERT(!all.empty());

    // Should find at least a cluster
    bool found_cluster = false;
    for (const auto& p : all) {
        if (p.pattern_type == "cluster") found_cluster = true;
    }
    ASSERT(found_cluster);
}

// ─── Test 10: PatternDiscovery find gaps ──────────────────────────────────

void test_find_gaps() {
    LongTermMemory ltm;

    auto animal = ltm.store_concept("animal", "Living thing",
        EpistemicMetadata(EpistemicType::FACT, EpistemicStatus::ACTIVE, 0.95));
    auto dog = ltm.store_concept("dog", "Canine",
        EpistemicMetadata(EpistemicType::FACT, EpistemicStatus::ACTIVE, 0.9));
    auto cat = ltm.store_concept("cat", "Feline",
        EpistemicMetadata(EpistemicType::FACT, EpistemicStatus::ACTIVE, 0.9));
    auto fur = ltm.store_concept("fur", "Body covering",
        EpistemicMetadata(EpistemicType::FACT, EpistemicStatus::ACTIVE, 0.9));

    // Both IS_A animal
    ltm.add_relation(dog, animal, RelationType::IS_A, 1.0);
    ltm.add_relation(cat, animal, RelationType::IS_A, 1.0);

    // Dog HAS_PROPERTY fur, but cat doesn't (gap!)
    ltm.add_relation(dog, fur, RelationType::HAS_PROPERTY, 0.9);

    PatternDiscovery discovery(ltm);
    auto gaps = discovery.find_gaps();

    // Should find gap: cat should maybe have HAS_PROPERTY fur
    ASSERT(!gaps.empty());
    bool found_cat_gap = false;
    for (const auto& g : gaps) {
        if (g.pattern_type == "gap") {
            for (auto c : g.involved_concepts) {
                if (c == cat) { found_cat_gap = true; break; }
            }
        }
    }
    ASSERT(found_cat_gap);
}

// ─── Test 11: ConceptProposal trust cap ───────────────────────────────────

void test_trust_cap() {
    ConceptProposal p("test", "desc", EpistemicType::HYPOTHESIS, 0.9,
                      "src", {}, "reason");
    ASSERT(p.initial_trust <= 0.5);  // Capped at 0.5
}

// ─── Test 12: ConceptProposal type enforcement ────────────────────────────

void test_type_enforcement() {
    ConceptProposal p("test", "desc", EpistemicType::FACT, 0.3,
                      "src", {}, "reason");
    // FACT not allowed for system-generated → forced to SPECULATION
    ASSERT(p.initial_type == EpistemicType::SPECULATION);
}

// ─── Test 13: Maintenance result ──────────────────────────────────────────

void test_maintenance() {
    LongTermMemory ltm;

    // Create speculation with enough evidence for promotion
    auto target = ltm.store_concept("spec", "speculation",
        EpistemicMetadata(EpistemicType::SPECULATION, EpistemicStatus::ACTIVE, 0.2));
    for (int i = 0; i < 4; ++i) {
        auto s = ltm.store_concept("ev_" + std::to_string(i), "evidence",
            EpistemicMetadata(EpistemicType::FACT, EpistemicStatus::ACTIVE, 0.9));
        ltm.add_relation(s, target, RelationType::SUPPORTS, 0.8);
    }

    EpistemicPromotion promo(ltm);
    auto result = promo.run_maintenance();
    ASSERT(result.promotions >= 1);
}

// ─── Main ─────────────────────────────────────────────────────────────────

int main() {
    std::cout << "=== Phase 6: Evolution Tests ===\n\n";

    TEST(from_curiosity_triggers);
    TEST(from_relevance_anomalies);
    TEST(dedup_and_ranking);
    TEST(speculation_to_hypothesis);
    TEST(hypothesis_to_theory);
    TEST(theory_to_fact_requires_human);
    TEST(demotion_on_contradiction);
    TEST(find_clusters);
    TEST(find_bridges);
    TEST(find_gaps);
    TEST(trust_cap);
    TEST(type_enforcement);
    TEST(maintenance);

    std::cout << "\n=== Results: " << tests_passed << " passed, "
              << tests_failed << " failed ===\n";

    return tests_failed > 0 ? 1 : 0;
}
