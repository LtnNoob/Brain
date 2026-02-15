#include "property_inheritance.hpp"
#include "../ltm/long_term_memory.hpp"
#include "../memory/active_relation.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
#include <string>

using namespace brain19;

static int tests_passed = 0;
static int tests_total = 0;

#define TEST(name) do { \
    tests_total++; \
    std::cout << "  TEST " << tests_total << ": " << name << "... "; \
} while(0)

#define PASS() do { \
    tests_passed++; \
    std::cout << "PASS" << std::endl; \
} while(0)

static constexpr double EPS = 1e-6;

static bool approx(double a, double b) {
    return std::fabs(a - b) < EPS;
}

// Helper: create a FACT concept
static ConceptId add_cpt(LongTermMemory& ltm,
                         const std::string& label,
                         const std::string& def = "") {
    return ltm.store_concept(label, def.empty() ? label : def,
        EpistemicMetadata(EpistemicType::FACT, EpistemicStatus::ACTIVE, 0.95));
}

// =============================================================================
// Test 1: Basic single-hop inheritance
// =============================================================================
//   Hund IS_A Animal
//   Animal HAS_PROPERTY Spine
//   => Hund inherits Spine at weight 0.9 * original
void test_basic_single_hop() {
    TEST("Basic single-hop property inheritance");

    LongTermMemory ltm;
    auto animal = add_cpt(ltm, "Animal");
    auto hund   = add_cpt(ltm, "Hund");
    auto spine  = add_cpt(ltm, "Spine");

    ltm.add_relation(hund, animal, RelationType::IS_A, 1.0);
    ltm.add_relation(animal, spine, RelationType::HAS_PROPERTY, 0.95);

    PropertyInheritance pi(ltm);
    PropertyInheritance::Config cfg;
    cfg.decay_per_hop = 0.9;
    cfg.trust_floor = 0.3;

    auto result = pi.propagate(cfg);

    assert(result.properties_inherited == 1);
    assert(result.converged);

    auto inherited = pi.get_inherited(hund);
    assert(inherited.size() == 1);
    assert(inherited[0].property_target == spine);
    assert(inherited[0].hop_count == 1);
    assert(approx(inherited[0].inherited_trust, 0.95 * 0.9));  // 0.855

    PASS();
}

// =============================================================================
// Test 2: Multi-hop chain (Pudel -> Hund -> Carnivore -> Mammal -> Animal)
// =============================================================================
void test_pudel_chain() {
    TEST("Multi-hop Pudel->Hund->Carnivore->Mammal->Animal chain");

    LongTermMemory ltm;
    auto animal    = add_cpt(ltm, "Animal");
    auto mammal    = add_cpt(ltm, "Mammal");
    auto carnivore = add_cpt(ltm, "Carnivore");
    auto hund      = add_cpt(ltm, "Hund");
    auto pudel     = add_cpt(ltm, "Pudel");
    auto spine     = add_cpt(ltm, "has_spine");

    // IS_A chain
    ltm.add_relation(pudel, hund, RelationType::IS_A, 1.0);
    ltm.add_relation(hund, carnivore, RelationType::IS_A, 1.0);
    ltm.add_relation(carnivore, mammal, RelationType::IS_A, 1.0);
    ltm.add_relation(mammal, animal, RelationType::IS_A, 1.0);

    // Property at the top
    ltm.add_relation(animal, spine, RelationType::HAS_PROPERTY, 0.95);

    PropertyInheritance pi(ltm);
    PropertyInheritance::Config cfg;
    cfg.decay_per_hop = 0.9;
    cfg.trust_floor = 0.3;

    auto result = pi.propagate(cfg);

    assert(result.converged);
    // Should propagate to all 4 descendants
    assert(result.properties_inherited == 4);

    // Check Mammal (1 hop from Animal): 0.95 * 0.9 = 0.855
    auto mammal_props = pi.get_inherited(mammal);
    assert(mammal_props.size() == 1);
    assert(approx(mammal_props[0].inherited_trust, 0.855));
    assert(mammal_props[0].hop_count == 1);

    // Check Carnivore (2 hops): 0.855 * 0.9 = 0.7695
    auto carn_props = pi.get_inherited(carnivore);
    assert(carn_props.size() == 1);
    assert(approx(carn_props[0].inherited_trust, 0.7695));
    assert(carn_props[0].hop_count == 2);

    // Check Hund (3 hops): 0.7695 * 0.9 = 0.69255
    auto hund_props = pi.get_inherited(hund);
    assert(hund_props.size() == 1);
    assert(approx(hund_props[0].inherited_trust, 0.69255));
    assert(hund_props[0].hop_count == 3);

    // Check Pudel (4 hops): 0.69255 * 0.9 = 0.623295
    auto pudel_props = pi.get_inherited(pudel);
    assert(pudel_props.size() == 1);
    assert(approx(pudel_props[0].inherited_trust, 0.623295));
    assert(pudel_props[0].hop_count == 4);

    PASS();
}

// =============================================================================
// Test 3: Trust floor cutoff
// =============================================================================
//   Long chain where trust decays below floor
void test_trust_floor_cutoff() {
    TEST("Trust floor cutoff stops propagation");

    LongTermMemory ltm;
    auto top = add_cpt(ltm, "Top");
    auto prop = add_cpt(ltm, "SomeProp");

    // Property at top with low weight
    ltm.add_relation(top, prop, RelationType::HAS_PROPERTY, 0.40);

    // Build chain: c0 -> c1 -> c2 -> ... -> top
    // At 0.9 decay: 0.40 * 0.9 = 0.36 (hop1), 0.36 * 0.9 = 0.324 (hop2),
    //               0.324 * 0.9 = 0.2916 < 0.3 (hop3 blocked)
    ConceptId prev = top;
    std::vector<ConceptId> chain;
    for (int i = 0; i < 5; ++i) {
        auto c = add_cpt(ltm, "Chain_" + std::to_string(i));
        ltm.add_relation(c, prev, RelationType::IS_A, 1.0);
        chain.push_back(c);
        prev = c;
    }

    PropertyInheritance pi(ltm);
    PropertyInheritance::Config cfg;
    cfg.decay_per_hop = 0.9;
    cfg.trust_floor = 0.3;

    auto result = pi.propagate(cfg);

    // Chain_0 (1 hop): 0.40 * 0.9 = 0.36 >= 0.3 => inherited
    assert(pi.get_inherited(chain[0]).size() == 1);

    // Chain_1 (2 hops): 0.36 * 0.9 = 0.324 >= 0.3 => inherited
    assert(pi.get_inherited(chain[1]).size() == 1);

    // Chain_2 (3 hops): 0.324 * 0.9 = 0.2916 < 0.3 => NOT inherited
    assert(pi.get_inherited(chain[2]).size() == 0);

    // Chain_3 and beyond: also not inherited
    assert(pi.get_inherited(chain[3]).size() == 0);
    assert(pi.get_inherited(chain[4]).size() == 0);

    assert(result.trust_floor_cutoffs > 0);
    assert(result.converged);

    PASS();
}

// =============================================================================
// Test 4: CONTRADICTS blocks inheritance
// =============================================================================
//   Bird IS_A Animal
//   Penguin IS_A Bird
//   Animal HAS_PROPERTY can_fly
//   Penguin CONTRADICTS can_fly
//   => Bird inherits can_fly, Penguin does NOT
void test_contradicts_blocking() {
    TEST("CONTRADICTS blocks property inheritance");

    LongTermMemory ltm;
    auto animal  = add_cpt(ltm, "Animal");
    auto bird    = add_cpt(ltm, "Bird");
    auto penguin = add_cpt(ltm, "Penguin");
    auto can_fly = add_cpt(ltm, "can_fly");

    ltm.add_relation(bird, animal, RelationType::IS_A, 1.0);
    ltm.add_relation(penguin, bird, RelationType::IS_A, 1.0);
    ltm.add_relation(animal, can_fly, RelationType::HAS_PROPERTY, 0.90);

    // Penguin contradicts can_fly
    ltm.add_relation(penguin, can_fly, RelationType::CONTRADICTS, 1.0);

    PropertyInheritance pi(ltm);
    PropertyInheritance::Config cfg;
    cfg.decay_per_hop = 0.9;
    cfg.trust_floor = 0.3;

    auto result = pi.propagate(cfg);

    // Bird should inherit can_fly
    auto bird_props = pi.get_inherited(bird);
    assert(bird_props.size() == 1);
    assert(bird_props[0].property_target == can_fly);

    // Penguin should NOT inherit can_fly (contradicted)
    auto penguin_props = pi.get_inherited(penguin);
    assert(penguin_props.size() == 0);

    assert(result.contradictions_blocked > 0);

    PASS();
}

// =============================================================================
// Test 5: Diamond inheritance
// =============================================================================
//   D IS_A B, D IS_A C
//   B IS_A A, C IS_A A
//   A HAS_PROPERTY prop
//   => D should inherit prop (best trust wins)
void test_diamond_inheritance() {
    TEST("Diamond inheritance picks best trust path");

    LongTermMemory ltm;
    auto a    = add_cpt(ltm, "A");
    auto b    = add_cpt(ltm, "B");
    auto c    = add_cpt(ltm, "C");
    auto d    = add_cpt(ltm, "D");
    auto prop = add_cpt(ltm, "prop");

    // Diamond: D -> B -> A, D -> C -> A
    ltm.add_relation(b, a, RelationType::IS_A, 1.0);
    ltm.add_relation(c, a, RelationType::IS_A, 1.0);
    ltm.add_relation(d, b, RelationType::IS_A, 1.0);
    ltm.add_relation(d, c, RelationType::IS_A, 1.0);

    ltm.add_relation(a, prop, RelationType::HAS_PROPERTY, 0.95);

    PropertyInheritance pi(ltm);
    PropertyInheritance::Config cfg;
    cfg.decay_per_hop = 0.9;
    cfg.trust_floor = 0.3;

    auto result = pi.propagate(cfg);

    // D should have the property — inherited via either path
    auto d_props = pi.get_inherited(d);
    assert(d_props.size() == 1);
    assert(d_props[0].property_target == prop);

    // Both B and C should also have it
    assert(pi.get_inherited(b).size() == 1);
    assert(pi.get_inherited(c).size() == 1);

    assert(result.converged);

    PASS();
}

// =============================================================================
// Test 6: Multiple properties propagate independently
// =============================================================================
void test_multiple_properties() {
    TEST("Multiple properties propagate independently");

    LongTermMemory ltm;
    auto animal    = add_cpt(ltm, "Animal");
    auto mammal    = add_cpt(ltm, "Mammal");
    auto has_spine = add_cpt(ltm, "has_spine");
    auto breathes  = add_cpt(ltm, "breathes");
    auto eats      = add_cpt(ltm, "eats");

    ltm.add_relation(mammal, animal, RelationType::IS_A, 1.0);
    ltm.add_relation(animal, has_spine, RelationType::HAS_PROPERTY, 0.95);
    ltm.add_relation(animal, breathes, RelationType::HAS_PROPERTY, 0.90);
    ltm.add_relation(animal, eats, RelationType::HAS_PROPERTY, 0.85);

    PropertyInheritance pi(ltm);
    auto result = pi.propagate();

    auto props = pi.get_inherited(mammal);
    assert(props.size() == 3);
    assert(result.properties_inherited == 3);

    // Sorted by trust descending
    assert(approx(props[0].inherited_trust, 0.95 * 0.9));  // has_spine
    assert(approx(props[1].inherited_trust, 0.90 * 0.9));  // breathes
    assert(approx(props[2].inherited_trust, 0.85 * 0.9));  // eats

    PASS();
}

// =============================================================================
// Test 7: REQUIRES/USES/PRODUCES also propagate
// =============================================================================
void test_other_inheritable_types() {
    TEST("REQUIRES, USES, PRODUCES also propagate");

    LongTermMemory ltm;
    auto parent = add_cpt(ltm, "Machine");
    auto child  = add_cpt(ltm, "Robot");
    auto power  = add_cpt(ltm, "Electricity");
    auto oil    = add_cpt(ltm, "Oil");
    auto output = add_cpt(ltm, "Work");

    ltm.add_relation(child, parent, RelationType::IS_A, 1.0);
    ltm.add_relation(parent, power, RelationType::REQUIRES, 0.9);
    ltm.add_relation(parent, oil, RelationType::USES, 0.8);
    ltm.add_relation(parent, output, RelationType::PRODUCES, 0.85);

    PropertyInheritance pi(ltm);
    PropertyInheritance::Config cfg;
    cfg.propagate_requires = true;
    cfg.propagate_uses = true;
    cfg.propagate_produces = true;

    auto result = pi.propagate(cfg);

    // Robot should inherit all 3
    auto props = pi.get_inherited(child);
    assert(props.size() == 3);
    assert(result.properties_inherited == 3);

    PASS();
}

// =============================================================================
// Test 8: Disabling optional types
// =============================================================================
void test_disable_optional_types() {
    TEST("Disabling optional inheritable types");

    LongTermMemory ltm;
    auto parent = add_cpt(ltm, "Machine");
    auto child  = add_cpt(ltm, "Robot");
    auto power  = add_cpt(ltm, "Electricity");
    auto oil    = add_cpt(ltm, "Oil");

    ltm.add_relation(child, parent, RelationType::IS_A, 1.0);
    ltm.add_relation(parent, power, RelationType::REQUIRES, 0.9);
    ltm.add_relation(parent, oil, RelationType::HAS_PROPERTY, 0.8);

    PropertyInheritance pi(ltm);
    PropertyInheritance::Config cfg;
    cfg.propagate_requires = false;  // disable REQUIRES
    cfg.propagate_uses = false;
    cfg.propagate_produces = false;

    auto result = pi.propagate(cfg);

    // Only HAS_PROPERTY should propagate
    auto props = pi.get_inherited(child);
    assert(props.size() == 1);
    assert(props[0].property_target == oil);

    PASS();
}

// =============================================================================
// Test 9: Cycle safety (A IS_A B IS_A A)
// =============================================================================
void test_cycle_safety() {
    TEST("IS_A cycle does not cause infinite loop");

    LongTermMemory ltm;
    auto a    = add_cpt(ltm, "A");
    auto b    = add_cpt(ltm, "B");
    auto prop = add_cpt(ltm, "prop");

    ltm.add_relation(a, b, RelationType::IS_A, 1.0);
    ltm.add_relation(b, a, RelationType::IS_A, 1.0);  // cycle!
    ltm.add_relation(a, prop, RelationType::HAS_PROPERTY, 0.9);

    PropertyInheritance pi(ltm);
    PropertyInheritance::Config cfg;
    cfg.max_iterations = 10;

    auto result = pi.propagate(cfg);

    // Should converge (dedup prevents infinite growth)
    assert(result.converged);
    // Both should have the property
    assert(pi.get_inherited(b).size() >= 1);

    PASS();
}

// =============================================================================
// Test 10: Empty graph
// =============================================================================
void test_empty_graph() {
    TEST("Empty graph produces no inheritance");

    LongTermMemory ltm;

    PropertyInheritance pi(ltm);
    auto result = pi.propagate();

    assert(result.properties_inherited == 0);
    assert(result.converged);
    assert(result.iterations_run == 1);

    PASS();
}

// =============================================================================
// Test 11: No IS_A relations (properties don't propagate)
// =============================================================================
void test_no_isa_relations() {
    TEST("No IS_A relations means no inheritance");

    LongTermMemory ltm;
    auto a    = add_cpt(ltm, "A");
    auto b    = add_cpt(ltm, "B");
    auto prop = add_cpt(ltm, "prop");

    ltm.add_relation(a, b, RelationType::SIMILAR_TO, 1.0);  // not IS_A
    ltm.add_relation(b, prop, RelationType::HAS_PROPERTY, 0.9);

    PropertyInheritance pi(ltm);
    auto result = pi.propagate();

    assert(result.properties_inherited == 0);
    assert(pi.get_inherited(a).size() == 0);

    PASS();
}

// =============================================================================
// Test 12: CONTRADICTS is symmetric
// =============================================================================
//   If prop CONTRADICTS concept (incoming direction), also blocks
void test_contradicts_symmetric() {
    TEST("CONTRADICTS works in both directions");

    LongTermMemory ltm;
    auto parent  = add_cpt(ltm, "Parent");
    auto child   = add_cpt(ltm, "Child");
    auto prop    = add_cpt(ltm, "dangerous_prop");

    ltm.add_relation(child, parent, RelationType::IS_A, 1.0);
    ltm.add_relation(parent, prop, RelationType::HAS_PROPERTY, 0.9);

    // Incoming CONTRADICTS: prop contradicts child (reverse direction)
    ltm.add_relation(prop, child, RelationType::CONTRADICTS, 1.0);

    PropertyInheritance pi(ltm);
    auto result = pi.propagate();

    // Child should NOT inherit prop (blocked by incoming CONTRADICTS)
    assert(pi.get_inherited(child).size() == 0);
    assert(result.contradictions_blocked > 0);

    PASS();
}

// =============================================================================
// Test 13: Fixpoint converges in multiple iterations
// =============================================================================
void test_multi_iteration_convergence() {
    TEST("Fixpoint converges across multiple iterations");

    LongTermMemory ltm;
    auto a    = add_cpt(ltm, "A");
    auto b    = add_cpt(ltm, "B");
    auto c    = add_cpt(ltm, "C");
    auto d    = add_cpt(ltm, "D");
    auto prop = add_cpt(ltm, "prop");

    // Chain: D IS_A C IS_A B IS_A A
    ltm.add_relation(b, a, RelationType::IS_A, 1.0);
    ltm.add_relation(c, b, RelationType::IS_A, 1.0);
    ltm.add_relation(d, c, RelationType::IS_A, 1.0);
    ltm.add_relation(a, prop, RelationType::HAS_PROPERTY, 0.95);

    PropertyInheritance pi(ltm);
    PropertyInheritance::Config cfg;
    cfg.decay_per_hop = 0.9;
    cfg.trust_floor = 0.3;

    auto result = pi.propagate(cfg);

    // Iter 1: B inherits from A (direct)
    // Iter 2: C inherits from B (inherited)
    // Iter 3: D inherits from C (inherited)
    // Iter 4: no new -> converged
    assert(result.iterations_run >= 3);
    assert(result.converged);
    assert(result.properties_inherited == 3);

    // D should have it at 3 hops
    auto d_props = pi.get_inherited(d);
    assert(d_props.size() == 1);
    double expected = 0.95 * 0.9 * 0.9 * 0.9;  // 0.69255
    assert(approx(d_props[0].inherited_trust, expected));
    assert(d_props[0].hop_count == 3);

    PASS();
}

// =============================================================================
// Test 14: Properties at different levels
// =============================================================================
void test_properties_at_multiple_levels() {
    TEST("Properties defined at different hierarchy levels");

    LongTermMemory ltm;
    auto animal    = add_cpt(ltm, "Animal");
    auto mammal    = add_cpt(ltm, "Mammal");
    auto hund      = add_cpt(ltm, "Hund");
    auto spine     = add_cpt(ltm, "has_spine");
    auto warm_blood = add_cpt(ltm, "warm_blooded");

    ltm.add_relation(mammal, animal, RelationType::IS_A, 1.0);
    ltm.add_relation(hund, mammal, RelationType::IS_A, 1.0);

    // Property at Animal level
    ltm.add_relation(animal, spine, RelationType::HAS_PROPERTY, 0.95);
    // Property at Mammal level
    ltm.add_relation(mammal, warm_blood, RelationType::HAS_PROPERTY, 0.90);

    PropertyInheritance pi(ltm);
    PropertyInheritance::Config cfg;
    cfg.decay_per_hop = 0.9;
    cfg.trust_floor = 0.3;

    auto result = pi.propagate(cfg);

    // Hund should have both properties
    auto hund_props = pi.get_inherited(hund);
    assert(hund_props.size() == 2);

    // warm_blooded at 1 hop: 0.90 * 0.9 = 0.81
    // spine at 2 hops: 0.95 * 0.9 * 0.9 = 0.7695
    // Sorted by trust descending
    assert(approx(hund_props[0].inherited_trust, 0.81));    // warm_blooded
    assert(approx(hund_props[1].inherited_trust, 0.7695));  // spine

    PASS();
}

// =============================================================================
// Test 15: max_hop_depth respected
// =============================================================================
void test_max_hop_depth() {
    TEST("max_hop_depth limits propagation depth");

    LongTermMemory ltm;
    auto top  = add_cpt(ltm, "Top");
    auto prop = add_cpt(ltm, "prop");
    ltm.add_relation(top, prop, RelationType::HAS_PROPERTY, 0.99);

    // Build chain of 10 concepts
    ConceptId prev = top;
    std::vector<ConceptId> chain;
    for (int i = 0; i < 10; ++i) {
        auto c = add_cpt(ltm, "Level_" + std::to_string(i));
        ltm.add_relation(c, prev, RelationType::IS_A, 1.0);
        chain.push_back(c);
        prev = c;
    }

    PropertyInheritance pi(ltm);
    PropertyInheritance::Config cfg;
    cfg.decay_per_hop = 0.95;   // slow decay to avoid trust floor
    cfg.trust_floor = 0.1;     // low floor
    cfg.max_hop_depth = 3;      // but limit hops

    auto result = pi.propagate(cfg);

    // First 3 in chain should inherit (hops 1-3)
    assert(pi.get_inherited(chain[0]).size() == 1);  // 1 hop
    assert(pi.get_inherited(chain[1]).size() == 1);  // 2 hops
    assert(pi.get_inherited(chain[2]).size() == 1);  // 3 hops

    // 4th and beyond: blocked by max_hop_depth
    assert(pi.get_inherited(chain[3]).size() == 0);

    PASS();
}

// =============================================================================
// Test 16: Result statistics correct
// =============================================================================
void test_result_statistics() {
    TEST("Result statistics are correct");

    LongTermMemory ltm;
    auto animal  = add_cpt(ltm, "Animal");
    auto bird    = add_cpt(ltm, "Bird");
    auto penguin = add_cpt(ltm, "Penguin");
    auto fly     = add_cpt(ltm, "can_fly");
    auto walk    = add_cpt(ltm, "can_walk");

    ltm.add_relation(bird, animal, RelationType::IS_A, 1.0);
    ltm.add_relation(penguin, bird, RelationType::IS_A, 1.0);
    ltm.add_relation(animal, fly, RelationType::HAS_PROPERTY, 0.8);
    ltm.add_relation(animal, walk, RelationType::HAS_PROPERTY, 0.9);
    ltm.add_relation(penguin, fly, RelationType::CONTRADICTS, 1.0);

    PropertyInheritance pi(ltm);
    auto result = pi.propagate();

    // Bird gets fly + walk (2), Penguin gets walk only (1) = 3 total
    assert(result.properties_inherited == 3);
    assert(result.contradictions_blocked >= 1);  // penguin/fly
    assert(result.concepts_processed == 5);
    assert(result.converged);

    PASS();
}

// =============================================================================
// Test 17: Wide hierarchy (many siblings)
// =============================================================================
void test_wide_hierarchy() {
    TEST("Wide hierarchy: many children inherit from one parent");

    LongTermMemory ltm;
    auto parent = add_cpt(ltm, "Vehicle");
    auto prop1  = add_cpt(ltm, "has_wheels");
    auto prop2  = add_cpt(ltm, "has_engine");

    ltm.add_relation(parent, prop1, RelationType::HAS_PROPERTY, 0.9);
    ltm.add_relation(parent, prop2, RelationType::HAS_PROPERTY, 0.85);

    // 10 children
    std::vector<ConceptId> children;
    for (int i = 0; i < 10; ++i) {
        auto c = add_cpt(ltm, "Vehicle_" + std::to_string(i));
        ltm.add_relation(c, parent, RelationType::IS_A, 1.0);
        children.push_back(c);
    }

    PropertyInheritance pi(ltm);
    auto result = pi.propagate();

    // Each of 10 children gets 2 properties = 20
    assert(result.properties_inherited == 20);
    for (auto cid : children) {
        assert(pi.get_inherited(cid).size() == 2);
    }

    PASS();
}

// =============================================================================
// Main
// =============================================================================
int main() {
    std::cout << "\n=== PropertyInheritance Tests ===\n\n";

    test_basic_single_hop();
    test_pudel_chain();
    test_trust_floor_cutoff();
    test_contradicts_blocking();
    test_diamond_inheritance();
    test_multiple_properties();
    test_other_inheritable_types();
    test_disable_optional_types();
    test_cycle_safety();
    test_empty_graph();
    test_no_isa_relations();
    test_contradicts_symmetric();
    test_multi_iteration_convergence();
    test_properties_at_multiple_levels();
    test_max_hop_depth();
    test_result_statistics();
    test_wide_hierarchy();

    std::cout << "\n=== Results: " << tests_passed << "/" << tests_total
              << " passed ===\n";

    if (tests_passed == tests_total) {
        std::cout << "ALL TESTS PASSED\n\n";
        return 0;
    } else {
        std::cout << "SOME TESTS FAILED\n\n";
        return 1;
    }
}
