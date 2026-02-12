// Unit Test: Template Engine
// Tests RelationType → German sentence pattern mapping
//
// Build:
//   make test_template_engine

#include "template_engine.hpp"
#include "../ltm/long_term_memory.hpp"
#include "../epistemic/epistemic_metadata.hpp"
#include <cassert>
#include <iostream>
#include <string>

using namespace brain19;

// =============================================================================
// Helper: Build a mini knowledge graph
// =============================================================================
struct MiniGraph {
    LongTermMemory ltm;
    ConceptId eis, schmelzen, wasser, fluessig, dampf, eis_typ;

    MiniGraph() {
        EpistemicMetadata meta(EpistemicType::FACT, EpistemicStatus::ACTIVE, 0.9);

        eis       = ltm.store_concept("Eis", "Gefrorenes Wasser", meta);
        schmelzen = ltm.store_concept("Schmelzen", "Phasenuebergang fest zu fluessig", meta);
        wasser    = ltm.store_concept("Wasser", "H2O in fluessiger Form", meta);
        fluessig  = ltm.store_concept("Fluessig", "Aggregatzustand fluessig", meta);
        dampf     = ltm.store_concept("Dampf", "H2O in gasfoermiger Form", meta);
        eis_typ   = ltm.store_concept("Festkoerper", "Aggregatzustand fest", meta);

        ltm.add_relation(eis, schmelzen, RelationType::CAUSES, 0.9);
        ltm.add_relation(schmelzen, wasser, RelationType::CAUSES, 0.85);
        ltm.add_relation(wasser, fluessig, RelationType::HAS_PROPERTY, 0.8);
        ltm.add_relation(wasser, dampf, RelationType::CAUSES, 0.7);
        ltm.add_relation(eis, eis_typ, RelationType::IS_A, 0.95);
    }
};

// =============================================================================
// Test 1: Single relation sentence — CAUSES
// =============================================================================
void test_single_causes() {
    std::cout << "TEST: Single CAUSES sentence... ";

    MiniGraph g;
    TemplateEngine te(g.ltm);

    std::string s = te.relation_sentence("Eis", "Schmelzen", RelationType::CAUSES);
    assert(s == "Eis verursacht Schmelzen.");

    std::cout << "PASS (\"" << s << "\")\n";
}

// =============================================================================
// Test 2: Single relation sentence — all types
// =============================================================================
void test_all_relation_types() {
    std::cout << "TEST: All relation types... ";

    MiniGraph g;
    TemplateEngine te(g.ltm);

    // Test each type produces non-empty, correct output
    auto check = [&](RelationType t, const std::string& expected_verb) {
        std::string s = te.relation_sentence("A", "B", t);
        assert(s.find(expected_verb) != std::string::npos);
    };

    check(RelationType::IS_A,            "ist ein(e)");
    check(RelationType::HAS_PROPERTY,    "hat die Eigenschaft");
    check(RelationType::CAUSES,          "verursacht");
    check(RelationType::ENABLES,         "ermoeglicht");
    check(RelationType::PART_OF,         "ist Teil von");
    check(RelationType::SIMILAR_TO,      "ist aehnlich wie");
    check(RelationType::CONTRADICTS,     "widerspricht");
    check(RelationType::SUPPORTS,        "unterstuetzt");
    check(RelationType::TEMPORAL_BEFORE, "geschieht vor");
    check(RelationType::CUSTOM,          "steht in Beziehung zu");

    std::cout << "PASS\n";
}

// =============================================================================
// Test 3: Full chain generation — causal chain
// =============================================================================
void test_chain_generation() {
    std::cout << "TEST: Chain generation... ";

    MiniGraph g;
    TemplateEngine te(g.ltm);

    // Eis →CAUSES→ Schmelzen →CAUSES→ Wasser →HAS_PROPERTY→ Fluessig
    std::vector<ConceptId> concepts = {g.eis, g.schmelzen, g.wasser, g.fluessig};
    std::vector<RelationType> relations = {
        RelationType::CAUSES, RelationType::CAUSES, RelationType::HAS_PROPERTY
    };

    auto result = te.generate(concepts, relations);

    assert(result.sentences_generated == 3);
    assert(result.text.find("Eis verursacht Schmelzen.") != std::string::npos);
    assert(result.text.find("Schmelzen verursacht Wasser.") != std::string::npos);
    assert(result.text.find("Wasser hat die Eigenschaft Fluessig.") != std::string::npos);

    std::cout << "PASS\n  \"" << result.text << "\"\n";
}

// =============================================================================
// Test 4: Template type classification
// =============================================================================
void test_template_classification() {
    std::cout << "TEST: Template classification... ";

    MiniGraph g;
    TemplateEngine te(g.ltm);

    // Causal chain
    std::vector<RelationType> causal = {RelationType::CAUSES, RelationType::CAUSES};
    assert(te.classify(causal) == TemplateType::KAUSAL_ERKLAEREND);

    // Definitional chain
    std::vector<RelationType> defn = {RelationType::IS_A, RelationType::HAS_PROPERTY};
    assert(te.classify(defn) == TemplateType::DEFINITIONAL);

    // Comparative
    std::vector<RelationType> compare = {RelationType::SIMILAR_TO, RelationType::CONTRADICTS};
    assert(te.classify(compare) == TemplateType::VERGLEICHEND);

    // Mixed — CAUSES dominates
    std::vector<RelationType> mixed = {
        RelationType::CAUSES, RelationType::HAS_PROPERTY, RelationType::ENABLES
    };
    assert(te.classify(mixed) == TemplateType::KAUSAL_ERKLAEREND);

    // Empty
    std::vector<RelationType> empty;
    assert(te.classify(empty) == TemplateType::DEFINITIONAL);

    std::cout << "PASS\n";
}

// =============================================================================
// Test 5: TraversalResult generation
// =============================================================================
void test_traversal_result_generation() {
    std::cout << "TEST: TraversalResult generation... ";

    MiniGraph g;
    TemplateEngine te(g.ltm);

    TraversalResult tr;
    tr.concept_sequence = {g.eis, g.schmelzen, g.wasser};
    tr.relation_sequence = {RelationType::CAUSES, RelationType::CAUSES};
    tr.chain_score = 0.8;
    tr.total_steps = 3;

    auto result = te.generate(tr);

    assert(result.sentences_generated == 2);
    assert(result.template_type == TemplateType::KAUSAL_ERKLAEREND);
    assert(result.text.find("Eis verursacht Schmelzen.") != std::string::npos);
    assert(result.text.find("Schmelzen verursacht Wasser.") != std::string::npos);

    std::cout << "PASS\n  \"" << result.text << "\"\n";
}

// =============================================================================
// Test 6: Single concept (no relations)
// =============================================================================
void test_single_concept() {
    std::cout << "TEST: Single concept... ";

    MiniGraph g;
    TemplateEngine te(g.ltm);

    auto result = te.generate({g.eis}, {});
    assert(result.sentences_generated == 1);
    assert(result.text.find("Eis") != std::string::npos);
    assert(result.text.find("Gefrorenes Wasser") != std::string::npos);

    std::cout << "PASS\n  \"" << result.text << "\"\n";
}

// =============================================================================
// Test 7: Empty input
// =============================================================================
void test_empty_input() {
    std::cout << "TEST: Empty input... ";

    MiniGraph g;
    TemplateEngine te(g.ltm);

    auto result = te.generate({}, {});
    assert(result.sentences_generated == 0);
    assert(result.text.empty());

    std::cout << "PASS\n";
}

// =============================================================================
// Test 8: IS_A chain → definitional output
// =============================================================================
void test_definitional_chain() {
    std::cout << "TEST: Definitional chain... ";

    MiniGraph g;
    TemplateEngine te(g.ltm);

    std::vector<ConceptId> concepts = {g.eis, g.eis_typ};
    std::vector<RelationType> relations = {RelationType::IS_A};

    auto result = te.generate(concepts, relations);
    assert(result.sentences_generated == 1);
    assert(result.template_type == TemplateType::DEFINITIONAL);
    assert(result.text == "Eis ist ein(e) Festkoerper.");

    std::cout << "PASS\n  \"" << result.text << "\"\n";
}

// =============================================================================
// Test 9: relation_name_de static function
// =============================================================================
void test_relation_name_de() {
    std::cout << "TEST: relation_name_de... ";

    assert(TemplateEngine::relation_name_de(RelationType::CAUSES) == "verursacht");
    assert(TemplateEngine::relation_name_de(RelationType::IS_A) == "ist ein(e)");
    assert(TemplateEngine::relation_name_de(RelationType::CONTRADICTS) == "widerspricht");

    std::cout << "PASS\n";
}

// =============================================================================
// Main
// =============================================================================
int main() {
    std::cout << "\n=== Template Engine Unit Tests ===\n\n";

    test_single_causes();
    test_all_relation_types();
    test_chain_generation();
    test_template_classification();
    test_traversal_result_generation();
    test_single_concept();
    test_empty_input();
    test_definitional_chain();
    test_relation_name_de();

    std::cout << "\n=== ALL TESTS PASSED ===\n\n";
    return 0;
}
