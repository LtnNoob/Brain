#include "relation_type_registry.hpp"
#include "../cursor/template_engine.hpp"
#include "../ltm/relation.hpp"
#include <cassert>
#include <iostream>
#include <set>

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

void test_original_10_types() {
    TEST("Original 10 types registered");
    auto& reg = RelationTypeRegistry::instance();

    assert(reg.has(RelationType::IS_A));
    assert(reg.has(RelationType::HAS_PROPERTY));
    assert(reg.has(RelationType::CAUSES));
    assert(reg.has(RelationType::ENABLES));
    assert(reg.has(RelationType::PART_OF));
    assert(reg.has(RelationType::SIMILAR_TO));
    assert(reg.has(RelationType::CONTRADICTS));
    assert(reg.has(RelationType::SUPPORTS));
    assert(reg.has(RelationType::TEMPORAL_BEFORE));
    assert(reg.has(RelationType::CUSTOM));
    PASS();
}

void test_new_builtin_types() {
    TEST("New built-in types (10-19) registered");
    auto& reg = RelationTypeRegistry::instance();

    assert(reg.has(RelationType::PRODUCES));
    assert(reg.has(RelationType::REQUIRES));
    assert(reg.has(RelationType::USES));
    assert(reg.has(RelationType::SOURCE));
    assert(reg.has(RelationType::HAS_PART));
    assert(reg.has(RelationType::TEMPORAL_AFTER));
    assert(reg.has(RelationType::INSTANCE_OF));
    assert(reg.has(RelationType::DERIVED_FROM));
    assert(reg.has(RelationType::IMPLIES));
    assert(reg.has(RelationType::ASSOCIATED_WITH));
    PASS();
}

void test_total_builtin_count() {
    TEST("Total 20 built-in types");
    auto& reg = RelationTypeRegistry::instance();
    auto builtins = reg.builtin_types();
    assert(builtins.size() == 20);
    PASS();
}

void test_german_names() {
    TEST("German names for original 10");
    auto& reg = RelationTypeRegistry::instance();

    assert(reg.get_name_de(RelationType::IS_A) == "ist ein(e)");
    assert(reg.get_name_de(RelationType::CAUSES) == "verursacht");
    assert(reg.get_name_de(RelationType::CUSTOM) == "steht in Beziehung zu");
    PASS();
}

void test_german_names_new() {
    TEST("German names for new types");
    auto& reg = RelationTypeRegistry::instance();

    assert(reg.get_name_de(RelationType::PRODUCES) == "erzeugt");
    assert(reg.get_name_de(RelationType::REQUIRES) == "benoetigt");
    assert(reg.get_name_de(RelationType::USES) == "verwendet");
    PASS();
}

void test_slugs() {
    TEST("Slug strings");
    auto& reg = RelationTypeRegistry::instance();

    assert(reg.get_slug(RelationType::IS_A) == "is-a");
    assert(reg.get_slug(RelationType::HAS_PROPERTY) == "has-property");
    assert(reg.get_slug(RelationType::PRODUCES) == "produces");
    PASS();
}

void test_categories() {
    TEST("Categories correct");
    auto& reg = RelationTypeRegistry::instance();

    assert(reg.get_category(RelationType::IS_A) == RelationCategory::HIERARCHICAL);
    assert(reg.get_category(RelationType::CAUSES) == RelationCategory::CAUSAL);
    assert(reg.get_category(RelationType::PART_OF) == RelationCategory::COMPOSITIONAL);
    assert(reg.get_category(RelationType::SIMILAR_TO) == RelationCategory::SIMILARITY);
    assert(reg.get_category(RelationType::CONTRADICTS) == RelationCategory::OPPOSITION);
    assert(reg.get_category(RelationType::SUPPORTS) == RelationCategory::EPISTEMIC);
    assert(reg.get_category(RelationType::TEMPORAL_BEFORE) == RelationCategory::TEMPORAL);
    assert(reg.get_category(RelationType::USES) == RelationCategory::FUNCTIONAL);
    assert(reg.get_category(RelationType::CUSTOM) == RelationCategory::CUSTOM_CATEGORY);
    PASS();
}

void test_embeddings_nonzero() {
    TEST("All embeddings non-zero");
    auto& reg = RelationTypeRegistry::instance();

    for (auto type : reg.all_types()) {
        const Vec10& emb = reg.get_embedding(type);
        double sum = 0.0;
        for (double v : emb) sum += std::abs(v);
        assert(sum > 0.0);
    }
    PASS();
}

void test_find_by_name() {
    TEST("find_by_name round-trips");
    auto& reg = RelationTypeRegistry::instance();

    auto found = reg.find_by_name("IS_A");
    assert(found.has_value());
    assert(*found == RelationType::IS_A);

    auto found2 = reg.find_by_name("PRODUCES");
    assert(found2.has_value());
    assert(*found2 == RelationType::PRODUCES);

    auto not_found = reg.find_by_name("DOES_NOT_EXIST");
    assert(!not_found.has_value());
    PASS();
}

void test_runtime_registration() {
    TEST("Runtime type registration");
    auto& reg = RelationTypeRegistry::instance();

    Vec10 emb = {0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5};
    auto new_type = reg.register_type("EXPLAINS", "erklaert", RelationCategory::CAUSAL, emb);

    assert(static_cast<uint16_t>(new_type) >= 1000);
    assert(reg.has(new_type));
    assert(reg.get_name(new_type) == "EXPLAINS");
    assert(reg.get_name_de(new_type) == "erklaert");
    assert(reg.get_category(new_type) == RelationCategory::CAUSAL);

    // Duplicate name returns same type
    auto same = reg.register_type("EXPLAINS", "erklaert", RelationCategory::CAUSAL, emb);
    assert(same == new_type);
    PASS();
}

void test_unknown_type_fallback() {
    TEST("Unknown type returns fallback");
    auto& reg = RelationTypeRegistry::instance();

    // Type 999 is not registered
    auto type999 = static_cast<RelationType>(999);
    assert(!reg.has(type999));
    // Should return fallback info
    const auto& info = reg.get(type999);
    assert(info.name == "UNKNOWN");
    PASS();
}

void test_template_engine_uses_registry() {
    TEST("TemplateEngine::relation_name_de uses registry");

    // Original types
    assert(TemplateEngine::relation_name_de(RelationType::IS_A) == "ist ein(e)");
    assert(TemplateEngine::relation_name_de(RelationType::CAUSES) == "verursacht");

    // New built-in types
    assert(TemplateEngine::relation_name_de(RelationType::PRODUCES) == "erzeugt");
    assert(TemplateEngine::relation_name_de(RelationType::REQUIRES) == "benoetigt");
    PASS();
}

void test_relation_type_to_string() {
    TEST("relation_type_to_string uses registry");

    assert(std::string(relation_type_to_string(RelationType::IS_A)) == "IS_A");
    assert(std::string(relation_type_to_string(RelationType::PRODUCES)) == "PRODUCES");
    PASS();
}

void test_enum_values_stable() {
    TEST("Enum values unchanged for backward compat");

    assert(static_cast<uint16_t>(RelationType::IS_A) == 0);
    assert(static_cast<uint16_t>(RelationType::HAS_PROPERTY) == 1);
    assert(static_cast<uint16_t>(RelationType::CAUSES) == 2);
    assert(static_cast<uint16_t>(RelationType::ENABLES) == 3);
    assert(static_cast<uint16_t>(RelationType::PART_OF) == 4);
    assert(static_cast<uint16_t>(RelationType::SIMILAR_TO) == 5);
    assert(static_cast<uint16_t>(RelationType::CONTRADICTS) == 6);
    assert(static_cast<uint16_t>(RelationType::SUPPORTS) == 7);
    assert(static_cast<uint16_t>(RelationType::TEMPORAL_BEFORE) == 8);
    assert(static_cast<uint16_t>(RelationType::CUSTOM) == 9);
    assert(static_cast<uint16_t>(RelationType::PRODUCES) == 10);
    assert(static_cast<uint16_t>(RelationType::RUNTIME_BASE) == 1000);
    PASS();
}

int main() {
    std::cout << "=== RelationTypeRegistry Tests ===" << std::endl;

    test_original_10_types();
    test_new_builtin_types();
    test_total_builtin_count();
    test_german_names();
    test_german_names_new();
    test_slugs();
    test_categories();
    test_embeddings_nonzero();
    test_find_by_name();
    test_runtime_registration();
    test_unknown_type_fallback();
    test_template_engine_uses_registry();
    test_relation_type_to_string();
    test_enum_values_stable();

    std::cout << "\n=== " << tests_passed << "/" << tests_total << " PASSED ===" << std::endl;
    return (tests_passed == tests_total) ? 0 : 1;
}
