#include "json_parser.hpp"
#include "foundation_concepts.hpp"
#include "../ltm/long_term_memory.hpp"
#include <cassert>
#include <iostream>
#include <fstream>

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

void test_parse_string() {
    TEST("Parse string");
    auto v = JsonParser::parse("\"hello world\"");
    assert(v.has_value());
    assert(v->is_string());
    assert(v->as_string() == "hello world");
    PASS();
}

void test_parse_number() {
    TEST("Parse number");
    auto v = JsonParser::parse("42.5");
    assert(v.has_value());
    assert(v->is_number());
    assert(v->as_number() == 42.5);
    PASS();
}

void test_parse_negative_number() {
    TEST("Parse negative number");
    auto v = JsonParser::parse("-3.14");
    assert(v.has_value());
    assert(v->is_number());
    assert(v->as_number() < -3.13 && v->as_number() > -3.15);
    PASS();
}

void test_parse_bool() {
    TEST("Parse booleans");
    auto t = JsonParser::parse("true");
    assert(t.has_value() && t->is_bool() && t->as_bool() == true);
    auto f = JsonParser::parse("false");
    assert(f.has_value() && f->is_bool() && f->as_bool() == false);
    PASS();
}

void test_parse_null() {
    TEST("Parse null");
    auto v = JsonParser::parse("null");
    assert(v.has_value() && v->is_null());
    PASS();
}

void test_parse_empty_object() {
    TEST("Parse empty object");
    auto v = JsonParser::parse("{}");
    assert(v.has_value() && v->is_object());
    assert(v->as_object().empty());
    PASS();
}

void test_parse_object() {
    TEST("Parse object with fields");
    auto v = JsonParser::parse("{\"name\":\"test\",\"value\":42}");
    assert(v.has_value() && v->is_object());
    auto name = v->get("name");
    assert(name && name->is_string() && name->as_string() == "test");
    auto val = v->get("value");
    assert(val && val->is_number() && val->as_number() == 42.0);
    PASS();
}

void test_parse_array() {
    TEST("Parse array");
    auto v = JsonParser::parse("[1, 2, 3]");
    assert(v.has_value() && v->is_array());
    assert(v->as_array().size() == 3);
    assert(v->as_array()[0].as_number() == 1.0);
    assert(v->as_array()[2].as_number() == 3.0);
    PASS();
}

void test_parse_nested() {
    TEST("Parse nested structure");
    auto v = JsonParser::parse("{\"arr\":[{\"x\":1},{\"x\":2}]}");
    assert(v.has_value() && v->is_object());
    auto arr = v->get("arr");
    assert(arr && arr->is_array());
    assert(arr->as_array().size() == 2);
    auto x1 = arr->as_array()[0].get("x");
    assert(x1 && x1->as_number() == 1.0);
    PASS();
}

void test_escape_sequences() {
    TEST("Parse escape sequences");
    auto v = JsonParser::parse("\"line1\\nline2\\ttab\"");
    assert(v.has_value() && v->is_string());
    assert(v->as_string() == "line1\nline2\ttab");
    PASS();
}

void test_reject_trailing_garbage() {
    TEST("Reject trailing garbage");
    auto v = JsonParser::parse("42 garbage");
    assert(!v.has_value());
    PASS();
}

void test_reject_invalid() {
    TEST("Reject invalid JSON");
    assert(!JsonParser::parse("{bad}").has_value());
    assert(!JsonParser::parse("").has_value());
    assert(!JsonParser::parse("[1,]").has_value());
    PASS();
}

void test_foundation_json_parseable() {
    TEST("Parse foundation.json");
    auto v = JsonParser::parse_file("../data/foundation.json");
    assert(v.has_value());
    assert(v->is_object());

    auto concepts = v->get("concepts");
    assert(concepts && concepts->is_array());
    std::cout << "(" << concepts->as_array().size() << " concepts) ";
    assert(concepts->as_array().size() == FoundationConcepts::concept_count());

    auto relations = v->get("relations");
    assert(relations && relations->is_array());
    std::cout << "(" << relations->as_array().size() << " relations) ";
    assert(relations->as_array().size() == FoundationConcepts::relation_count());

    // Check first concept has all fields
    auto& first = concepts->as_array()[0];
    assert(first.get("label") && first.get("label")->is_string());
    assert(first.get("definition") && first.get("definition")->is_string());
    assert(first.get("epistemic_type") && first.get("epistemic_type")->is_string());
    assert(first.get("trust") && first.get("trust")->is_number());
    assert(first.get("label")->as_string() == "Entity");

    // Check first relation
    auto& first_rel = relations->as_array()[0];
    assert(first_rel.get("source") && first_rel.get("source")->is_string());
    assert(first_rel.get("target") && first_rel.get("target")->is_string());
    assert(first_rel.get("type") && first_rel.get("type")->is_string());
    assert(first_rel.get("weight") && first_rel.get("weight")->is_number());
    PASS();
}

void test_seed_from_file() {
    TEST("seed_from_file loads into LTM");
    LongTermMemory ltm;
    bool ok = FoundationConcepts::seed_from_file(ltm, "../data/foundation.json");
    assert(ok);

    auto ids = ltm.get_all_concept_ids();
    std::cout << "(" << ids.size() << " concepts) ";
    assert(ids.size() == FoundationConcepts::concept_count());

    // Verify a known concept
    bool found_entity = false;
    for (auto cid : ids) {
        auto info = ltm.retrieve_concept(cid);
        if (info && info->label == "Entity") {
            found_entity = true;
            assert(info->definition == "The most general category; anything that exists or can be conceived.");
            break;
        }
    }
    assert(found_entity);

    // Check relations exist
    assert(ltm.total_relation_count() == FoundationConcepts::relation_count());
    PASS();
}

void test_seed_from_file_matches_hardcoded() {
    TEST("seed_from_file matches hardcoded counts");
    // Cross-check: file and hardcoded should produce same counts
    LongTermMemory ltm_file, ltm_hard;
    FoundationConcepts::seed_from_file(ltm_file, "../data/foundation.json");
    FoundationConcepts::seed_all(ltm_hard);
    assert(ltm_file.get_all_concept_ids().size() == ltm_hard.get_all_concept_ids().size());
    assert(ltm_file.total_relation_count() == ltm_hard.total_relation_count());
    PASS();
}

void test_seed_from_file_nonexistent() {
    TEST("seed_from_file returns false for missing file");
    LongTermMemory ltm;
    bool ok = FoundationConcepts::seed_from_file(ltm, "/nonexistent/path.json");
    assert(!ok);
    assert(ltm.get_all_concept_ids().empty());
    PASS();
}

int main() {
    std::cout << "=== JSON Parser & Foundation File Tests ===" << std::endl;

    // JSON parser tests
    test_parse_string();
    test_parse_number();
    test_parse_negative_number();
    test_parse_bool();
    test_parse_null();
    test_parse_empty_object();
    test_parse_object();
    test_parse_array();
    test_parse_nested();
    test_escape_sequences();
    test_reject_trailing_garbage();
    test_reject_invalid();

    // Foundation file tests
    test_foundation_json_parseable();
    test_seed_from_file();
    test_seed_from_file_matches_hardcoded();
    test_seed_from_file_nonexistent();

    std::cout << "\n=== " << tests_passed << "/" << tests_total << " PASSED ===" << std::endl;
    return (tests_passed == tests_total) ? 0 : 1;
}
