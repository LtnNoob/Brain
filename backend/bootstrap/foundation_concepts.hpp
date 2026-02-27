#pragma once

#include "../ltm/long_term_memory.hpp"
#include <cstddef>

namespace brain19 {

// FoundationConcepts: Pre-seeded core knowledge for bootstrapping Brain19
//
// Solves the "chicken-and-egg" problem: an empty LTM has no context for
// meaningful human review. Foundation concepts provide a minimal ontological
// scaffold so that new knowledge can be typed and related from the start.
//
// All seeded concepts have:
//   - EpistemicType::DEFINITION or FACT
//   - EpistemicStatus::ACTIVE
//   - Trust 0.95-0.99
//   - Source: "bootstrap:foundation_v1"
//
// Tiers:
//   1. Meta-Ontology (~50 concepts)   — Entity, Object, Action, ...
//   2. Basic Categories (~100 concepts) — Person, Place, Science, ...
//   3. Common Relations (~200 relations) — IS_A, HAS_PROPERTY, CAUSES
//   4. Science Foundation (~150 concepts) — Atom, Cell, Number, ...
class FoundationConcepts {
public:
    // Seed all tiers in order
    static void seed_all(LongTermMemory& ltm);

    // Individual tier seeding
    static void seed_tier1_ontology(LongTermMemory& ltm);
    static void seed_tier2_categories(LongTermMemory& ltm);
    static void seed_tier3_relations(LongTermMemory& ltm);
    static void seed_tier4_science(LongTermMemory& ltm);

    // Seed from JSON file (returns true on success, false = fallback to hardcoded)
    // include_weak_relations: retain RELATES_TO (at 0.15x) and low-weight (at 0.5x)
    static bool seed_from_file(LongTermMemory& ltm, const std::string& path,
                               bool include_weak_relations = false);

    // Counts (available before seeding — these are compile-time constants)
    static size_t concept_count();
    static size_t relation_count();
};

} // namespace brain19
