#pragma once

#include "../importers/knowledge_proposal.hpp"
#include "entity_extractor.hpp"
#include "relation_extractor.hpp"
#include "trust_tagger.hpp"
#include "proposal_queue.hpp"
#include <string>
#include <vector>
#include <memory>

namespace brain19 {

// StructuredConcept: A concept parsed from structured input (JSON/CSV)
struct StructuredConcept {
    std::string label;
    std::string definition;
    std::string trust_category;     // "FACT", "THEORY", "SPECULATION", etc.
    double trust_value;             // Optional explicit trust (0.0 = use default)
    std::vector<std::string> tags;  // Optional categorization tags

    StructuredConcept()
        : trust_value(0.0) {}
};

// StructuredRelation: A relation parsed from structured input
struct StructuredRelation {
    std::string source_label;
    std::string target_label;
    std::string relation_type;      // "is-a", "causes", "part-of", etc.
    double weight;                  // [0.0, 1.0]

    StructuredRelation()
        : weight(1.0) {}
};

// StructuredInput: Complete structured data package
struct StructuredInput {
    std::vector<StructuredConcept> concepts;
    std::vector<StructuredRelation> relations;
    std::string source_reference;
    std::string import_notes;
};

// ParseResult: Result of parsing structured input
struct ParseResult {
    bool success;
    std::string error_message;
    StructuredInput data;
    size_t concepts_parsed;
    size_t relations_parsed;

    static ParseResult ok(const StructuredInput& input) {
        ParseResult r;
        r.success = true;
        r.data = input;
        r.concepts_parsed = input.concepts.size();
        r.relations_parsed = input.relations.size();
        return r;
    }

    static ParseResult error(const std::string& msg) {
        ParseResult r;
        r.success = false;
        r.error_message = msg;
        r.concepts_parsed = 0;
        r.relations_parsed = 0;
        return r;
    }

private:
    ParseResult() : success(false), concepts_parsed(0), relations_parsed(0) {}
};

// KnowledgeIngestor: Parses structured input formats (JSON, CSV)
//
// DESIGN:
// - Parses JSON and CSV into StructuredInput
// - Maps trust category strings to TrustCategory enum
// - Maps relation type strings to RelationType enum
// - No external JSON library - uses minimal hand-written parser
// - Produces IngestProposals for the ProposalQueue
//
// JSON FORMAT:
// {
//   "source": "reference string",
//   "concepts": [
//     { "label": "Cat", "definition": "A mammal", "trust": "FACT", "trust_value": 0.98 }
//   ],
//   "relations": [
//     { "source": "Cat", "target": "Mammal", "type": "is-a", "weight": 0.9 }
//   ]
// }
//
// CSV FORMAT (concepts):
//   label,definition,trust_category,trust_value
//   Cat,"A small domesticated mammal",FACT,0.98
//
// CSV FORMAT (relations):
//   source,target,type,weight
//   Cat,Mammal,is-a,0.9
class KnowledgeIngestor {
public:
    KnowledgeIngestor();

    // Parse JSON string
    ParseResult parse_json(const std::string& json_str) const;

    // Parse CSV string (concepts)
    ParseResult parse_csv_concepts(const std::string& csv_str) const;

    // Parse CSV string (relations)
    ParseResult parse_csv_relations(const std::string& csv_str,
                                    StructuredInput& existing) const;

    // Convert structured input to IngestProposals
    std::vector<IngestProposal> to_proposals(
        const StructuredInput& input,
        const TrustTagger& tagger) const;

    // Map string to TrustCategory
    static TrustCategory parse_trust_category(const std::string& str);

    // Map string to RelationType
    static RelationType parse_relation_type(const std::string& str);

private:
    // Minimal JSON parsing (no external library)
    std::string json_extract_string(const std::string& json, const std::string& key) const;
    std::string json_extract_array(const std::string& json, const std::string& key) const;
    std::vector<std::string> json_split_array(const std::string& array_str) const;
    double json_extract_number(const std::string& json, const std::string& key, double default_val) const;

    // CSV parsing
    std::vector<std::string> csv_split_line(const std::string& line) const;
    std::vector<std::string> split_lines(const std::string& text) const;
    std::string unquote(const std::string& s) const;
};

} // namespace brain19
