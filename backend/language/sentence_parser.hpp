#pragma once

#include "../common/types.hpp"
#include "../ltm/long_term_memory.hpp"
#include "../evolution/graph_densifier.hpp"
#include <optional>
#include <string>
#include <vector>

namespace brain19 {

// POS tag enum — replaces string comparisons with compile-time safety
enum class POSTag : uint8_t {
    UNKNOWN = 0,
    NOUN,
    VERB,
    ADJ,
    ADV,
    DET,
    PREP,
    CONJ,
};

inline const char* pos_tag_str(POSTag tag) {
    switch (tag) {
        case POSTag::UNKNOWN: return "UNKNOWN";
        case POSTag::NOUN:    return "NOUN";
        case POSTag::VERB:    return "VERB";
        case POSTag::ADJ:     return "ADJ";
        case POSTag::ADV:     return "ADV";
        case POSTag::DET:     return "DET";
        case POSTag::PREP:    return "PREP";
        case POSTag::CONJ:    return "CONJ";
    }
    return "UNKNOWN";
}

struct Token {
    std::string surface;  // lowercased word
    POSTag pos = POSTag::UNKNOWN;
    int position;         // position index in sentence

    // Probabilistic POS (iterative two-level system)
    double p_noun = 0.0;
    double p_verb = 0.0;
    double p_adj  = 0.0;
    double p_adv  = 0.0;
    double p_func = 0.0;  // DET/PREP/CONJ
};

struct ParsedSentence {
    ConceptId sentence_id;
    ConceptId subject_word = 0;   // 0 if absent
    ConceptId verb_word = 0;
    ConceptId object_word = 0;
    std::vector<ConceptId> modifiers;
    std::optional<ConceptId> subject_semantic;
    std::optional<ConceptId> verb_semantic;
    std::optional<ConceptId> object_semantic;
};

// Sentence-level structure hypothesis
enum class SentencePattern { SVO, SOV, VS, QUESTION, UNKNOWN };

class SentenceParser {
public:
    SentenceParser(LongTermMemory& ltm, GraphDensifier& densifier);

    // Parse a single sentence: create word + sentence concepts, link with relations
    ParsedSentence parse_and_store(const std::string& sentence);

    // Parse multiple sentences with PRECEDES chain
    std::vector<ParsedSentence> parse_discourse(const std::vector<std::string>& sentences);

    // Lookup existing word concepts by surface form
    std::vector<ConceptId> lookup_word(const std::string& surface_form) const;

    // Manually link a word concept to a semantic concept via DENOTES
    void link_word_to_concept(ConceptId word, ConceptId semantic);

private:
    LongTermMemory& ltm_;
    GraphDensifier& densifier_;

    // --- Level 1: Word-level POS probabilities ---
    std::vector<Token> tokenize(const std::string& sentence) const;
    void compute_word_probs(std::vector<Token>& tokens) const;
    POSTag infer_pos_from_graph(ConceptId cid) const;

    // --- Level 2: Sentence-level structure estimation ---
    SentencePattern estimate_structure(const std::vector<Token>& tokens) const;
    void update_word_probs_from_structure(std::vector<Token>& tokens,
                                          SentencePattern pattern) const;

    // --- Iterative convergence ---
    void resolve_pos_iterative(std::vector<Token>& tokens) const;

    // --- Concept management ---
    ConceptId get_or_create_word_concept(const Token& token);
    std::optional<ConceptId> find_semantic_concept(const std::string& surface) const;
};

} // namespace brain19
