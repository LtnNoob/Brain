#include "sentence_parser.hpp"
#include "../memory/active_relation.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <unordered_set>

namespace brain19 {

// =============================================================================
// Construction
// =============================================================================

SentenceParser::SentenceParser(LongTermMemory& ltm, GraphDensifier& densifier)
    : ltm_(ltm), densifier_(densifier)
{}

// =============================================================================
// Tokenizer: extract words, split on non-alpha
// =============================================================================

std::vector<Token> SentenceParser::tokenize(const std::string& sentence) const {
    std::vector<Token> tokens;
    std::string current;
    int pos = 0;

    for (size_t i = 0; i <= sentence.size(); ++i) {
        unsigned char c = (i < sentence.size()) ? static_cast<unsigned char>(sentence[i]) : ' ';
        // Accept ASCII alpha and high bytes (UTF-8 multibyte for umlauts)
        if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c >= 0x80) {
            current += static_cast<char>(c);
        } else {
            if (!current.empty()) {
                // Lowercase ASCII portion
                std::string lower;
                lower.reserve(current.size());
                for (unsigned char ch : current) {
                    if (ch >= 'A' && ch <= 'Z')
                        lower += static_cast<char>(ch + 32);
                    else
                        lower += static_cast<char>(ch);
                }
                Token tok;
                tok.surface = lower;
                tok.position = pos;
                tok.pos = POSTag::UNKNOWN;
                tokens.push_back(std::move(tok));
                ++pos;
                current.clear();
            }
        }
    }

    return tokens;
}

// =============================================================================
// Level 1: Word-level POS probabilities
// =============================================================================
// Based on: lexicon membership, suffix heuristics, graph relations

void SentenceParser::compute_word_probs(std::vector<Token>& tokens) const {
    // Known function words
    static const std::unordered_set<std::string> determiners = {
        "der", "die", "das", "dem", "den", "des", "ein", "eine", "einem",
        "einen", "einer", "the", "a", "an"
    };
    static const std::unordered_set<std::string> prepositions = {
        "in", "auf", "mit", "von", "zu", "bei", "nach", "unter",
        "to", "from", "with", "at", "by", "for", "of"
    };
    static const std::unordered_set<std::string> conjunctions = {
        "und", "oder", "aber", "denn", "weil", "dass",
        "and", "or", "but", "because"
    };
    static const std::unordered_set<std::string> known_verbs = {
        "ist", "sind", "hat", "haben", "wird", "werden", "war", "waren",
        "hatte", "hatten", "kann", "konnte",
        "is", "are", "has", "have", "was", "were", "do", "does", "did",
        "will", "would", "could", "should", "can", "may", "might"
    };
    static const std::unordered_set<std::string> adverbs = {
        "sehr", "oft", "nie", "immer", "hier", "dort",
        "very", "often", "never", "always", "here", "there"
    };

    for (auto& tok : tokens) {
        const auto& w = tok.surface;

        // Start with uniform priors
        tok.p_noun = 0.2;
        tok.p_verb = 0.2;
        tok.p_adj  = 0.2;
        tok.p_adv  = 0.2;
        tok.p_func = 0.2;

        // --- Hard lexicon matches → near-deterministic ---
        if (determiners.count(w) || prepositions.count(w) || conjunctions.count(w)) {
            tok.p_func = 0.95; tok.p_noun = 0.01; tok.p_verb = 0.01;
            tok.p_adj = 0.01; tok.p_adv = 0.02;
            continue;  // Function words are certain
        }
        if (known_verbs.count(w)) {
            tok.p_verb = 0.90; tok.p_noun = 0.05; tok.p_adj = 0.02;
            tok.p_adv = 0.01; tok.p_func = 0.02;
            continue;
        }
        if (adverbs.count(w)) {
            tok.p_adv = 0.85; tok.p_adj = 0.08; tok.p_noun = 0.03;
            tok.p_verb = 0.02; tok.p_func = 0.02;
            continue;
        }

        // --- Suffix heuristics → shift probabilities ---
        auto ends_with = [&](const std::string& suffix) {
            return w.size() >= suffix.size() &&
                   w.compare(w.size() - suffix.size(), suffix.size(), suffix) == 0;
        };

        // German verb suffixes
        if (w.size() > 3 && (ends_with("iert") || ends_with("tet"))) {
            tok.p_verb = 0.80; tok.p_noun = 0.10; tok.p_adj = 0.05;
            tok.p_adv = 0.03; tok.p_func = 0.02;
        } else if (w.size() > 4 && ends_with("en")) {
            tok.p_verb = 0.60; tok.p_noun = 0.25; tok.p_adj = 0.10;
            tok.p_adv = 0.03; tok.p_func = 0.02;
        } else if (w.size() > 3 && ends_with("te")) {
            tok.p_verb = 0.55; tok.p_noun = 0.20; tok.p_adj = 0.15;
            tok.p_adv = 0.05; tok.p_func = 0.05;
        }
        // English verb suffixes
        else if (ends_with("ing")) {
            tok.p_verb = 0.65; tok.p_noun = 0.20; tok.p_adj = 0.10;
            tok.p_adv = 0.03; tok.p_func = 0.02;
        } else if (w.size() > 3 && ends_with("ed")) {
            tok.p_verb = 0.55; tok.p_adj = 0.25; tok.p_noun = 0.15;
            tok.p_adv = 0.03; tok.p_func = 0.02;
        }
        // Adjective suffixes (German)
        else if (ends_with("lich") || ends_with("ig") || ends_with("isch") ||
                 ends_with("bar") || ends_with("sam") || ends_with("haft")) {
            tok.p_adj = 0.80; tok.p_adv = 0.10; tok.p_noun = 0.05;
            tok.p_verb = 0.03; tok.p_func = 0.02;
        }
        // Adjective suffixes (English)
        else if (ends_with("ly")) {
            tok.p_adv = 0.60; tok.p_adj = 0.30; tok.p_noun = 0.05;
            tok.p_verb = 0.03; tok.p_func = 0.02;
        } else if (ends_with("ous") || ends_with("ive") || ends_with("ful")) {
            tok.p_adj = 0.75; tok.p_noun = 0.10; tok.p_adv = 0.08;
            tok.p_verb = 0.05; tok.p_func = 0.02;
        } else if (w.size() > 3 && ends_with("al")) {
            tok.p_adj = 0.55; tok.p_noun = 0.30; tok.p_adv = 0.08;
            tok.p_verb = 0.05; tok.p_func = 0.02;
        }
        // Default: lean toward noun (most common open-class word)
        else {
            tok.p_noun = 0.50; tok.p_verb = 0.20; tok.p_adj = 0.15;
            tok.p_adv = 0.10; tok.p_func = 0.05;
        }

        // --- Graph-based refinement: check if word matches a known concept ---
        auto candidates = ltm_.find_by_label(w);
        for (auto cid : candidates) {
            auto info = ltm_.retrieve_concept(cid);
            if (!info) continue;
            // Skip linguistic concepts
            if (info->label.size() >= 5 &&
                (info->label.substr(0, 5) == "word:" || info->label.substr(0, 4) == "sent:"))
                continue;

            POSTag graph_pos = infer_pos_from_graph(cid);
            if (graph_pos == POSTag::VERB) {
                tok.p_verb = std::min(0.95, tok.p_verb + 0.30);
            } else if (graph_pos == POSTag::ADJ) {
                tok.p_adj = std::min(0.95, tok.p_adj + 0.25);
            } else {
                // CAUSES, HAS_PROPERTY etc. → likely noun (entity)
                tok.p_noun = std::min(0.95, tok.p_noun + 0.25);
            }
            break;  // Use first matching semantic concept
        }

        // Normalize to sum=1
        double total = tok.p_noun + tok.p_verb + tok.p_adj + tok.p_adv + tok.p_func;
        if (total > 0.0) {
            tok.p_noun /= total;
            tok.p_verb /= total;
            tok.p_adj  /= total;
            tok.p_adv  /= total;
            tok.p_func /= total;
        }
    }
}

// Infer POS hint from a concept's graph relations
POSTag SentenceParser::infer_pos_from_graph(ConceptId cid) const {
    auto rels = ltm_.get_outgoing_relations(cid);
    bool has_causes = false;
    bool has_property = false, has_isa = false;

    for (const auto& r : rels) {
        if (r.type == RelationType::CAUSES || r.type == RelationType::ENABLES)
            has_causes = true;
        if (r.type == RelationType::HAS_PROPERTY)
            has_property = true;
        if (r.type == RelationType::IS_A)
            has_isa = true;
    }

    // Processes/actions with CAUSES/ENABLES → likely VERB
    if (has_causes && !has_isa) return POSTag::VERB;
    // Properties → likely ADJ
    if (has_property && !has_causes) return POSTag::ADJ;
    // Default: noun-like entity
    return POSTag::NOUN;
}

// =============================================================================
// Level 2: Sentence-level structure estimation
// =============================================================================

SentencePattern SentenceParser::estimate_structure(const std::vector<Token>& tokens) const {
    if (tokens.empty()) return SentencePattern::UNKNOWN;

    // Check for question markers
    const auto& first = tokens.front().surface;
    static const std::unordered_set<std::string> q_words = {
        "was", "wer", "wie", "wo", "warum", "wann", "welche", "welcher",
        "what", "who", "how", "where", "why", "when", "which"
    };
    if (q_words.count(first)) return SentencePattern::QUESTION;

    // Count probable nouns before/after probable verb
    int first_verb_pos = -1;
    int noun_before_verb = 0;
    int noun_after_verb = 0;

    for (const auto& tok : tokens) {
        if (tok.p_verb > 0.4 && first_verb_pos < 0) {
            first_verb_pos = tok.position;
        }
    }

    if (first_verb_pos < 0) return SentencePattern::UNKNOWN;

    for (const auto& tok : tokens) {
        if (tok.p_noun > 0.3) {
            if (tok.position < first_verb_pos) noun_before_verb++;
            else if (tok.position > first_verb_pos) noun_after_verb++;
        }
    }

    // Verb-first → VS (German Fragesatz/Imperativ without question word)
    if (first_verb_pos == 0) return SentencePattern::VS;

    // Noun-Verb-Noun → SVO (English, German main clause)
    if (noun_before_verb >= 1 && noun_after_verb >= 1) return SentencePattern::SVO;

    // Noun-Noun-Verb → SOV (German subordinate clause)
    if (noun_before_verb >= 2 && noun_after_verb == 0) return SentencePattern::SOV;

    return SentencePattern::SVO;  // Default assumption
}

// =============================================================================
// Level 2 → Level 1 feedback: update word probs based on sentence structure
// =============================================================================

void SentenceParser::update_word_probs_from_structure(
    std::vector<Token>& tokens, SentencePattern pattern) const
{
    if (tokens.empty()) return;

    // Find the most likely verb position
    int best_verb_pos = -1;
    double best_verb_prob = 0.0;
    for (const auto& tok : tokens) {
        if (tok.p_verb > best_verb_prob && tok.p_func < 0.5) {
            best_verb_prob = tok.p_verb;
            best_verb_pos = tok.position;
        }
    }

    for (auto& tok : tokens) {
        if (tok.p_func > 0.8) continue;  // Don't touch function words

        switch (pattern) {
            case SentencePattern::SVO:
                // Position before verb → boost noun (subject)
                if (best_verb_pos >= 0 && tok.position < best_verb_pos && tok.p_noun > 0.2) {
                    tok.p_noun *= 1.3;
                }
                // Position after verb → boost noun (object)
                if (best_verb_pos >= 0 && tok.position > best_verb_pos && tok.p_noun > 0.2) {
                    tok.p_noun *= 1.2;
                }
                // If token is at verb position → boost verb
                if (tok.position == best_verb_pos) {
                    tok.p_verb *= 1.3;
                }
                break;

            case SentencePattern::SOV:
                // Last content word → boost verb
                if (tok.position == tokens.back().position && tok.p_verb > 0.15) {
                    tok.p_verb *= 1.5;
                }
                // Words before last → boost noun
                if (tok.position < tokens.back().position && tok.p_noun > 0.2) {
                    tok.p_noun *= 1.3;
                }
                break;

            case SentencePattern::VS:
                // First word → boost verb
                if (tok.position == 0) {
                    tok.p_verb *= 1.4;
                }
                // After first → boost noun (subject/object)
                if (tok.position > 0 && tok.p_noun > 0.2) {
                    tok.p_noun *= 1.2;
                }
                break;

            case SentencePattern::QUESTION:
                // Second word often verb in questions
                if (tok.position == 1 && tok.p_verb > 0.15) {
                    tok.p_verb *= 1.4;
                }
                break;

            case SentencePattern::UNKNOWN:
                break;
        }

        // Renormalize
        double total = tok.p_noun + tok.p_verb + tok.p_adj + tok.p_adv + tok.p_func;
        if (total > 0.0) {
            tok.p_noun /= total;
            tok.p_verb /= total;
            tok.p_adj  /= total;
            tok.p_adv  /= total;
            tok.p_func /= total;
        }
    }
}

// =============================================================================
// Iterative convergence: word probs ↔ sentence structure
// =============================================================================

void SentenceParser::resolve_pos_iterative(std::vector<Token>& tokens) const {
    // Level 1: Compute initial word-level probabilities
    compute_word_probs(tokens);

    // Helper: determine best POS from probabilities
    auto best_pos = [](const Token& tok) -> POSTag {
        double max_p = tok.p_func;
        POSTag best = POSTag::DET;  // function word default
        if (tok.p_noun > max_p) { max_p = tok.p_noun; best = POSTag::NOUN; }
        if (tok.p_verb > max_p) { max_p = tok.p_verb; best = POSTag::VERB; }
        if (tok.p_adj > max_p)  { max_p = tok.p_adj;  best = POSTag::ADJ; }
        if (tok.p_adv > max_p)  { max_p = tok.p_adv;  best = POSTag::ADV; }
        return best;
    };

    // Iterate until convergence (max 3 iterations)
    for (int iter = 0; iter < 3; ++iter) {
        // Save old POS assignments for convergence check
        std::vector<POSTag> old_pos;
        for (const auto& tok : tokens) {
            old_pos.push_back(best_pos(tok));
        }

        // Level 2: Estimate sentence structure from current word probs
        auto pattern = estimate_structure(tokens);

        // Level 2 → Level 1: Feed structure back into word probs
        update_word_probs_from_structure(tokens, pattern);

        // Check convergence: all POS assignments stable?
        bool converged = true;
        for (size_t i = 0; i < tokens.size(); ++i) {
            if (best_pos(tokens[i]) != old_pos[i]) { converged = false; break; }
        }
        if (converged) break;
    }

    // Final assignment: pick highest probability POS
    for (auto& tok : tokens) {
        tok.pos = best_pos(tok);

        // Refine function words → DET/PREP/CONJ
        if (tok.pos == POSTag::DET) {
            static const std::unordered_set<std::string> preps = {
                "in", "auf", "mit", "von", "zu", "bei", "nach", "unter",
                "to", "from", "with", "at", "by", "for", "of"
            };
            static const std::unordered_set<std::string> conjs = {
                "und", "oder", "aber", "denn", "weil", "dass",
                "and", "or", "but", "because"
            };
            if (preps.count(tok.surface)) tok.pos = POSTag::PREP;
            else if (conjs.count(tok.surface)) tok.pos = POSTag::CONJ;
        }
    }
}

// =============================================================================
// Get or create word concept
// =============================================================================

ConceptId SentenceParser::get_or_create_word_concept(const Token& token) {
    std::string pos_str = pos_tag_str(token.pos);
    std::string label = "word:" + token.surface + ":" + pos_str;

    // Check existing via label index
    auto existing = ltm_.find_by_label(label);
    if (!existing.empty()) {
        return existing[0];
    }

    // Create new word concept
    EpistemicMetadata meta(EpistemicType::DEFINITION, EpistemicStatus::ACTIVE, 0.95);
    ConceptId word_cid = ltm_.store_concept(
        label,
        "Linguistic word: " + token.surface + " (" + pos_str + ")",
        meta
    );

    // Try to auto-link via DENOTES
    auto semantic = find_semantic_concept(token.surface);
    if (semantic) {
        ltm_.add_relation(word_cid, *semantic, RelationType::DENOTES, 0.9);
    }

    return word_cid;
}

// =============================================================================
// Find semantic concept by surface form
// =============================================================================

std::optional<ConceptId> SentenceParser::find_semantic_concept(const std::string& surface) const {
    // Direct label lookup — skip word:/sent: prefixed concepts
    auto candidates = ltm_.find_by_label(surface);
    for (auto cid : candidates) {
        auto info = ltm_.retrieve_concept(cid);
        if (!info) continue;
        if (info->label.size() >= 5 && info->label.substr(0, 5) == "word:") continue;
        if (info->label.size() >= 4 && info->label.substr(0, 4) == "sent:") continue;
        return cid;
    }

    // German morphological normalization: try stripping common suffixes
    auto try_stripped = [&](const std::string& form) -> std::optional<ConceptId> {
        auto cands = ltm_.find_by_label(form);
        for (auto cid : cands) {
            auto info = ltm_.retrieve_concept(cid);
            if (!info) continue;
            if (info->label.size() >= 5 && info->label.substr(0, 5) == "word:") continue;
            if (info->label.size() >= 4 && info->label.substr(0, 4) == "sent:") continue;
            return cid;
        }
        return std::nullopt;
    };

    if (surface.size() > 3) {
        // -en
        if (surface.size() > 4 && surface.substr(surface.size() - 2) == "en") {
            auto r = try_stripped(surface.substr(0, surface.size() - 2));
            if (r) return r;
        }
        // -n
        if (surface.back() == 'n') {
            auto r = try_stripped(surface.substr(0, surface.size() - 1));
            if (r) return r;
        }
        // -e
        if (surface.back() == 'e') {
            auto r = try_stripped(surface.substr(0, surface.size() - 1));
            if (r) return r;
        }
        // -er
        if (surface.size() > 4 && surface.substr(surface.size() - 2) == "er") {
            auto r = try_stripped(surface.substr(0, surface.size() - 2));
            if (r) return r;
        }
        // -es
        if (surface.size() > 4 && surface.substr(surface.size() - 2) == "es") {
            auto r = try_stripped(surface.substr(0, surface.size() - 2));
            if (r) return r;
        }
    }

    return std::nullopt;
}

// =============================================================================
// Parse and store a single sentence
// =============================================================================

ParsedSentence SentenceParser::parse_and_store(const std::string& sentence) {
    printf("[SentenceParser] Parsing: %s\n", sentence.c_str());
    fflush(stdout);
    ParsedSentence result{};

    // Tokenize and run iterative two-level POS resolution
    auto all_tokens = tokenize(sentence);
    resolve_pos_iterative(all_tokens);

    // Filter to content words (NOUN, VERB, ADJ, ADV)
    std::vector<Token> content_tokens;
    for (const auto& tok : all_tokens) {
        if (tok.pos == POSTag::NOUN || tok.pos == POSTag::VERB ||
            tok.pos == POSTag::ADJ || tok.pos == POSTag::ADV) {
            content_tokens.push_back(tok);
        }
    }

    if (content_tokens.empty()) {
        EpistemicMetadata sent_meta(EpistemicType::HYPOTHESIS, EpistemicStatus::ACTIVE, 0.60);
        result.sentence_id = ltm_.store_concept(
            "sent:" + sentence, sentence, sent_meta);
        return result;
    }

    // Role assignment based on iterative POS + sentence structure
    int verb_pos = -1;
    Token* verb_token = nullptr;
    Token* subject_token = nullptr;
    Token* object_token = nullptr;
    std::vector<Token*> modifier_tokens;

    // Find first verb (highest p_verb among VERB-tagged tokens)
    for (auto& tok : content_tokens) {
        if (tok.pos == POSTag::VERB) {
            verb_token = &tok;
            verb_pos = tok.position;
            break;
        }
    }

    // Assign subject/object/modifiers
    for (auto& tok : content_tokens) {
        if (tok.pos == POSTag::NOUN) {
            if (verb_pos < 0) {
                if (!subject_token)
                    subject_token = &tok;
                else if (!object_token)
                    object_token = &tok;
                else
                    modifier_tokens.push_back(&tok);
            } else if (tok.position < verb_pos) {
                if (!subject_token)
                    subject_token = &tok;
                else
                    modifier_tokens.push_back(&tok);
            } else {
                if (!object_token)
                    object_token = &tok;
                else
                    modifier_tokens.push_back(&tok);
            }
        } else if (&tok != verb_token) {
            modifier_tokens.push_back(&tok);
        }
    }

    // Create word concepts
    ConceptId subj_cid = subject_token ? get_or_create_word_concept(*subject_token) : 0;
    ConceptId verb_cid = verb_token ? get_or_create_word_concept(*verb_token) : 0;
    ConceptId obj_cid = object_token ? get_or_create_word_concept(*object_token) : 0;

    // Check for duplicate sentence (same SVO triple)
    if (subj_cid && verb_cid && obj_cid) {
        auto dup = densifier_.find_duplicate_sentence(subj_cid, verb_cid, obj_cid);
        if (dup) {
            result.sentence_id = *dup;
            result.subject_word = subj_cid;
            result.verb_word = verb_cid;
            result.object_word = obj_cid;
            auto fill_semantic = [&](ConceptId word_cid) -> std::optional<ConceptId> {
                for (const auto& rel : ltm_.get_outgoing_relations(word_cid)) {
                    if (rel.type == RelationType::DENOTES) return rel.target;
                }
                return std::nullopt;
            };
            result.subject_semantic = fill_semantic(subj_cid);
            result.verb_semantic = fill_semantic(verb_cid);
            result.object_semantic = fill_semantic(obj_cid);
            return result;
        }
    }

    // Create sentence concept
    EpistemicMetadata sent_meta(EpistemicType::HYPOTHESIS, EpistemicStatus::ACTIVE, 0.60);
    ConceptId sent_cid = ltm_.store_concept("sent:" + sentence, sentence, sent_meta);
    result.sentence_id = sent_cid;

    // Add SVO relations: word → sentence
    if (subj_cid) {
        ltm_.add_relation(subj_cid, sent_cid, RelationType::SUBJECT_OF, 1.0);
        result.subject_word = subj_cid;
    }
    if (verb_cid) {
        ltm_.add_relation(verb_cid, sent_cid, RelationType::VERB_OF, 1.0);
        result.verb_word = verb_cid;
    }
    if (obj_cid) {
        ltm_.add_relation(obj_cid, sent_cid, RelationType::OBJECT_OF, 1.0);
        result.object_word = obj_cid;
    }

    // Modifier relations
    for (auto* mod_tok : modifier_tokens) {
        ConceptId mod_cid = get_or_create_word_concept(*mod_tok);
        ltm_.add_relation(mod_cid, sent_cid, RelationType::MODIFIER_OF, 0.8);
        result.modifiers.push_back(mod_cid);
    }

    // Fill semantic fields from DENOTES targets
    auto fill_semantic = [&](ConceptId word_cid) -> std::optional<ConceptId> {
        for (const auto& rel : ltm_.get_outgoing_relations(word_cid)) {
            if (rel.type == RelationType::DENOTES) return rel.target;
        }
        return std::nullopt;
    };
    if (subj_cid) result.subject_semantic = fill_semantic(subj_cid);
    if (verb_cid) result.verb_semantic = fill_semantic(verb_cid);
    if (obj_cid) result.object_semantic = fill_semantic(obj_cid);

    return result;
}

// =============================================================================
// Parse discourse: multiple sentences with PRECEDES chain
// =============================================================================

std::vector<ParsedSentence> SentenceParser::parse_discourse(
    const std::vector<std::string>& sentences)
{
    std::vector<ParsedSentence> results;
    results.reserve(sentences.size());

    for (const auto& sent : sentences) {
        auto parsed = parse_and_store(sent);

        if (!results.empty()) {
            ltm_.add_relation(results.back().sentence_id, parsed.sentence_id,
                              RelationType::PRECEDES, 1.0);
        }

        results.push_back(std::move(parsed));
    }

    return results;
}

// =============================================================================
// Lookup word concepts by surface form
// =============================================================================

std::vector<ConceptId> SentenceParser::lookup_word(const std::string& surface_form) const {
    std::vector<ConceptId> results;

    std::string lower = surface_form;
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    static const POSTag all_pos[] = {
        POSTag::NOUN, POSTag::VERB, POSTag::ADJ, POSTag::ADV,
        POSTag::DET, POSTag::PREP, POSTag::CONJ
    };

    for (auto pos : all_pos) {
        std::string label = "word:" + lower + ":" + pos_tag_str(pos);
        auto found = ltm_.find_by_label(label);
        results.insert(results.end(), found.begin(), found.end());
    }

    return results;
}

// =============================================================================
// Link word concept to semantic concept via DENOTES
// =============================================================================

void SentenceParser::link_word_to_concept(ConceptId word, ConceptId semantic) {
    auto existing = ltm_.get_relations_between(word, semantic);
    for (const auto& rel : existing) {
        if (rel.type == RelationType::DENOTES) return;
    }
    ltm_.add_relation(word, semantic, RelationType::DENOTES, 0.9);
}

} // namespace brain19
