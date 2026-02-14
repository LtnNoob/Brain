#include "chat_interface.hpp"
#include "../memory/relation_type_registry.hpp"
#include <sstream>
#include <algorithm>
#include <iostream>
#include <cctype>
#include <cmath>
#include <set>
#include <unordered_set>

namespace brain19 {

ChatInterface::ChatInterface() = default;
ChatInterface::~ChatInterface() = default;

bool ChatInterface::is_llm_available() const {
    return false;
}

// ─── Stop Words ──────────────────────────────────────────────────────────────

static const std::unordered_set<std::string> STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "must",
    "i", "me", "my", "we", "our", "you", "your", "he", "she", "it",
    "they", "them", "their", "his", "her", "its",
    "what", "which", "who", "whom", "this", "that", "these", "those",
    "am", "if", "or", "and", "but", "not", "no", "so", "too", "very",
    "of", "in", "on", "at", "to", "for", "with", "by", "from", "about",
    "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "each",
    "every", "both", "few", "more", "most", "other", "some", "such",
    "only", "own", "same", "than", "just", "also", "now",
    // German common
    "ein", "eine", "einen", "einem", "einer", "eines",
    "der", "die", "das", "den", "dem", "des",
    "ist", "sind", "war", "waren", "wird", "werden", "wurde", "wurden",
    "hat", "haben", "hatte", "hatten",
    "und", "oder", "nicht", "kein", "keine", "keinem", "keinen",
    "ich", "du", "er", "sie", "es", "wir", "ihr", "man",
    "von", "zu", "zum", "zur", "mit", "auf", "aus", "fuer", "ueber", "nach",
    "bei", "vor", "hinter", "neben", "zwischen", "unter", "an", "in", "im",
    "wie", "was", "wer", "wo", "warum", "wann", "wenn", "ob", "dass", "weil",
    "denn", "aber", "doch", "auch", "noch", "schon", "sehr", "nur", "mal",
    "dann", "dort", "hier", "da", "so", "als", "bis",
    "kann", "muss", "soll", "will", "darf", "mag",
    "alle", "alles", "jeder", "jede", "jedes", "jedem",
    "welch", "welche", "welcher", "welches", "welchem",
    // German pronouns
    "mir", "mich", "dir", "dich", "ihm", "ihn", "uns", "euch", "ihnen",
    "mein", "meine", "meinem", "meinen", "meiner", "meines",
    "dein", "deine", "deinem", "deinen", "deiner", "deines",
    "sein", "seine", "seinem", "seinen", "seiner", "seines",
    "unser", "unsere", "unserem", "unseren", "unserer", "unseres",
    "euer", "eure", "eurem", "euren", "eurer", "eures",
};

static const std::unordered_set<std::string> INTENT_VERBS = {
    "erklaer", "erklaere", "beschreib", "beschreibe",
    "zeig", "zeige", "definier", "definiere",
    "vergleich", "vergleiche", "finde", "such", "suche",
    "nenn", "nenne", "sag", "sage",
    "passiert", "geschieht", "bedeutet", "heisst",
    "explain", "describe", "show", "tell", "define",
    "compare", "find", "search", "list", "happens",
};

static std::string to_lower(const std::string& s) {
    std::string result = s;
    std::transform(result.begin(), result.end(), result.begin(), ::tolower);
    return result;
}

static std::vector<std::string> extract_keywords(const std::string& text) {
    std::string lower = to_lower(text);
    std::vector<std::string> keywords;
    std::string word;
    for (char c : lower) {
        if (std::isalnum(static_cast<unsigned char>(c))) {
            word += c;
        } else {
            if (word.size() >= 2 &&
                STOP_WORDS.find(word) == STOP_WORDS.end() &&
                INTENT_VERBS.find(word) == INTENT_VERBS.end()) {
                keywords.push_back(word);
            }
            word.clear();
        }
    }
    if (word.size() >= 2 &&
        STOP_WORDS.find(word) == STOP_WORDS.end() &&
        INTENT_VERBS.find(word) == INTENT_VERBS.end()) {
        keywords.push_back(word);
    }
    return keywords;
}

// ─── Levenshtein Distance ────────────────────────────────────────────────────

static size_t levenshtein(const std::string& a, const std::string& b) {
    size_t m = a.size(), n = b.size();
    std::vector<size_t> prev(n + 1), curr(n + 1);
    for (size_t j = 0; j <= n; ++j) prev[j] = j;
    for (size_t i = 1; i <= m; ++i) {
        curr[0] = i;
        for (size_t j = 1; j <= n; ++j) {
            size_t cost = (a[i-1] == b[j-1]) ? 0 : 1;
            curr[j] = std::min({prev[j] + 1, curr[j-1] + 1, prev[j-1] + cost});
        }
        std::swap(prev, curr);
    }
    return prev[n];
}

// ─── Intent Classification ───────────────────────────────────────────────────

static const std::vector<std::string> GREETING_WORDS = {
    "hey", "hi", "hello", "hallo", "moin", "greetings", "sup", "yo",
    "good morning", "good evening", "good afternoon", "guten tag",
    "guten morgen", "guten abend", "howdy", "whats up", "what's up",
};

QueryIntent ChatInterface::classify_intent(const std::string& question) {
    std::string lower = to_lower(question);

    // Strip punctuation for matching
    std::string stripped;
    for (char c : lower) {
        if (std::isalnum(static_cast<unsigned char>(c)) || c == ' ' || c == '\'')
            stripped += c;
    }
    // Trim
    while (!stripped.empty() && stripped.front() == ' ') stripped.erase(stripped.begin());
    while (!stripped.empty() && stripped.back() == ' ') stripped.pop_back();

    // Greeting: exact or prefix match against known greetings
    for (const auto& g : GREETING_WORDS) {
        if (stripped == g || stripped.find(g) == 0) {
            // Make sure it's actually a greeting, not "hello world is a program"
            if (stripped.size() <= g.size() + 10) return QueryIntent::GREETING;
        }
    }

    // Question: contains '?' or starts with question word
    if (lower.find('?') != std::string::npos) return QueryIntent::QUESTION;
    for (const auto& qw : {"what ", "how ", "why ", "when ", "where ", "who ",
                            "which ", "is ", "are ", "does ", "do ", "can ",
                            "was ", "wie ", "warum ", "wer ", "wo "}) {
        if (lower.find(qw) == 0) return QueryIntent::QUESTION;
    }

    // Command: starts with imperative
    for (const auto& cw : {"tell ", "explain ", "show ", "list ", "describe ",
                            "define ", "compare ", "find ", "search ",
                            "zeig ", "erklaer "}) {
        if (lower.find(cw) == 0) return QueryIntent::COMMAND;
    }

    // Default: treat as a question if it mentions a topic, else statement
    auto kws = extract_keywords(lower);
    if (!kws.empty()) return QueryIntent::QUESTION;

    return QueryIntent::STATEMENT;
}

// ─── Multi-Strategy Concept Finding ──────────────────────────────────────────

struct ScoredConcept {
    ConceptInfo info;
    double score;  // higher = more relevant
};

std::vector<ConceptInfo> ChatInterface::find_relevant_concepts(
    const std::string& question,
    const LongTermMemory& ltm
) {
    auto all_ids = ltm.get_active_concepts();
    std::string lower_q = to_lower(question);
    auto keywords = extract_keywords(question);

    // Score each concept with multiple strategies
    std::vector<ScoredConcept> scored;

    for (auto id : all_ids) {
        auto info_opt = ltm.retrieve_concept(id);
        if (!info_opt.has_value()) continue;
        const ConceptInfo& info = info_opt.value();

        std::string lower_label = to_lower(info.label);
        std::string lower_def = to_lower(info.definition);
        double score = 0.0;

        // Strategy 1: Exact label match (highest priority)
        if (lower_q == lower_label) {
            score += 10.0;
        }

        // Strategy 2: Label substring in query
        if (lower_label.size() >= 3 && lower_q.find(lower_label) != std::string::npos) {
            // Weight by label length — longer matches are more specific
            score += 5.0 * (static_cast<double>(lower_label.size()) / lower_q.size());
        }

        // Strategy 3: Query substring in label (reverse)
        if (lower_q.size() >= 3 && lower_label.find(lower_q) != std::string::npos) {
            score += 4.0;
        }

        // Strategy 4: Keyword match on labels (weighted by keyword length)
        for (const auto& kw : keywords) {
            double kw_weight = std::min(2.0, 0.5 + static_cast<double>(kw.size()) / 8.0);
            if (lower_label.find(kw) != std::string::npos) {
                score += 3.0 * kw_weight;
            }
        }

        // Strategy 5: Keyword match on definitions (weighted by keyword length)
        for (const auto& kw : keywords) {
            double kw_weight = std::min(2.0, 0.5 + static_cast<double>(kw.size()) / 8.0);
            if (lower_def.find(kw) != std::string::npos) {
                score += 1.5 * kw_weight;
            }
        }

        // Strategy 6: Fuzzy match — Levenshtein on label vs each keyword
        for (const auto& kw : keywords) {
            double kw_weight = std::min(2.0, 0.5 + static_cast<double>(kw.size()) / 8.0);
            if (kw.size() >= 3 && lower_label.size() >= 3) {
                size_t dist = levenshtein(kw, lower_label);
                size_t max_len = std::max(kw.size(), lower_label.size());
                double similarity = 1.0 - (static_cast<double>(dist) / max_len);
                if (similarity >= 0.7) {
                    score += 2.0 * similarity * kw_weight;
                }
            }
            // Also check prefix match
            if (kw.size() >= 3 && lower_label.size() >= kw.size() &&
                lower_label.substr(0, kw.size()) == kw) {
                score += 2.5 * kw_weight;
            }
        }

        // Strategy 7: Multi-word label — check each word
        {
            std::vector<std::string> label_words;
            std::string w;
            for (char c : lower_label) {
                if (std::isalnum(static_cast<unsigned char>(c))) {
                    w += c;
                } else if (!w.empty()) {
                    label_words.push_back(w);
                    w.clear();
                }
            }
            if (!w.empty()) label_words.push_back(w);

            size_t matched_words = 0;
            for (const auto& lw : label_words) {
                if (lw.size() < 3) continue;
                bool word_matched = false;
                for (const auto& kw : keywords) {
                    double kw_weight = std::min(2.0, 0.5 + static_cast<double>(kw.size()) / 8.0);
                    if (lw == kw) {
                        score += 3.5 * kw_weight;
                        word_matched = true;
                    } else if (lw.size() >= 4 && kw.size() >= 4 &&
                               lw.substr(0, 4) == kw.substr(0, 4)) {
                        // Stem prefix match (e.g., "grav" matches "gravity")
                        score += 2.0 * kw_weight;
                        word_matched = true;
                    }
                }
                if (word_matched) matched_words++;
            }

            // Multi-word label penalty: if full phrase not in query and
            // only partial word match (<50% coverage), heavily discount
            if (label_words.size() > 1 &&
                lower_q.find(lower_label) == std::string::npos) {
                double coverage = (label_words.empty()) ? 0.0
                    : static_cast<double>(matched_words) / static_cast<double>(label_words.size());
                if (coverage < 0.5) {
                    score *= 0.2;
                }
            }
        }

        // Epistemic trust bonus: higher-trust concepts rank higher
        score *= (0.8 + 0.2 * info.epistemic.trust);

        if (score > 0.0) {
            scored.push_back({info, score});
        }
    }

    // Sort by score descending
    std::sort(scored.begin(), scored.end(),
        [](const ScoredConcept& a, const ScoredConcept& b) {
            return a.score > b.score;
        });

    // Return top results (limit to avoid noise)
    std::vector<ConceptInfo> results;
    size_t limit = std::min(scored.size(), size_t(10));
    for (size_t i = 0; i < limit; ++i) {
        results.push_back(scored[i].info);
    }
    return results;
}

// ─── Epistemic Context Builder ───────────────────────────────────────────────

std::string ChatInterface::build_epistemic_context(
    const std::vector<ConceptInfo>& concepts
) {
    if (concepts.empty()) {
        return "NO RELEVANT KNOWLEDGE IN LTM";
    }

    std::ostringstream ctx;
    ctx << "AVAILABLE KNOWLEDGE FROM LTM:\n\n";

    for (const auto& info : concepts) {
        ctx << "--- Concept: " << info.label << " ---\n";
        ctx << "Type: " << epistemic_type_to_string(info.epistemic.type) << "\n";
        ctx << "Status: " << epistemic_status_to_string(info.epistemic.status) << "\n";
        ctx << "Trust: " << (info.epistemic.trust * 100.0) << "%\n";
        ctx << "Definition: " << info.definition << "\n\n";
    }

    return ctx.str();
}

// ─── Intent-Aware Formatters ─────────────────────────────────────────────────

std::string ChatInterface::format_greeting(const std::vector<ConceptInfo>& top_concepts) {
    std::ostringstream ans;
    ans << "Hallo! Ich bin Brain19.\n\n";
    ans << "Ich kenne " << total_concepts_ << " Konzepte und " << total_relations_
        << " Relationen in meinem Wissensnetzwerk. ";
    if (!top_concepts.empty()) {
        ans << "Frag mich z.B. nach **" << top_concepts[0].label << "**";
        if (top_concepts.size() > 1) {
            ans << " oder **" << top_concepts[1].label << "**";
        }
        ans << ".\n";
    } else {
        ans << "Stell mir eine Frage!\n";
    }
    return ans.str();
}

std::string ChatInterface::format_question(
    const std::vector<ConceptInfo>& top_concepts,
    const std::vector<std::string>& /*thought_paths*/
) {
    std::ostringstream ans;

    if (top_concepts.empty()) {
        ans << "Dazu habe ich kein Wissen in meinem Netzwerk.\n";
        return ans.str();
    }

    const auto& primary = top_concepts[0];
    ans << "**" << primary.label << "**: " << primary.definition << "\n";

    if (top_concepts.size() > 1) {
        ans << "\nVerwandte Konzepte: ";
        size_t shown = 0;
        for (size_t i = 1; i < top_concepts.size() && shown < 3; ++i) {
            const auto& c = top_concepts[i];
            if (shown > 0) ans << " | ";
            ans << "**" << c.label << "**";
            if (!c.definition.empty()) {
                ans << " — " << c.definition.substr(0, 100);
                if (c.definition.size() > 100) ans << "...";
            }
            ++shown;
        }
        ans << "\n";
    }

    return ans.str();
}

std::string ChatInterface::format_statement(const std::vector<ConceptInfo>& top_concepts) {
    std::ostringstream ans;

    if (top_concepts.empty()) {
        ans << "Ich kann diese Aussage nicht meinem Wissensnetzwerk zuordnen.\n";
        return ans.str();
    }

    ans << "This touches the following concepts in my knowledge network:\n\n";
    size_t shown = 0;
    for (const auto& c : top_concepts) {
        if (shown >= 3) break;
        ans << "**" << c.label << "** ("
            << epistemic_type_to_string(c.epistemic.type)
            << ", Trust: " << static_cast<int>(c.epistemic.trust * 100) << "%)\n";
        ans << "  " << c.definition << "\n\n";
        ++shown;
    }
    return ans.str();
}

// ─── Ask (no thinking context) ───────────────────────────────────────────────

ChatResponse ChatInterface::ask(
    const std::string& question,
    const LongTermMemory& ltm
) {
    ChatResponse response;
    response.used_llm = false;
    response.intent = classify_intent(question);

    auto relevant = find_relevant_concepts(question, ltm);

    for (const auto& info : relevant) {
        response.referenced_concepts.push_back(info.id);
        if (info.epistemic.type == EpistemicType::SPECULATION ||
            info.epistemic.type == EpistemicType::HYPOTHESIS) {
            response.contains_speculation = true;
        }
        if (info.epistemic.status == EpistemicStatus::INVALIDATED) {
            response.epistemic_note = "Contains invalidated knowledge";
        } else if (info.epistemic.trust < 0.5 && response.epistemic_note.empty()) {
            response.epistemic_note = "Low certainty information";
        }
    }

    switch (response.intent) {
        case QueryIntent::GREETING:
            response.answer = format_greeting(relevant);
            break;
        case QueryIntent::QUESTION:
        case QueryIntent::COMMAND:
            response.answer = format_question(relevant, {});
            break;
        case QueryIntent::STATEMENT:
            response.answer = format_statement(relevant);
            break;
        default:
            response.answer = format_question(relevant, {});
            break;
    }

    return response;
}

// ─── Ask With Context (thinking pipeline results) ────────────────────────────

ChatResponse ChatInterface::ask_with_context(
    const std::string& question,
    const LongTermMemory& ltm,
    const std::vector<ConceptId>& salient_concepts,
    const std::vector<std::string>& thought_paths_summary,
    QueryIntent intent
) {
    ChatResponse response;
    response.used_llm = false;
    response.contains_speculation = false;
    response.intent = (intent != QueryIntent::UNKNOWN) ? intent : classify_intent(question);

    // Collect salient concepts
    std::vector<ConceptInfo> relevant;
    for (auto cid : salient_concepts) {
        auto info_opt = ltm.retrieve_concept(cid);
        if (info_opt.has_value()) {
            relevant.push_back(info_opt.value());
            response.referenced_concepts.push_back(cid);
            if (info_opt->epistemic.type == EpistemicType::SPECULATION ||
                info_opt->epistemic.type == EpistemicType::HYPOTHESIS) {
                response.contains_speculation = true;
            }
            if (info_opt->epistemic.status == EpistemicStatus::INVALIDATED) {
                response.epistemic_note = "Contains invalidated knowledge";
            } else if (info_opt->epistemic.trust < 0.5 && response.epistemic_note.empty()) {
                response.epistemic_note = "Low certainty information";
            }
        }
    }

    // Merge keyword matches (dedup)
    auto keyword_matches = find_relevant_concepts(question, ltm);
    std::set<ConceptId> seen;
    for (const auto& r : relevant) seen.insert(r.id);
    for (const auto& info : keyword_matches) {
        if (seen.find(info.id) == seen.end()) {
            relevant.push_back(info);
            response.referenced_concepts.push_back(info.id);
            seen.insert(info.id);
        }
    }

    // Re-rank: concepts matching the query text should come first
    // Build a score map from keyword_matches (already scored by find_relevant_concepts)
    std::unordered_map<ConceptId, size_t> text_rank;
    for (size_t i = 0; i < keyword_matches.size(); ++i) {
        text_rank[keyword_matches[i].id] = i;
    }
    // Stable sort: keyword-matched concepts first (by their rank), then others
    std::stable_sort(relevant.begin(), relevant.end(),
        [&text_rank](const ConceptInfo& a, const ConceptInfo& b) {
            auto it_a = text_rank.find(a.id);
            auto it_b = text_rank.find(b.id);
            size_t rank_a = (it_a != text_rank.end()) ? it_a->second : 9999;
            size_t rank_b = (it_b != text_rank.end()) ? it_b->second : 9999;
            return rank_a < rank_b;
        });

    // Format based on intent
    switch (response.intent) {
        case QueryIntent::GREETING:
            response.answer = format_greeting(relevant);
            break;
        case QueryIntent::QUESTION:
        case QueryIntent::COMMAND:
            response.answer = format_question(relevant, thought_paths_summary);
            break;
        case QueryIntent::STATEMENT:
            response.answer = format_statement(relevant);
            break;
        default:
            response.answer = format_question(relevant, thought_paths_summary);
            break;
    }

    return response;
}

// ─── Ask With Thinking (Full Cognitive Pipeline) ──────────────────────────────

ChatResponse ChatInterface::ask_with_thinking(
    const std::string& question,
    const LongTermMemory& ltm,
    const ThinkingContext& thinking,
    QueryIntent intent
) {
    ChatResponse response;
    response.used_llm = false;
    response.contains_speculation = false;
    response.intent = (intent != QueryIntent::UNKNOWN) ? intent : classify_intent(question);

    // Collect concept info for salient concepts
    std::vector<ConceptInfo> relevant;
    for (auto cid : thinking.salient_concepts) {
        auto info_opt = ltm.retrieve_concept(cid);
        if (info_opt.has_value()) {
            relevant.push_back(info_opt.value());
            response.referenced_concepts.push_back(cid);
            if (info_opt->epistemic.type == EpistemicType::SPECULATION ||
                info_opt->epistemic.type == EpistemicType::HYPOTHESIS) {
                response.contains_speculation = true;
            }
            if (info_opt->epistemic.status == EpistemicStatus::INVALIDATED) {
                response.epistemic_note = "Contains invalidated knowledge";
            } else if (info_opt->epistemic.trust < 0.5 && response.epistemic_note.empty()) {
                response.epistemic_note = "Low certainty information";
            }
        }
    }

    // Merge keyword matches (dedup by ID)
    auto keyword_matches = find_relevant_concepts(question, ltm);
    std::set<ConceptId> seen;
    for (const auto& r : relevant) seen.insert(r.id);
    for (const auto& info : keyword_matches) {
        if (seen.find(info.id) == seen.end()) {
            relevant.push_back(info);
            response.referenced_concepts.push_back(info.id);
            seen.insert(info.id);
        }
    }

    // Re-rank: keyword-matched first
    std::unordered_map<ConceptId, size_t> text_rank;
    for (size_t i = 0; i < keyword_matches.size(); ++i) {
        text_rank[keyword_matches[i].id] = i;
    }
    std::stable_sort(relevant.begin(), relevant.end(),
        [&text_rank](const ConceptInfo& a, const ConceptInfo& b) {
            auto it_a = text_rank.find(a.id);
            auto it_b = text_rank.find(b.id);
            size_t rank_a = (it_a != text_rank.end()) ? it_a->second : 9999;
            size_t rank_b = (it_b != text_rank.end()) ? it_b->second : 9999;
            return rank_a < rank_b;
        });

    // Format based on intent
    switch (response.intent) {
        case QueryIntent::GREETING:
            response.answer = format_greeting(relevant);
            break;
        case QueryIntent::STATEMENT:
            response.answer = format_statement(relevant);
            break;
        default:
            response.answer = format_thinking_response(relevant, thinking, ltm);
            break;
    }

    return response;
}

// ─── NLG Helpers ────────────────────────────────────────────────────────────

// Fallback German verbs for common custom relation slugs from foundation data
static const std::unordered_map<std::string, std::string> CUSTOM_VERB_DE = {
    {"relates to", "steht in Beziehung zu"}, {"related to", "steht in Beziehung zu"},
    {"occurs in", "findet statt in"}, {"occurs-in", "findet statt in"},
    {"enabled by", "wird ermoeglicht durch"}, {"enabled-by", "wird ermoeglicht durch"},
    {"used in", "wird verwendet in"}, {"used-in", "wird verwendet in"},
    {"used by", "wird verwendet von"}, {"used-by", "wird verwendet von"},
    {"used for", "wird verwendet fuer"}, {"used-for", "wird verwendet fuer"},
    {"created by", "wurde erschaffen von"}, {"created-by", "wurde erschaffen von"},
    {"type of", "ist ein Typ von"}, {"type-of", "ist ein Typ von"},
    {"made of", "besteht aus"}, {"made-of", "besteht aus"},
    {"found in", "kommt vor in"}, {"found-in", "kommt vor in"},
    {"located in", "befindet sich in"}, {"located-in", "befindet sich in"},
    {"based on", "basiert auf"}, {"based-on", "basiert auf"},
    {"caused by", "wird verursacht durch"}, {"caused-by", "wird verursacht durch"},
    {"active in", "ist aktiv in"}, {"active-in", "ist aktiv in"},
    {"applies to", "gilt fuer"}, {"applies-to", "gilt fuer"},
    {"leads to", "fuehrt zu"}, {"leads-to", "fuehrt zu"},
    {"depends on", "haengt ab von"}, {"depends-on", "haengt ab von"},
    {"connected to", "ist verbunden mit"}, {"connected-to", "ist verbunden mit"},
    {"example of", "ist ein Beispiel fuer"}, {"example-of", "ist ein Beispiel fuer"},
    {"component of", "ist Bestandteil von"}, {"component-of", "ist Bestandteil von"},
    {"influences", "beeinflusst"}, {"controls", "steuert"},
    {"prevents", "verhindert"}, {"triggers", "loest aus"},
    {"produces", "erzeugt"}, {"contains", "enthaelt"},
    {"protects", "schuetzt"}, {"generates", "erzeugt"},
    {"measures", "misst"}, {"affects", "beeinflusst"},
};

static std::string relation_verb_de(const std::string& relation_name) {
    auto& reg = RelationTypeRegistry::instance();
    auto type_opt = reg.find_by_name(relation_name);
    if (type_opt) {
        const auto& de = reg.get_name_de(*type_opt);
        if (!de.empty() && de != relation_name) return de;
    }
    // Humanize slug first, then check custom map
    std::string humanized = relation_name;
    for (auto& ch : humanized) {
        if (ch == '_' || ch == '-') ch = ' ';
        else ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    auto it = CUSTOM_VERB_DE.find(humanized);
    if (it != CUSTOM_VERB_DE.end()) return it->second;
    return humanized;
}

// ─── Full-Pipeline Response Formatter ────────────────────────────────────────

std::string ChatInterface::format_thinking_response(
    const std::vector<ConceptInfo>& top_concepts,
    const ThinkingContext& thinking,
    const LongTermMemory& /*ltm*/
) {
    std::ostringstream ans;

    if (top_concepts.empty()) {
        ans << "Dazu habe ich kein Wissen in meinem Netzwerk.\n";
        return ans.str();
    }

    const auto& primary = top_concepts[0];

    // ── Build connected set: concepts linked to primary via relations ──
    std::unordered_set<ConceptId> connected;
    connected.insert(primary.id);
    for (const auto& rl : thinking.relation_links) {
        if (rl.source == primary.id) connected.insert(rl.target);
        if (rl.target == primary.id) connected.insert(rl.source);
    }

    // ── Section 1: Primary concept + definition ──
    ans << "**" << primary.label << "**: " << primary.definition << "\n";

    // ── Section 2: Relations as natural German sentences (weight > 0.5) ──
    // Group by (source_label, relation_verb) → list of target_labels
    {
        struct GroupKey {
            std::string source;
            std::string verb;
            std::string raw_relation;
        };
        std::vector<std::pair<GroupKey, std::vector<std::string>>> groups;

        for (const auto& rl : thinking.relation_links) {
            if (rl.weight <= 0.5) continue;
            if (rl.source != primary.id && rl.target != primary.id) continue;
            // Skip generic "relates to" noise
            std::string rn_lower = rl.relation_name;
            for (auto& ch : rn_lower) {
                if (ch == '_' || ch == '-') ch = ' ';
                else ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
            }
            if (rn_lower == "relates to" || rn_lower == "related to") continue;

            std::string verb = relation_verb_de(rl.relation_name);
            bool found = false;
            for (auto& [k, targets] : groups) {
                if (k.source == rl.source_label && k.raw_relation == rl.relation_name) {
                    // Dedup targets
                    bool dup = false;
                    for (const auto& t : targets) {
                        if (t == rl.target_label) { dup = true; break; }
                    }
                    if (!dup) targets.push_back(rl.target_label);
                    found = true;
                    break;
                }
            }
            if (!found) {
                groups.push_back({{rl.source_label, verb, rl.relation_name}, {rl.target_label}});
            }
        }

        if (!groups.empty()) {
            for (const auto& [key, targets] : groups) {
                ans << key.source << " " << key.verb << " ";
                for (size_t i = 0; i < targets.size(); ++i) {
                    if (i > 0 && i == targets.size() - 1) ans << " und ";
                    else if (i > 0) ans << ", ";
                    ans << targets[i];
                }
                ans << ". ";
            }
            ans << "\n";
        }
    }

    // ── Section 3: Related concepts (max 3, connected only) ──
    if (top_concepts.size() > 1) {
        std::vector<const ConceptInfo*> filtered;
        for (size_t i = 1; i < top_concepts.size() && filtered.size() < 3; ++i) {
            if (connected.count(top_concepts[i].id)) {
                filtered.push_back(&top_concepts[i]);
            }
        }
        if (!filtered.empty()) {
            ans << "\nVerwandte Konzepte: ";
            for (size_t i = 0; i < filtered.size(); ++i) {
                if (i > 0) ans << " | ";
                ans << "**" << filtered[i]->label << "**";
                if (!filtered[i]->definition.empty()) {
                    ans << " — " << filtered[i]->definition.substr(0, 100);
                    if (filtered[i]->definition.size() > 100) ans << "...";
                }
            }
            ans << "\n";
        }
    }

    // ── Section 4: Hypotheses (confidence > 0.5 only) ──
    {
        bool first = true;
        for (const auto& hyp : thinking.hypothesis_insights) {
            if (hyp.confidence <= 0.5) continue;
            if (first) { ans << "\n"; first = false; }
            ans << "- " << hyp.statement
                << " (Hypothese, " << static_cast<int>(hyp.confidence * 100) << "%";
            if (hyp.kan_validated) {
                ans << ", " << hyp.validation_status;
            }
            ans << ")\n";
        }
    }

    // ── Section 5: Contradictions (severity > 0.7 only) ──
    {
        bool first = true;
        for (const auto& c : thinking.contradiction_notes) {
            if (c.severity <= 0.7) continue;
            if (first) { ans << "\n"; first = false; }
            ans << "- " << c.description
                << " (Schwere: " << static_cast<int>(c.severity * 100) << "%)\n";
        }
    }

    return ans.str();
}

// ─── Explain Concept ─────────────────────────────────────────────────────────

std::string ChatInterface::explain_concept(
    ConceptId id,
    const LongTermMemory& ltm
) {
    auto info_opt = ltm.retrieve_concept(id);
    if (!info_opt.has_value()) {
        return "Concept not found.";
    }

    const ConceptInfo& info = info_opt.value();

    std::ostringstream out;
    out << "=== " << info.label << " ===\n\n";
    out << "Type: " << epistemic_type_to_string(info.epistemic.type) << "\n";
    out << "Status: " << epistemic_status_to_string(info.epistemic.status) << "\n";
    out << "Trust: " << (info.epistemic.trust * 100.0) << "%\n\n";
    out << info.definition << "\n";
    return out.str();
}

// ─── Compare ─────────────────────────────────────────────────────────────────

std::string ChatInterface::compare(
    ConceptId id1,
    ConceptId id2,
    const LongTermMemory& ltm
) {
    auto info1_opt = ltm.retrieve_concept(id1);
    auto info2_opt = ltm.retrieve_concept(id2);

    if (!info1_opt.has_value() || !info2_opt.has_value()) {
        return "One or both concepts not found.";
    }

    const ConceptInfo& info1 = info1_opt.value();
    const ConceptInfo& info2 = info2_opt.value();

    std::ostringstream out;
    out << "=== Comparison ===\n\n";
    out << "1. " << info1.label << " (" << epistemic_type_to_string(info1.epistemic.type);
    out << ", " << (info1.epistemic.trust * 100.0) << "%)\n";
    out << "   " << info1.definition << "\n\n";
    out << "2. " << info2.label << " (" << epistemic_type_to_string(info2.epistemic.type);
    out << ", " << (info2.epistemic.trust * 100.0) << "%)\n";
    out << "   " << info2.definition << "\n";
    return out.str();
}

// ─── List Knowledge ──────────────────────────────────────────────────────────

std::string ChatInterface::list_knowledge(
    const LongTermMemory& ltm,
    EpistemicType type
) {
    auto ids = ltm.get_concepts_by_type(type);
    std::ostringstream out;

    out << "=== " << epistemic_type_to_string(type) << " ===\n\n";

    if (ids.empty()) {
        out << "No " << epistemic_type_to_string(type) << " found.\n";
        return out.str();
    }

    out << "Found: " << ids.size() << " concept(s)\n\n";

    for (auto id : ids) {
        auto info_opt = ltm.retrieve_concept(id);
        if (info_opt.has_value()) {
            const ConceptInfo& info = info_opt.value();
            out << "• " << info.label;
            out << " (Trust: " << (info.epistemic.trust * 100.0) << "%)";

            if (info.epistemic.status == EpistemicStatus::INVALIDATED) {
                out << " INVALIDATED";
            }

            out << "\n  ID: " << id << "\n";
        }
    }

    return out.str();
}

// ─── Summary ─────────────────────────────────────────────────────────────────

std::string ChatInterface::get_summary(const LongTermMemory& ltm) {
    auto facts = ltm.get_concepts_by_type(EpistemicType::FACT);
    auto theories = ltm.get_concepts_by_type(EpistemicType::THEORY);
    auto hypotheses = ltm.get_concepts_by_type(EpistemicType::HYPOTHESIS);
    auto specs = ltm.get_concepts_by_type(EpistemicType::SPECULATION);
    auto all = ltm.get_active_concepts();

    std::ostringstream out;

    out << "=== Brain19 - Knowledge Overview ===\n\n";
    out << "Total: " << all.size() << " active concepts\n\n";

    out << "By type:\n";
    out << "  Facts:        " << facts.size() << "\n";
    out << "  Theories:     " << theories.size() << "\n";
    out << "  Hypotheses:   " << hypotheses.size() << "\n";
    out << "  Speculations: " << specs.size() << "\n\n";

    out << "Output: Template-Engine (no LLM)\n";

    int invalidated_count = 0;
    for (auto id : all) {
        auto info_opt = ltm.retrieve_concept(id);
        if (info_opt.has_value()) {
            if (info_opt->epistemic.status == EpistemicStatus::INVALIDATED) {
                invalidated_count++;
            }
        }
    }

    if (invalidated_count > 0) {
        out << "\n" << invalidated_count << " invalidated concept(s)\n";
    }

    return out.str();
}

} // namespace brain19
