#include "chat_interface.hpp"
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
    "ein", "eine", "der", "die", "das", "ist", "sind", "war", "und",
    "oder", "nicht", "ich", "du", "er", "sie", "es", "wir", "ihr",
    "von", "zu", "mit", "auf", "aus", "fuer", "ueber", "nach",
    "wie", "was", "wer", "wo", "warum",
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
            if (word.size() >= 2 && STOP_WORDS.find(word) == STOP_WORDS.end()) {
                keywords.push_back(word);
            }
            word.clear();
        }
    }
    if (word.size() >= 2 && STOP_WORDS.find(word) == STOP_WORDS.end()) {
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

        // Strategy 4: Keyword match on labels
        for (const auto& kw : keywords) {
            if (lower_label.find(kw) != std::string::npos) {
                score += 3.0;
            }
        }

        // Strategy 5: Keyword match on definitions
        for (const auto& kw : keywords) {
            if (lower_def.find(kw) != std::string::npos) {
                score += 1.5;
            }
        }

        // Strategy 6: Fuzzy match — Levenshtein on label vs each keyword
        for (const auto& kw : keywords) {
            if (kw.size() >= 3 && lower_label.size() >= 3) {
                size_t dist = levenshtein(kw, lower_label);
                size_t max_len = std::max(kw.size(), lower_label.size());
                double similarity = 1.0 - (static_cast<double>(dist) / max_len);
                if (similarity >= 0.7) {
                    score += 2.0 * similarity;
                }
            }
            // Also check prefix match
            if (kw.size() >= 3 && lower_label.size() >= kw.size() &&
                lower_label.substr(0, kw.size()) == kw) {
                score += 2.5;
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

            for (const auto& lw : label_words) {
                if (lw.size() < 3) continue;
                for (const auto& kw : keywords) {
                    if (lw == kw) {
                        score += 3.5;
                    } else if (lw.size() >= 4 && kw.size() >= 4 &&
                               lw.substr(0, 4) == kw.substr(0, 4)) {
                        // Stem prefix match (e.g., "grav" matches "gravity")
                        score += 2.0;
                    }
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
    ans << "Ich kenne " << total_concepts_ << " Konzepte und " << total_relations_ << " Relationen in meinem Wissensnetz. ";
    if (!top_concepts.empty()) {
        ans << "Frag mich z.B. ueber **" << top_concepts[0].label << "**";
        if (top_concepts.size() > 1) {
            ans << " oder **" << top_concepts[1].label << "**";
        }
        ans << ".\n";
    } else {
        ans << "Frag mich etwas!\n";
    }
    return ans.str();
}

std::string ChatInterface::format_question(
    const std::vector<ConceptInfo>& top_concepts,
    const std::vector<std::string>& thought_paths
) {
    std::ostringstream ans;

    if (top_concepts.empty()) {
        ans << "Ich habe dazu kein direktes Wissen in meinem Netz.\n";
        return ans.str();
    }

    // Primary concept — detailed answer
    const auto& primary = top_concepts[0];
    ans << "**" << primary.label << "** ("
        << epistemic_type_to_string(primary.epistemic.type)
        << ", Trust: " << static_cast<int>(primary.epistemic.trust * 100) << "%)\n";
    ans << primary.definition << "\n\n";

    // Secondary concepts — brief context
    if (top_concepts.size() > 1) {
        ans << "Verwandte Konzepte:\n";
        size_t shown = 0;
        for (size_t i = 1; i < top_concepts.size() && shown < 3; ++i) {
            const auto& c = top_concepts[i];
            ans << "  - **" << c.label << "** ("
                << epistemic_type_to_string(c.epistemic.type)
                << ", " << static_cast<int>(c.epistemic.trust * 100) << "%): "
                << c.definition.substr(0, 120);
            if (c.definition.size() > 120) ans << "...";
            ans << "\n";
            ++shown;
        }
        ans << "\n";
    }

    // Thought paths — top 3
    if (!thought_paths.empty()) {
        ans << "Gedankenpfade:\n";
        size_t shown = 0;
        for (const auto& p : thought_paths) {
            if (shown >= 3) break;
            ans << "  " << p << "\n";
            ++shown;
        }
    }

    return ans.str();
}

std::string ChatInterface::format_statement(const std::vector<ConceptInfo>& top_concepts) {
    std::ostringstream ans;

    if (top_concepts.empty()) {
        ans << "Ich kann diese Aussage nicht in mein Wissensnetz einordnen.\n";
        return ans.str();
    }

    ans << "Das beruehrt folgende Konzepte in meinem Wissensnetz:\n\n";
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
        case QueryIntent::COMMAND: {
            std::ostringstream ans;
            ans << "Basierend auf meinem Denken (" << salient_concepts.size()
                << " aktivierte Konzepte):\n";
            ans << format_question(relevant, thought_paths_summary);
            response.answer = ans.str();
            break;
        }
        case QueryIntent::STATEMENT:
            response.answer = format_statement(relevant);
            break;
        default: {
            std::ostringstream ans;
            ans << "Basierend auf meinem Denken (" << salient_concepts.size()
                << " aktivierte Konzepte):\n";
            ans << format_question(relevant, thought_paths_summary);
            response.answer = ans.str();
            break;
        }
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

// ─── Full-Pipeline Response Formatter ────────────────────────────────────────

std::string ChatInterface::format_thinking_response(
    const std::vector<ConceptInfo>& top_concepts,
    const ThinkingContext& thinking,
    const LongTermMemory& ltm
) {
    std::ostringstream ans;

    if (top_concepts.empty()) {
        ans << "Ich habe dazu kein direktes Wissen in meinem Netz.\n";
        return ans.str();
    }

    // ── Pipeline summary header ──
    ans << "Basierend auf meinem Denken ("
        << thinking.salient_concepts.size() << " aktivierte Konzepte, "
        << thinking.steps_completed << " Pipeline-Schritte, "
        << static_cast<int>(thinking.thinking_duration_ms) << "ms):\n";

    // ── Multi-domain detection ──
    // Only flag multi-domain if there are 2+ domains with significant relevance
    // AND the top two have comparable relevance (ratio < 3:1)
    bool multi_domain = false;
    if (thinking.detected_domains.size() > 1) {
        double top = thinking.detected_domains[0].relevance;
        double second = thinking.detected_domains[1].relevance;
        multi_domain = (second > 0.1 && (top / std::max(second, 0.01)) < 3.0);
    }

    if (multi_domain) {
        // Ambiguity detected — show all domains
        ans << "\n**Mehrere Wissensbereiche erkannt:**\n";
        for (const auto& domain : thinking.detected_domains) {
            ans << "  - **" << domain.domain_name << "** ("
                << domain.concepts.size() << " Konzepte, Relevanz: "
                << static_cast<int>(domain.relevance * 100) << "%)\n";
        }
        ans << "\n";

        // Show primary concept from each domain
        for (const auto& domain : thinking.detected_domains) {
            if (domain.concepts.empty()) continue;
            ans << "**" << domain.domain_name << ":**\n";
            size_t shown = 0;
            for (auto cid : domain.concepts) {
                if (shown >= 2) break;
                auto info_opt = ltm.retrieve_concept(cid);
                if (!info_opt) continue;
                ans << "  **" << info_opt->label << "** ("
                    << epistemic_type_to_string(info_opt->epistemic.type)
                    << ", Trust: " << static_cast<int>(info_opt->epistemic.trust * 100) << "%)\n"
                    << "  " << info_opt->definition << "\n";
                ++shown;
            }
            ans << "\n";
        }
    } else {
        // Single domain — detailed answer
        const auto& primary = top_concepts[0];
        ans << "**" << primary.label << "** ("
            << epistemic_type_to_string(primary.epistemic.type)
            << ", Trust: " << static_cast<int>(primary.epistemic.trust * 100) << "%)\n";
        ans << primary.definition << "\n\n";

        // Related concepts
        if (top_concepts.size() > 1) {
            ans << "Verwandte Konzepte:\n";
            size_t shown = 0;
            for (size_t i = 1; i < top_concepts.size() && shown < 3; ++i) {
                const auto& c = top_concepts[i];
                ans << "  - **" << c.label << "** ("
                    << epistemic_type_to_string(c.epistemic.type)
                    << ", " << static_cast<int>(c.epistemic.trust * 100) << "%): "
                    << c.definition.substr(0, 120);
                if (c.definition.size() > 120) ans << "...";
                ans << "\n";
                ++shown;
            }
            ans << "\n";
        }
    }

    // ── KAN-Relations between salient concepts ──
    if (!thinking.relation_links.empty()) {
        ans << "KAN-Relationen:\n";
        size_t shown = 0;
        for (const auto& rl : thinking.relation_links) {
            if (shown >= 8) break;
            ans << "  " << rl.source_label << " --[" << rl.relation_name
                << "]--> " << rl.target_label;
            if (rl.weight < 1.0) {
                ans << " (w=" << static_cast<int>(rl.weight * 100) << "%)";
            }
            ans << "\n";
            ++shown;
        }
        ans << "\n";
    }

    // ── MiniLLM meaning insights ──
    if (!thinking.meaning_insights.empty()) {
        ans << "Semantische Analyse (MiniLLMs):\n";
        size_t shown = 0;
        for (const auto& insight : thinking.meaning_insights) {
            if (shown >= 5) break;
            ans << "  - " << insight.interpretation
                << " (Konfidenz: " << static_cast<int>(insight.confidence * 100) << "%, "
                << insight.source_model << ")\n";
            ++shown;
        }
        ans << "\n";
    }

    // ── Hypotheses (with KAN validation status) ──
    if (!thinking.hypothesis_insights.empty()) {
        ans << "Hypothesen:\n";
        size_t shown = 0;
        for (const auto& hyp : thinking.hypothesis_insights) {
            if (shown >= 3) break;
            ans << "  - " << hyp.statement
                << " (Konfidenz: " << static_cast<int>(hyp.confidence * 100) << "%";
            if (hyp.kan_validated) {
                ans << ", KAN: " << hyp.validation_status;
            }
            ans << ")\n";
            ++shown;
        }
        ans << "\n";
    }

    // ── Contradictions ──
    if (!thinking.contradiction_notes.empty()) {
        ans << "Widersprueche erkannt:\n";
        for (const auto& c : thinking.contradiction_notes) {
            ans << "  - " << c.description
                << " (Schwere: " << static_cast<int>(c.severity * 100) << "%)\n";
        }
        ans << "\n";
    }

    // ── Thought paths ──
    if (!thinking.thought_path_summaries.empty()) {
        ans << "Gedankenpfade:\n";
        size_t shown = 0;
        for (const auto& p : thinking.thought_path_summaries) {
            if (shown >= 3) break;
            ans << "  " << p << "\n";
            ++shown;
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
        return "Konzept nicht gefunden.";
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
        return "Ein oder beide Konzepte nicht gefunden.";
    }

    const ConceptInfo& info1 = info1_opt.value();
    const ConceptInfo& info2 = info2_opt.value();

    std::ostringstream out;
    out << "=== Vergleich ===\n\n";
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
        out << "Keine " << epistemic_type_to_string(type) << " vorhanden.\n";
        return out.str();
    }

    out << "Gefunden: " << ids.size() << " Konzept(e)\n\n";

    for (auto id : ids) {
        auto info_opt = ltm.retrieve_concept(id);
        if (info_opt.has_value()) {
            const ConceptInfo& info = info_opt.value();
            out << "• " << info.label;
            out << " (Trust: " << (info.epistemic.trust * 100.0) << "%)";

            if (info.epistemic.status == EpistemicStatus::INVALIDATED) {
                out << " INVALIDIERT";
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

    out << "=== Brain19 - Wissensuebersicht ===\n\n";
    out << "Gesamt: " << all.size() << " aktive Konzepte\n\n";

    out << "Nach Typ:\n";
    out << "  Fakten:        " << facts.size() << "\n";
    out << "  Theorien:      " << theories.size() << "\n";
    out << "  Hypothesen:    " << hypotheses.size() << "\n";
    out << "  Spekulationen: " << specs.size() << "\n\n";

    out << "Sprachausgabe: Template-Engine (kein LLM)\n";

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
        out << "\n" << invalidated_count << " invalidierte(s) Konzept(e)\n";
    }

    return out.str();
}

} // namespace brain19
