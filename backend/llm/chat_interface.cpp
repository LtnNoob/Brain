#include "chat_interface.hpp"
#include <sstream>
#include <algorithm>
#include <iostream>

namespace brain19 {

ChatInterface::ChatInterface()
    : ollama_(std::make_unique<OllamaClient>())
    , llm_available_(false)
{
}

ChatInterface::~ChatInterface() = default;

bool ChatInterface::initialize(const OllamaConfig& config) {
    if (!ollama_->initialize(config)) {
        std::cerr << "Warning: Ollama not available. Falling back to simple mode.\n";
        llm_available_ = false;
        return false;
    }
    
    llm_available_ = true;
    std::cout << "✓ Ollama LLM initialized: " << config.model << "\n";
    return true;
}

bool ChatInterface::is_llm_available() const {
    return llm_available_;
}

std::string ChatInterface::build_system_prompt() {
    return R"(You are Brain19's verbalization interface. Your role is to communicate knowledge stored in Long-Term Memory (LTM) with ABSOLUTE EPISTEMIC RIGOR.

CRITICAL RULES:
1. You ONLY verbalize knowledge that exists in LTM
2. You ALWAYS include epistemic metadata in responses
3. You NEVER invent or infer information not in LTM
4. You MUST distinguish facts from speculation

EPISTEMIC TYPES:
- FACT: Verified, reproducible knowledge (high certainty)
- THEORY: Well-supported, falsifiable scientific knowledge
- HYPOTHESIS: Testable claim needing more evidence
- SPECULATION: Exploratory idea with low certainty

RESPONSE FORMAT:
When discussing knowledge, ALWAYS state:
1. The concept name
2. Its epistemic type
3. Its trust level (certainty)
4. The actual information

Example: "According to my knowledge, cats (FACT, 98% certainty) are small domesticated mammals..."

WARNINGS:
- If knowledge is SPECULATION or HYPOTHESIS, say: "⚠️ This is speculative/hypothetical"
- If knowledge is INVALIDATED, say: "⚠️ This knowledge is known to be incorrect"
- If trust < 50%, say: "⚠️ Low certainty information"

If you don't have knowledge about something, say so clearly.
Be concise but informative. Use German language.)";
}

std::vector<ConceptInfo> ChatInterface::find_relevant_concepts(
    const std::string& question,
    const LongTermMemory& ltm
) {
    std::vector<ConceptInfo> relevant;
    auto all_ids = ltm.get_active_concepts();
    
    // Simple keyword matching (could be improved with TF-IDF, etc.)
    std::string lower_q = question;
    std::transform(lower_q.begin(), lower_q.end(), lower_q.begin(), ::tolower);
    
    for (auto id : all_ids) {
        auto info_opt = ltm.retrieve_concept(id);
        if (info_opt.has_value()) {
            const ConceptInfo& info = info_opt.value();
            
            // Check if label or definition contains keywords
            std::string lower_label = info.label;
            std::string lower_def = info.definition;
            std::transform(lower_label.begin(), lower_label.end(), 
                          lower_label.begin(), ::tolower);
            std::transform(lower_def.begin(), lower_def.end(), 
                          lower_def.begin(), ::tolower);
            
            if (lower_q.find(lower_label) != std::string::npos ||
                lower_label.find(lower_q) != std::string::npos ||
                lower_def.find(lower_q) != std::string::npos) {
                relevant.push_back(info);
            }
        }
    }
    
    return relevant;
}

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

ChatResponse ChatInterface::ask(
    const std::string& question,
    const LongTermMemory& ltm
) {
    ChatResponse response;
    response.used_llm = false;
    
    // Find relevant concepts
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

    // If no LLM, use simple fallback
    if (!llm_available_) {
        if (relevant.empty()) {
            response.answer = "Ich habe kein Wissen darüber in meiner LTM.\n";
        } else {
            std::ostringstream ans;
            ans << "Basierend auf meinem Wissen:\n\n";
            for (const auto& info : relevant) {
                ans << "**" << info.label << "** (" << epistemic_type_to_string(info.epistemic.type);
                ans << ", Trust: " << (info.epistemic.trust * 100.0) << "%)\n";
                ans << info.definition << "\n\n";
            }
            response.answer = ans.str();
        }
        return response;
    }
    
    // Use LLM
    auto start = std::chrono::high_resolution_clock::now();
    
    try {
        // Build epistemic context
        std::string context = build_epistemic_context(relevant);
        
        // Build messages
        std::vector<OllamaMessage> messages;
        messages.push_back({"system", build_system_prompt()});
        messages.push_back({"user", context + "\n\nFrage: " + question});
        
        // Get LLM response
        auto llm_resp = ollama_->chat(messages);
        
        auto end = std::chrono::high_resolution_clock::now();
        response.llm_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        
        if (llm_resp.success) {
            response.answer = llm_resp.content;
            response.used_llm = true;
        } else {
            response.answer = "LLM Error: " + llm_resp.error_message;
        }
        
    } catch (const std::exception& e) {
        response.answer = std::string("Exception: ") + e.what();
    }
    
    return response;
}

ChatResponse ChatInterface::ask_with_context(
    const std::string& question,
    const LongTermMemory& ltm,
    const std::vector<ConceptId>& salient_concepts,
    const std::vector<std::string>& thought_paths_summary
) {
    ChatResponse response;
    response.used_llm = false;
    response.contains_speculation = false;

    // Gather concepts from thinking context
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

    // Also add any keyword-matched concepts not already included
    auto keyword_matches = find_relevant_concepts(question, ltm);
    for (const auto& info : keyword_matches) {
        bool already = false;
        for (const auto& r : relevant) {
            if (r.id == info.id) { already = true; break; }
        }
        if (!already) {
            relevant.push_back(info);
            response.referenced_concepts.push_back(info.id);
        }
    }

    // If no LLM, use simple fallback
    if (!llm_available_) {
        if (relevant.empty()) {
            response.answer = "Ich habe kein Wissen darüber in meiner LTM.\n";
        } else {
            std::ostringstream ans;
            ans << "Basierend auf meinem Denken (" << salient_concepts.size() << " aktivierte Konzepte):\n\n";
            for (const auto& info : relevant) {
                ans << "**" << info.label << "** (" << epistemic_type_to_string(info.epistemic.type);
                ans << ", Trust: " << (info.epistemic.trust * 100.0) << "%)\n";
                ans << info.definition << "\n\n";
            }
            if (!thought_paths_summary.empty()) {
                ans << "Gedankenpfade:\n";
                for (const auto& p : thought_paths_summary) {
                    ans << "  • " << p << "\n";
                }
            }
            response.answer = ans.str();
        }
        return response;
    }

    // Use LLM with thinking context
    auto start = std::chrono::high_resolution_clock::now();
    try {
        std::string context = build_epistemic_context(relevant);
        if (!thought_paths_summary.empty()) {
            context += "\nTHOUGHT PATHS (from cognitive dynamics):\n";
            for (const auto& p : thought_paths_summary) {
                context += "  - " + p + "\n";
            }
        }

        std::vector<OllamaMessage> messages;
        messages.push_back({"system", build_system_prompt()});
        messages.push_back({"user", context + "\n\nFrage: " + question});

        auto llm_resp = ollama_->chat(messages);
        auto end = std::chrono::high_resolution_clock::now();
        response.llm_time_ms = std::chrono::duration<double, std::milli>(end - start).count();

        if (llm_resp.success) {
            response.answer = llm_resp.content;
            response.used_llm = true;
        } else {
            response.answer = "LLM Error: " + llm_resp.error_message;
        }
    } catch (const std::exception& e) {
        response.answer = std::string("Exception: ") + e.what();
    }

    return response;
}

std::string ChatInterface::explain_concept(
    ConceptId id,
    const LongTermMemory& ltm
) {
    auto info_opt = ltm.retrieve_concept(id);
    if (!info_opt.has_value()) {
        return "Konzept nicht gefunden.";
    }
    
    const ConceptInfo& info = info_opt.value();
    
    // Without LLM: structured output
    if (!llm_available_) {
        std::ostringstream out;
        out << "╔═══════════════════════════════════════════╗\n";
        out << "║  " << info.label << "\n";
        out << "╚═══════════════════════════════════════════╝\n\n";
        out << "Type: " << epistemic_type_to_string(info.epistemic.type) << "\n";
        out << "Status: " << epistemic_status_to_string(info.epistemic.status) << "\n";
        out << "Trust: " << (info.epistemic.trust * 100.0) << "%\n\n";
        out << info.definition << "\n";
        return out.str();
    }
    
    // With LLM: natural explanation
    try {
        std::string context = build_epistemic_context({info});
        
        std::vector<OllamaMessage> messages;
        messages.push_back({"system", build_system_prompt()});
        messages.push_back({"user", context + "\n\nBitte erkläre dieses Konzept ausführlich, inklusive epistemischer Einordnung."});
        
        auto resp = ollama_->chat(messages);
        return resp.success ? resp.content : ("Error: " + resp.error_message);
        
    } catch (const std::exception& e) {
        return std::string("Exception: ") + e.what();
    }
}

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
    
    if (!llm_available_) {
        std::ostringstream out;
        out << "=== Vergleich ===\n\n";
        out << "1. " << info1.label << " (" << epistemic_type_to_string(info1.epistemic.type);
        out << ", " << (info1.epistemic.trust * 100.0) << "%)\n\n";
        out << "2. " << info2.label << " (" << epistemic_type_to_string(info2.epistemic.type);
        out << ", " << (info2.epistemic.trust * 100.0) << "%)\n";
        return out.str();
    }
    
    // With LLM
    try {
        std::string context = build_epistemic_context({info1, info2});
        
        std::vector<OllamaMessage> messages;
        messages.push_back({"system", build_system_prompt()});
        messages.push_back({"user", context + "\n\nVergleiche diese beiden Konzepte epistemisch."});
        
        auto resp = ollama_->chat(messages);
        return resp.success ? resp.content : ("Error: " + resp.error_message);
        
    } catch (const std::exception& e) {
        return std::string("Exception: ") + e.what();
    }
}

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
                out << " ⚠️ INVALIDIERT";
            }
            
            out << "\n  ID: " << id << "\n";
        }
    }
    
    return out.str();
}

std::string ChatInterface::get_summary(const LongTermMemory& ltm) {
    auto facts = ltm.get_concepts_by_type(EpistemicType::FACT);
    auto theories = ltm.get_concepts_by_type(EpistemicType::THEORY);
    auto hypotheses = ltm.get_concepts_by_type(EpistemicType::HYPOTHESIS);
    auto specs = ltm.get_concepts_by_type(EpistemicType::SPECULATION);
    auto all = ltm.get_active_concepts();
    
    std::ostringstream out;
    
    out << "╔═══════════════════════════════════════════╗\n";
    out << "║  Brain19 - Wissensübersicht               ║\n";
    out << "╚═══════════════════════════════════════════╝\n\n";
    
    out << "Gesamt: " << all.size() << " aktive Konzepte\n\n";
    
    out << "Nach Typ:\n";
    out << "  • Fakten:       " << facts.size() << "\n";
    out << "  • Theorien:     " << theories.size() << "\n";
    out << "  • Hypothesen:   " << hypotheses.size() << "\n";
    out << "  • Spekulationen: " << specs.size() << "\n\n";
    
    if (llm_available_) {
        out << "LLM: ✓ Aktiv (" << ollama_->get_config().model << ")\n";
    } else {
        out << "LLM: ✗ Nicht verfügbar (Fallback-Modus)\n";
    }
    
    // Check for invalidated
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
        out << "\n⚠️ " << invalidated_count << " invalidierte(s) Konzept(e)\n";
    }
    
    return out.str();
}

} // namespace brain19
