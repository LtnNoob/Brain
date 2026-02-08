#pragma once

#include "../ltm/long_term_memory.hpp"
#include "../memory/brain_controller.hpp"
#include "ollama_client.hpp"
#include <string>
#include <vector>
#include <memory>

namespace brain19 {

// ChatResponse: Response from Brain19
struct ChatResponse {
    std::string answer;
    std::vector<ConceptId> referenced_concepts;
    bool contains_speculation;
    std::string epistemic_note;
    bool used_llm;
    double llm_time_ms;
};

// ChatInterface: LLM-powered verbalization of Brain19 knowledge
//
// WICHTIG: LLM ist ein TOOL, kein Agent!
// - LLM verbalisiert NUR vorhandenes LTM-Wissen
// - LLM hat read-only Zugriff
// - LLM kann NICHT modifizieren
// - Epistemic metadata wird in Prompt eingebaut
// - LLM MUSS epistemic metadata in Antworten einbauen
class ChatInterface {
public:
    ChatInterface();
    ~ChatInterface();
    
    // Initialize with Ollama
    bool initialize(const OllamaConfig& config);
    
    // Check if LLM is available
    bool is_llm_available() const;
    
    // Ask a question (with LLM)
    ChatResponse ask(
        const std::string& question,
        const LongTermMemory& ltm
    );
    
    // List all knowledge of specific type
    std::string list_knowledge(
        const LongTermMemory& ltm,
        EpistemicType type
    );
    
    // Explain a concept (with LLM)
    std::string explain_concept(
        ConceptId id,
        const LongTermMemory& ltm
    );
    
    // Compare two concepts (with LLM)
    std::string compare(
        ConceptId id1,
        ConceptId id2,
        const LongTermMemory& ltm
    );
    
    // Get knowledge summary
    std::string get_summary(const LongTermMemory& ltm);
    
private:
    // Build epistemic context for LLM
    std::string build_epistemic_context(
        const std::vector<ConceptInfo>& concepts
    );
    
    // Build system prompt with epistemic rules
    std::string build_system_prompt();
    
    // Search for relevant concepts
    std::vector<ConceptInfo> find_relevant_concepts(
        const std::string& question,
        const LongTermMemory& ltm
    );
    
    std::unique_ptr<OllamaClient> ollama_;
    bool llm_available_;
};

} // namespace brain19
