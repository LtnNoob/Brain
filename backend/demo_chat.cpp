#include "ltm/long_term_memory.hpp"
#include "memory/brain_controller.hpp"
#include "llm/chat_interface.hpp"
#include <iostream>
#include <string>

using namespace brain19;

void print_banner() {
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════╗\n";
    std::cout << "║          Brain19 - Interaktiver Chat                    ║\n";
    std::cout << "║     Lokales Wissens-System mit epistemischer Awareness   ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════╝\n";
    std::cout << "\n";
}

void print_help() {
    std::cout << "Befehle:\n";
    std::cout << "  help          - Diese Hilfe\n";
    std::cout << "  summary       - Wissensübersicht\n";
    std::cout << "  list facts    - Alle Fakten auflisten\n";
    std::cout << "  list theories - Alle Theorien auflisten\n";
    std::cout << "  list hypo     - Alle Hypothesen auflisten\n";
    std::cout << "  list spec     - Alle Spekulationen auflisten\n";
    std::cout << "  explain <id>  - Konzept erklären\n";
    std::cout << "  compare <id1> <id2> - Zwei Konzepte vergleichen\n";
    std::cout << "  quit          - Beenden\n";
    std::cout << "\n";
    std::cout << "Oder stelle eine Frage über dein Wissen!\n\n";
}

void setup_demo_knowledge(LongTermMemory& ltm) {
    std::cout << "Lade Demo-Wissen...\n\n";
    
    // Facts
    ltm.store_concept(
        "Katze",
        "Ein kleines fleischfressendes Säugetier (Felis catus), seit Jahrtausenden domestiziert",
        EpistemicMetadata(EpistemicType::FACT, EpistemicStatus::ACTIVE, 0.98)
    );
    
    ltm.store_concept(
        "Erde",
        "Der dritte Planet von der Sonne, Heimat des Lebens",
        EpistemicMetadata(EpistemicType::FACT, EpistemicStatus::ACTIVE, 0.99)
    );
    
    // Theories
    ltm.store_concept(
        "Evolution",
        "Wissenschaftliche Theorie über biologische Veränderung durch natürliche Selektion",
        EpistemicMetadata(EpistemicType::THEORY, EpistemicStatus::ACTIVE, 0.95)
    );
    
    ltm.store_concept(
        "Quantenmechanik",
        "Fundamentale Theorie der Physik über Materie und Energie auf atomarer Ebene",
        EpistemicMetadata(EpistemicType::THEORY, EpistemicStatus::ACTIVE, 0.95)
    );
    
    // Hypothesis
    ltm.store_concept(
        "Dunkle Materie",
        "Hypothetische Materieform, die nicht mit Licht interagiert aber gravitative Effekte hat",
        EpistemicMetadata(EpistemicType::HYPOTHESIS, EpistemicStatus::ACTIVE, 0.70)
    );
    
    // Speculation
    ltm.store_concept(
        "Multiversum",
        "Spekulative Hypothese über die Existenz multipler paralleler Universen",
        EpistemicMetadata(EpistemicType::SPECULATION, EpistemicStatus::ACTIVE, 0.30)
    );
    
    // Invalidated knowledge
    auto phlogiston_id = ltm.store_concept(
        "Phlogiston-Theorie",
        "Historische Theorie der Verbrennung, heute als falsch bekannt",
        EpistemicMetadata(EpistemicType::THEORY, EpistemicStatus::ACTIVE, 0.75)
    );
    ltm.invalidate_concept(phlogiston_id, 0.05);
    
    std::cout << "✓ 7 Konzepte geladen\n";
    std::cout << "  - 2 FAKTEN\n";
    std::cout << "  - 3 THEORIEN (1 invalidiert)\n";
    std::cout << "  - 1 HYPOTHESE\n";
    std::cout << "  - 1 SPEKULATION\n\n";
}

int main() {
    print_banner();
    
    std::cout << "Initialisiere Brain19...\n";
    
    BrainController brain;
    LongTermMemory ltm;
    ChatInterface chat;
    
    if (!brain.initialize()) {
        std::cerr << "Fehler: BrainController init fehlgeschlagen\n";
        return 1;
    }
    
    std::cout << "✓ BrainController\n";
    std::cout << "✓ LongTermMemory\n";
    
    // Initialize ChatInterface with Ollama
    OllamaConfig ollama_config;
    ollama_config.host = "http://localhost:11434";
    ollama_config.model = "llama3.2:3b";  // Default model
    ollama_config.temperature = 0.7;
    ollama_config.num_predict = 512;
    
    if (chat.initialize(ollama_config)) {
        std::cout << "✓ ChatInterface (LLM aktiv)\n\n";
    } else {
        std::cout << "⚠ ChatInterface (Fallback-Modus, kein LLM)\n";
        std::cout << "  Tipp: Starte Ollama mit 'ollama serve'\n\n";
    }
    
    setup_demo_knowledge(ltm);
    print_help();
    
    std::string input;
    bool running = true;
    
    std::cout << "Brain19> ";
    
    while (running && std::getline(std::cin, input)) {
        if (input.empty()) {
            std::cout << "Brain19> ";
            continue;
        }
        
        // Commands
        if (input == "quit" || input == "exit" || input == "q") {
            running = false;
            std::cout << "\nAuf Wiedersehen!\n";
            break;
        }
        
        if (input == "help" || input == "?") {
            print_help();
            std::cout << "Brain19> ";
            continue;
        }
        
        if (input == "summary") {
            std::cout << "\n" << chat.get_summary(ltm) << "\n";
            std::cout << "Brain19> ";
            continue;
        }
        
        if (input == "list facts") {
            std::cout << "\n" << chat.list_knowledge(ltm, EpistemicType::FACT) << "\n";
            std::cout << "Brain19> ";
            continue;
        }
        
        if (input == "list theories") {
            std::cout << "\n" << chat.list_knowledge(ltm, EpistemicType::THEORY) << "\n";
            std::cout << "Brain19> ";
            continue;
        }
        
        if (input == "list hypo") {
            std::cout << "\n" << chat.list_knowledge(ltm, EpistemicType::HYPOTHESIS) << "\n";
            std::cout << "Brain19> ";
            continue;
        }
        
        if (input == "list spec") {
            std::cout << "\n" << chat.list_knowledge(ltm, EpistemicType::SPECULATION) << "\n";
            std::cout << "Brain19> ";
            continue;
        }
        
        if (input.substr(0, 7) == "explain") {
            try {
                ConceptId id = std::stoull(input.substr(8));
                std::cout << "\n" << chat.explain_concept(id, ltm) << "\n";
            } catch (...) {
                std::cout << "\nFehler: Ungültige ID\n";
                std::cout << "Verwendung: explain <id>\n\n";
            }
            std::cout << "Brain19> ";
            continue;
        }
        
        if (input.substr(0, 7) == "compare") {
            try {
                std::string rest = input.substr(8);
                size_t space = rest.find(' ');
                ConceptId id1 = std::stoull(rest.substr(0, space));
                ConceptId id2 = std::stoull(rest.substr(space + 1));
                std::cout << "\n" << chat.compare(id1, id2, ltm) << "\n";
            } catch (...) {
                std::cout << "\nFehler: Ungültige IDs\n";
                std::cout << "Verwendung: compare <id1> <id2>\n\n";
            }
            std::cout << "Brain19> ";
            continue;
        }
        
        // Question
        std::cout << "\n";
        std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
        
        auto response = chat.ask(input, ltm);
        
        std::cout << response.answer << "\n";
        
        if (!response.epistemic_note.empty()) {
            std::cout << "\n" << response.epistemic_note << "\n";
        }
        
        if (!response.referenced_concepts.empty()) {
            std::cout << "\nReferenzierte IDs: ";
            for (size_t i = 0; i < response.referenced_concepts.size(); i++) {
                std::cout << response.referenced_concepts[i];
                if (i < response.referenced_concepts.size() - 1) {
                    std::cout << ", ";
                }
            }
            std::cout << "\n";
        }
        
        if (response.used_llm) {
            std::cout << "\n[LLM: " << static_cast<int>(response.llm_time_ms) << "ms]\n";
        }
        
        std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
        std::cout << "\nBrain19> ";
    }
    
    std::cout << "\nFahre herunter...\n";
    brain.shutdown();
    std::cout << "✓ Beendet\n\n";
    
    return 0;
}
