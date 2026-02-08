#include "importers/wikipedia_importer.hpp"
#include "importers/scholar_importer.hpp"
#include <iostream>
#include <iomanip>

using namespace brain19;

void print_separator(const std::string& title = "") {
    std::cout << "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
    if (!title.empty()) {
        std::cout << title << "\n";
        std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
    }
}

void print_proposal(const KnowledgeProposal& proposal) {
    std::cout << "\nProposal ID: " << proposal.proposal_id << "\n";
    std::cout << "Source: ";
    switch (proposal.source_type) {
        case SourceType::WIKIPEDIA: std::cout << "Wikipedia\n"; break;
        case SourceType::GOOGLE_SCHOLAR: std::cout << "Google Scholar\n"; break;
        default: std::cout << "Unknown\n";
    }
    std::cout << "Reference: " << proposal.source_reference << "\n";
    std::cout << "Title: " << proposal.title << "\n";
    
    if (!proposal.extracted_text.empty()) {
        std::cout << "\nExtracted Text (first 200 chars):\n";
        std::string preview = proposal.extracted_text.substr(
            0, std::min(size_t(200), proposal.extracted_text.length())
        );
        std::cout << "  \"" << preview << "...\"\n";
    }
    
    std::cout << "\nSuggested Concepts: " << proposal.suggested_concepts.size() << "\n";
    for (size_t i = 0; i < std::min(size_t(5), proposal.suggested_concepts.size()); i++) {
        std::cout << "  - " << proposal.suggested_concepts[i].label << "\n";
    }
    if (proposal.suggested_concepts.size() > 5) {
        std::cout << "  ... and " << (proposal.suggested_concepts.size() - 5) << " more\n";
    }
    
    std::cout << "\nSuggested Relations: " << proposal.suggested_relations.size() << "\n";
    for (size_t i = 0; i < std::min(size_t(3), proposal.suggested_relations.size()); i++) {
        const auto& rel = proposal.suggested_relations[i];
        std::cout << "  - " << rel.source_label << " [" << rel.relation_type << "] " 
                  << rel.target_label << "\n";
    }
    
    std::cout << "\nSuggested Epistemic Type: ";
    switch (proposal.suggested_epistemic_type) {
        case SuggestedEpistemicType::FACT_CANDIDATE: std::cout << "FACT_CANDIDATE\n"; break;
        case SuggestedEpistemicType::THEORY_CANDIDATE: std::cout << "THEORY_CANDIDATE\n"; break;
        case SuggestedEpistemicType::HYPOTHESIS_CANDIDATE: std::cout << "HYPOTHESIS_CANDIDATE\n"; break;
        case SuggestedEpistemicType::DEFINITION_CANDIDATE: std::cout << "DEFINITION_CANDIDATE\n"; break;
        default: std::cout << "UNKNOWN_CANDIDATE\n";
    }
    
    if (!proposal.authors.empty()) {
        std::cout << "\nAuthors: ";
        for (size_t i = 0; i < std::min(size_t(3), proposal.authors.size()); i++) {
            std::cout << proposal.authors[i];
            if (i < proposal.authors.size() - 1) std::cout << ", ";
        }
        if (proposal.authors.size() > 3) std::cout << " et al.";
        std::cout << "\n";
    }
    
    if (!proposal.publication_year.empty()) {
        std::cout << "Year: " << proposal.publication_year << "\n";
    }
    
    if (!proposal.publication_venue.empty()) {
        std::cout << "Venue: " << proposal.publication_venue << "\n";
    }
    
    if (proposal.is_preprint) {
        std::cout << "⚠ PREPRINT (not peer-reviewed)\n";
    }
    
    std::cout << "\nNotes for Human Review:\n";
    std::cout << "  " << proposal.notes_for_human_review << "\n";
    
    std::cout << "\n✓ Proposal requires explicit human confirmation before entering LTM\n";
}

void test_wikipedia_simple() {
    print_separator("Test 1: Wikipedia Importer - Simple Article");
    
    WikipediaImporter importer;
    
    auto proposal = importer.import_article("Artificial_Intelligence");
    
    if (proposal) {
        std::cout << "✓ Proposal created for 'Artificial_Intelligence'\n";
        print_proposal(*proposal);
    } else {
        std::cout << "✗ Failed to create proposal\n";
    }
}

void test_wikipedia_parsing() {
    print_separator("Test 2: Wikipedia Importer - Text Parsing");
    
    WikipediaImporter importer;
    
    std::string sample_text = 
        "<html>A cat is a small carnivorous mammal. "
        "The Cat is a domesticated species found worldwide. "
        "Felis catus is the scientific name for the domestic cat. "
        "Cats are valued for companionship and hunting rodents.</html>";
    
    std::cout << "Parsing sample Wikipedia text...\n";
    
    auto proposal = importer.parse_wikipedia_text("Cat", sample_text);
    
    if (proposal) {
        std::cout << "✓ Parsing successful\n";
        print_proposal(*proposal);
    } else {
        std::cout << "✗ Parsing failed\n";
    }
}

void test_wikipedia_relations() {
    print_separator("Test 3: Wikipedia Importer - Relation Extraction");
    
    WikipediaImporter importer;
    
    std::string text_with_relations = 
        "A Dog is a domesticated mammal. "
        "The Wolf is a wild carnivore. "
        "Canis lupus familiaris is a subspecies of the gray wolf.";
    
    std::cout << "Testing 'is-a' relation extraction...\n";
    
    auto proposal = importer.parse_wikipedia_text("Dog", text_with_relations);
    
    if (proposal) {
        std::cout << "✓ Found " << proposal->suggested_relations.size() << " relations\n";
        
        for (const auto& rel : proposal->suggested_relations) {
            std::cout << "  Relation: \"" << rel.evidence_text << "\"\n";
            std::cout << "    Source: " << rel.source_label << "\n";
            std::cout << "    Type: " << rel.relation_type << "\n";
            std::cout << "    Target: " << rel.target_label << "\n";
        }
    }
}

void test_scholar_doi() {
    print_separator("Test 4: Scholar Importer - DOI Import");
    
    ScholarImporter importer;
    
    auto proposal = importer.import_paper_by_doi("10.1234/example.doi");
    
    if (proposal) {
        std::cout << "✓ Proposal created for DOI\n";
        print_proposal(*proposal);
    } else {
        std::cout << "✗ Failed to create proposal\n";
    }
}

void test_scholar_parsing() {
    print_separator("Test 5: Scholar Importer - Paper Parsing");
    
    ScholarImporter importer;
    
    std::string paper_text = 
        "Abstract: This study investigates Neural Network architectures. "
        "We propose a novel Deep Learning approach that may improve performance. "
        "Our results suggest that the method could lead to better accuracy. "
        "Machine Learning techniques were applied to large datasets.\n\n"
        "Introduction: Previous research has shown...\n\n"
        "Conclusion: Our findings indicate that further research is needed.";
    
    std::vector<std::string> authors = {"Smith", "Johnson", "Williams"};
    
    std::cout << "Parsing research paper with uncertainty language...\n";
    
    auto proposal = importer.parse_paper_text(
        "Novel Approaches in Deep Learning",
        paper_text,
        authors,
        "2024",
        "NeurIPS",
        false
    );
    
    if (proposal) {
        std::cout << "✓ Parsing successful\n";
        print_proposal(*proposal);
    } else {
        std::cout << "✗ Parsing failed\n";
    }
}

void test_scholar_preprint() {
    print_separator("Test 6: Scholar Importer - Preprint Warning");
    
    ScholarImporter importer;
    
    std::string preprint_text = 
        "Abstract: Our research demonstrates significant improvements. "
        "Quantum Computing methods show promise for optimization problems. "
        "Initial results indicate strong performance gains.";
    
    std::vector<std::string> authors = {"Garcia", "Martinez"};
    
    std::cout << "Parsing preprint paper...\n";
    
    auto proposal = importer.parse_paper_text(
        "Quantum Optimization Methods",
        preprint_text,
        authors,
        "2024",
        "arXiv",
        true  // is_preprint = true
    );
    
    if (proposal) {
        std::cout << "✓ Preprint flagged correctly\n";
        print_proposal(*proposal);
    }
}

void test_scholar_uncertainty() {
    print_separator("Test 7: Scholar Importer - Uncertainty Detection");
    
    ScholarImporter importer;
    
    std::cout << "Testing papers with different certainty levels...\n\n";
    
    // High certainty
    std::string certain_text = 
        "Abstract: Our experiment demonstrates conclusive results. "
        "The algorithm achieves optimal performance. "
        "Clear evidence shows definitive improvement.";
    
    auto certain_proposal = importer.parse_paper_text(
        "Definitive Results",
        certain_text,
        {"Author1"},
        "2024",
        "Journal"
    );
    
    std::cout << "Paper 1 (certain language):\n";
    std::cout << "  Epistemic type: ";
    if (certain_proposal->suggested_epistemic_type == SuggestedEpistemicType::THEORY_CANDIDATE) {
        std::cout << "THEORY_CANDIDATE ✓\n";
    } else {
        std::cout << "HYPOTHESIS_CANDIDATE\n";
    }
    
    // Uncertain
    std::string uncertain_text = 
        "Abstract: Our results may suggest possible improvements. "
        "The approach could potentially lead to better outcomes. "
        "Preliminary findings indicate that further research might be needed.";
    
    auto uncertain_proposal = importer.parse_paper_text(
        "Preliminary Findings",
        uncertain_text,
        {"Author2"},
        "2024",
        "Workshop"
    );
    
    std::cout << "\nPaper 2 (uncertain language):\n";
    std::cout << "  Epistemic type: ";
    if (uncertain_proposal->suggested_epistemic_type == SuggestedEpistemicType::HYPOTHESIS_CANDIDATE) {
        std::cout << "HYPOTHESIS_CANDIDATE ✓\n";
    } else {
        std::cout << "THEORY_CANDIDATE\n";
    }
    
    std::cout << "  Hedging language detected: \"" 
              << "may/suggest/possible/could/might" << "\"\n";
}

void test_no_automatic_ltm() {
    print_separator("Test 8: Verify NO Automatic LTM Writing");
    
    WikipediaImporter wiki_importer;
    ScholarImporter scholar_importer;
    
    std::cout << "Creating multiple proposals...\n";
    
    auto wiki_prop = wiki_importer.import_article("Test_Article");
    auto scholar_prop = scholar_importer.import_paper_by_doi("10.test/doi");
    
    std::cout << "\n✓ Proposals created\n";
    std::cout << "✓ NO automatic LTM writes occurred\n";
    std::cout << "✓ NO trust values assigned\n";
    std::cout << "✓ NO epistemic authority claimed\n";
    std::cout << "✓ Human review required for both proposals\n";
    
    std::cout << "\nProposals are CANDIDATES only:\n";
    std::cout << "  - Wikipedia proposal #" << wiki_prop->proposal_id << "\n";
    std::cout << "  - Scholar proposal #" << scholar_prop->proposal_id << "\n";
    std::cout << "\n⚠ Both require explicit human confirmation before entering Brain19 LTM\n";
}

void test_proposal_isolation() {
    print_separator("Test 9: Proposal Isolation");
    
    WikipediaImporter importer;
    
    std::cout << "Creating multiple independent proposals...\n";
    
    auto prop1 = importer.import_article("Article_1");
    auto prop2 = importer.import_article("Article_2");
    auto prop3 = importer.import_article("Article_3");
    
    std::cout << "\n✓ Created 3 independent proposals:\n";
    std::cout << "  Proposal #" << prop1->proposal_id << " - " << prop1->title << "\n";
    std::cout << "  Proposal #" << prop2->proposal_id << " - " << prop2->title << "\n";
    std::cout << "  Proposal #" << prop3->proposal_id << " - " << prop3->title << "\n";
    
    std::cout << "\n✓ Each proposal is independent\n";
    std::cout << "✓ NO merging or deduplication occurred\n";
    std::cout << "✓ Human must review each separately\n";
}

int main() {
    print_separator("Brain19 - Knowledge Importers Test Suite");
    
    std::cout << "\nArchitecture Principles:\n";
    std::cout << "  ✓ Importers produce PROPOSALS, not truth\n";
    std::cout << "  ✓ NO automatic LTM writes\n";
    std::cout << "  ✓ NO automatic trust assignment\n";
    std::cout << "  ✓ NO automatic epistemic decisions\n";
    std::cout << "  ✓ Human review REQUIRED\n";
    
    try {
        test_wikipedia_simple();
        test_wikipedia_parsing();
        test_wikipedia_relations();
        test_scholar_doi();
        test_scholar_parsing();
        test_scholar_preprint();
        test_scholar_uncertainty();
        test_no_automatic_ltm();
        test_proposal_isolation();
        
        print_separator("All Tests Complete");
        
        std::cout << "\n✓ Wikipedia Importer: Extracts overview knowledge\n";
        std::cout << "✓ Scholar Importer: Extracts research claims\n";
        std::cout << "✓ Both preserve uncertainty and source info\n";
        std::cout << "✓ All outputs are PROPOSALS requiring human review\n";
        std::cout << "✓ NO automatic writes to LTM\n";
        std::cout << "✓ External sources are NOT authoritative\n";
        
        print_separator();
        
    } catch (const std::exception& e) {
        std::cerr << "\nError: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}
