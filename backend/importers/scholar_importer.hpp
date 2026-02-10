#pragma once

#include "knowledge_proposal.hpp"
#include <string>
#include <memory>

namespace brain19 {

// Paper section type
enum class PaperSection {
    ABSTRACT,
    BODY,
    CONCLUSION,
    UNKNOWN
};

// ScholarImporter: Extracts research-level knowledge from papers
// DOES NOT write to LTM, only produces proposals
class ScholarImporter {
public:
    ScholarImporter();
    ~ScholarImporter();
    
    // Import from DOI (stub, no HTTP)
    std::unique_ptr<KnowledgeProposal> import_paper_by_doi(
        const std::string& doi
    );
    
    // Import from DOI via Semantic Scholar API
    std::unique_ptr<KnowledgeProposal> import_paper_by_doi_online(
        const std::string& doi
    );
    
    // Search papers via Semantic Scholar API
    std::vector<std::unique_ptr<KnowledgeProposal>> search_papers(
        const std::string& query,
        int limit = 5
    );
    
    // Import from URL
    std::unique_ptr<KnowledgeProposal> import_paper_from_url(
        const std::string& url
    );
    
    // Parse paper text (offline)
    std::unique_ptr<KnowledgeProposal> parse_paper_text(
        const std::string& title,
        const std::string& text,
        const std::vector<std::string>& authors = {},
        const std::string& year = "",
        const std::string& venue = "",
        bool is_preprint = false
    );
    
private:
    uint64_t next_proposal_id_;
    
    // Parsing helpers
    std::string extract_abstract(const std::string& text) const;
    std::string extract_conclusion(const std::string& text) const;
    std::vector<SuggestedConcept> extract_research_concepts(const std::string& text) const;
    std::vector<SuggestedRelation> extract_claims(const std::string& text) const;
    std::vector<std::string> extract_methods(const std::string& text) const;
    std::vector<std::string> extract_citations(const std::string& text) const;
    
    // Uncertainty preservation
    bool contains_uncertainty_language(const std::string& text) const;
    std::string extract_hedging_language(const std::string& text) const;
};

} // namespace brain19
