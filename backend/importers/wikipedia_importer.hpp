#pragma once

#include "knowledge_proposal.hpp"
#include <string>
#include <memory>

namespace brain19 {

// WikipediaImporter: Extracts structured overview from Wikipedia
// DOES NOT write to LTM, only produces proposals
class WikipediaImporter {
public:
    WikipediaImporter();
    ~WikipediaImporter();
    
    // Extract knowledge proposal from Wikipedia article
    // Returns nullptr if extraction fails
    std::unique_ptr<KnowledgeProposal> import_article(
        const std::string& article_title
    );
    
    // Extract from URL (for testing with local files)
    std::unique_ptr<KnowledgeProposal> import_from_url(
        const std::string& url
    );
    
    // Parse Wikipedia HTML/text (offline)
    std::unique_ptr<KnowledgeProposal> parse_wikipedia_text(
        const std::string& title,
        const std::string& html_or_text
    );
    
private:
    uint64_t next_proposal_id_;
    
    // Parsing helpers
    std::string extract_lead_section(const std::string& text) const;
    std::vector<SuggestedConcept> extract_concepts(const std::string& text) const;
    std::vector<SuggestedRelation> extract_basic_relations(const std::string& text) const;
    std::vector<std::string> extract_references(const std::string& text) const;
    
    // Text cleaning
    std::string remove_html_tags(const std::string& html) const;
    std::string normalize_whitespace(const std::string& text) const;
};

} // namespace brain19
