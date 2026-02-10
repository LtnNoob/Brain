#include "wikipedia_importer.hpp"
#include "http_client.hpp"
#include <sstream>
#include <algorithm>
#include <regex>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace brain19 {

WikipediaImporter::WikipediaImporter()
    : next_proposal_id_(1)
{
}

WikipediaImporter::~WikipediaImporter() {
}

std::unique_ptr<KnowledgeProposal> WikipediaImporter::import_article(
    const std::string& article_title
) {
    // Note: Actual HTTP fetching would go here
    // For now, this is a stub that returns a proposal structure
    
    auto proposal = std::make_unique<KnowledgeProposal>();
    proposal->proposal_id = next_proposal_id_++;
    proposal->source_type = SourceType::WIKIPEDIA;
    proposal->source_reference = "https://en.wikipedia.org/wiki/" + article_title;
    proposal->title = article_title;
    proposal->import_timestamp = std::chrono::system_clock::now();
    
    proposal->notes_for_human_review = 
        "Wikipedia article: " + article_title + 
        ". Lead section extracted. Human review required.";
    
    proposal->suggested_epistemic_type = SuggestedEpistemicType::DEFINITION_CANDIDATE;
    
    return proposal;
}

std::unique_ptr<KnowledgeProposal> WikipediaImporter::import_article(
    const std::string& article_title,
    const std::string& lang
) {
    // Build MediaWiki API URL
    std::string encoded_title = HttpClient::url_encode(article_title);
    std::string url = "https://" + lang + ".wikipedia.org/w/api.php"
        "?action=query&titles=" + encoded_title +
        "&prop=extracts&explaintext=1&format=json";

    auto resp = HttpClient::http_get(url);
    if (!resp.success) {
        auto proposal = std::make_unique<KnowledgeProposal>();
        proposal->proposal_id = next_proposal_id_++;
        proposal->source_type = SourceType::WIKIPEDIA;
        proposal->title = article_title;
        proposal->import_timestamp = std::chrono::system_clock::now();
        proposal->notes_for_human_review = "FETCH FAILED: " + resp.error;
        return proposal;
    }

    // Parse MediaWiki JSON response
    try {
        auto j = json::parse(resp.body);
        auto& pages = j["query"]["pages"];

        // Pages is an object with page IDs as keys
        for (auto& [page_id, page_data] : pages.items()) {
            if (page_id == "-1") {
                // Article not found
                auto proposal = std::make_unique<KnowledgeProposal>();
                proposal->proposal_id = next_proposal_id_++;
                proposal->source_type = SourceType::WIKIPEDIA;
                proposal->title = article_title;
                proposal->import_timestamp = std::chrono::system_clock::now();
                proposal->notes_for_human_review =
                    "Article not found on " + lang + ".wikipedia.org: " + article_title;
                return proposal;
            }

            std::string extract = page_data.value("extract", "");
            std::string actual_title = page_data.value("title", article_title);

            if (extract.empty()) {
                auto proposal = std::make_unique<KnowledgeProposal>();
                proposal->proposal_id = next_proposal_id_++;
                proposal->source_type = SourceType::WIKIPEDIA;
                proposal->title = actual_title;
                proposal->import_timestamp = std::chrono::system_clock::now();
                proposal->notes_for_human_review = "Article found but extract is empty.";
                return proposal;
            }

            // Delegate to existing parse logic
            auto proposal = parse_wikipedia_text(actual_title, extract);
            proposal->source_reference = "https://" + lang + ".wikipedia.org/wiki/" +
                HttpClient::url_encode(actual_title);
            return proposal;
        }
    } catch (const std::exception& e) {
        auto proposal = std::make_unique<KnowledgeProposal>();
        proposal->proposal_id = next_proposal_id_++;
        proposal->source_type = SourceType::WIKIPEDIA;
        proposal->title = article_title;
        proposal->import_timestamp = std::chrono::system_clock::now();
        proposal->notes_for_human_review = std::string("JSON parse error: ") + e.what();
        return proposal;
    }

    // Should not reach here
    return nullptr;
}

std::unique_ptr<KnowledgeProposal> WikipediaImporter::import_from_url(
    const std::string& url
) {
    auto proposal = std::make_unique<KnowledgeProposal>();
    proposal->proposal_id = next_proposal_id_++;
    proposal->source_type = SourceType::WIKIPEDIA;
    proposal->source_reference = url;
    proposal->import_timestamp = std::chrono::system_clock::now();
    
    return proposal;
}

std::unique_ptr<KnowledgeProposal> WikipediaImporter::parse_wikipedia_text(
    const std::string& title,
    const std::string& html_or_text
) {
    auto proposal = std::make_unique<KnowledgeProposal>();
    proposal->proposal_id = next_proposal_id_++;
    proposal->source_type = SourceType::WIKIPEDIA;
    proposal->title = title;
    proposal->source_reference = "https://en.wikipedia.org/wiki/" + title;
    proposal->import_timestamp = std::chrono::system_clock::now();
    
    // Extract lead section
    std::string clean_text = remove_html_tags(html_or_text);
    clean_text = normalize_whitespace(clean_text);
    std::string lead = extract_lead_section(clean_text);
    
    proposal->extracted_text = lead;
    
    // Extract concepts
    proposal->suggested_concepts = extract_concepts(lead);
    
    // Extract basic relations
    proposal->suggested_relations = extract_basic_relations(lead);
    
    // Set suggested epistemic type
    // CRITICAL: This is a SUGGESTION only, NOT an assignment
    // Importers MUST NOT assign actual EpistemicType
    // Human must explicitly decide during LTM ingestion
    proposal->suggested_epistemic_type = SuggestedEpistemicType::DEFINITION_CANDIDATE;
    
    // Notes for human
    proposal->notes_for_human_review = 
        "Wikipedia article: " + title + 
        ". Lead section only. Contains " + 
        std::to_string(proposal->suggested_concepts.size()) + 
        " suggested concepts. Requires human verification.";
    
    return proposal;
}

std::string WikipediaImporter::extract_lead_section(const std::string& text) const {
    // Simple heuristic: take first paragraph up to first heading or ~500 chars
    size_t max_len = std::min(text.length(), size_t(500));
    std::string lead = text.substr(0, max_len);
    
    // Stop at first heading marker (if present)
    size_t heading_pos = lead.find("\n\n");
    if (heading_pos != std::string::npos) {
        lead = lead.substr(0, heading_pos);
    }
    
    return lead;
}

std::vector<SuggestedConcept> WikipediaImporter::extract_concepts(
    const std::string& text
) const {
    std::vector<SuggestedConcept> concepts;
    
    // Simple extraction: capitalized words (proper nouns)
    std::regex word_regex("\\b[A-Z][a-z]+(?:\\s+[A-Z][a-z]+)*\\b");
    std::smatch match;
    
    std::string::const_iterator searchStart(text.cbegin());
    while (std::regex_search(searchStart, text.cend(), match, word_regex)) {
        std::string concept_str = match[0];
        
        // Get context (20 chars before and after)
        size_t pos = match.position() + std::distance(text.cbegin(), searchStart);
        size_t ctx_start = (pos > 20) ? (pos - 20) : 0;
        size_t ctx_end = std::min(pos + concept_str.length() + 20, text.length());
        std::string context = text.substr(ctx_start, ctx_end - ctx_start);
        
        concepts.emplace_back(concept_str, context);
        
        searchStart = match.suffix().first;
        
        // Limit concepts
        if (concepts.size() >= 20) break;
    }
    
    return concepts;
}

std::vector<SuggestedRelation> WikipediaImporter::extract_basic_relations(
    const std::string& text
) const {
    std::vector<SuggestedRelation> relations;
    
    // Simple pattern matching for "X is a Y" relations
    std::regex is_a_regex("([A-Z][a-z]+(?:\\s+[A-Z][a-z]+)*)\\s+is\\s+a\\s+([a-z]+(?:\\s+[a-z]+)*)");
    std::smatch match;
    
    std::string::const_iterator searchStart(text.cbegin());
    while (std::regex_search(searchStart, text.cend(), match, is_a_regex)) {
        std::string source = match[1];
        std::string target = match[2];
        std::string evidence = match[0];
        
        relations.emplace_back(source, target, "is-a", evidence);
        
        searchStart = match.suffix().first;
        
        // Limit relations
        if (relations.size() >= 10) break;
    }
    
    return relations;
}

std::vector<std::string> WikipediaImporter::extract_references(
    const std::string& text
) const {
    std::vector<std::string> references;
    
    // Simple extraction of URLs
    std::regex url_regex("https?://[^\\s]+");
    std::smatch match;
    
    std::string::const_iterator searchStart(text.cbegin());
    while (std::regex_search(searchStart, text.cend(), match, url_regex)) {
        references.push_back(match[0]);
        searchStart = match.suffix().first;
        
        if (references.size() >= 20) break;
    }
    
    return references;
}

std::string WikipediaImporter::remove_html_tags(const std::string& html) const {
    std::string result;
    bool in_tag = false;
    
    for (char c : html) {
        if (c == '<') {
            in_tag = true;
        } else if (c == '>') {
            in_tag = false;
        } else if (!in_tag) {
            result += c;
        }
    }
    
    return result;
}

std::string WikipediaImporter::normalize_whitespace(const std::string& text) const {
    std::string result;
    bool prev_space = false;
    
    for (char c : text) {
        if (std::isspace(c)) {
            if (!prev_space) {
                result += ' ';
                prev_space = true;
            }
        } else {
            result += c;
            prev_space = false;
        }
    }
    
    return result;
}

} // namespace brain19
