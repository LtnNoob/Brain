#include "scholar_importer.hpp"
#include "http_client.hpp"
#include <sstream>
#include <algorithm>
#include <regex>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace brain19 {

ScholarImporter::ScholarImporter()
    : next_proposal_id_(1)
{
}

ScholarImporter::~ScholarImporter() {
}

std::unique_ptr<KnowledgeProposal> ScholarImporter::import_paper_by_doi(
    const std::string& doi
) {
    auto proposal = std::make_unique<KnowledgeProposal>();
    proposal->proposal_id = next_proposal_id_++;
    proposal->source_type = SourceType::GOOGLE_SCHOLAR;
    proposal->source_reference = "DOI: " + doi;
    proposal->import_timestamp = std::chrono::system_clock::now();
    
    proposal->notes_for_human_review = 
        "Research paper (DOI: " + doi + "). " +
        "Contains research claims requiring verification. " +
        "Check publication venue and peer review status.";
    
    proposal->suggested_epistemic_type = SuggestedEpistemicType::HYPOTHESIS_CANDIDATE;
    
    return proposal;
}

std::unique_ptr<KnowledgeProposal> ScholarImporter::import_paper_by_doi_online(
    const std::string& doi
) {
    std::string url = "https://api.semanticscholar.org/graph/v1/paper/DOI:" +
        HttpClient::url_encode(doi) +
        "?fields=title,abstract,authors,year,venue,isOpenAccess";

    auto resp = HttpClient::http_get(url);
    if (!resp.success) {
        auto proposal = std::make_unique<KnowledgeProposal>();
        proposal->proposal_id = next_proposal_id_++;
        proposal->source_type = SourceType::GOOGLE_SCHOLAR;
        proposal->source_reference = "DOI: " + doi;
        proposal->import_timestamp = std::chrono::system_clock::now();
        proposal->notes_for_human_review = "FETCH FAILED: " + resp.error;
        return proposal;
    }

    try {
        auto j = json::parse(resp.body);

        std::string title = j.value("title", "Unknown");
        std::string abstract_text = j.value("abstract", "");
        std::string year = j.contains("year") && !j["year"].is_null()
            ? std::to_string(j["year"].get<int>()) : "";
        std::string venue = j.value("venue", "");

        std::vector<std::string> authors;
        if (j.contains("authors") && j["authors"].is_array()) {
            for (auto& a : j["authors"]) {
                authors.push_back(a.value("name", ""));
            }
        }

        bool is_preprint = venue.empty() || venue.find("arXiv") != std::string::npos;

        // Use existing parse_paper_text
        std::string full_text = abstract_text.empty() ? title : abstract_text;
        auto proposal = parse_paper_text(title, full_text, authors, year, venue, is_preprint);
        proposal->source_reference = "DOI: " + doi;
        return proposal;

    } catch (const std::exception& e) {
        auto proposal = std::make_unique<KnowledgeProposal>();
        proposal->proposal_id = next_proposal_id_++;
        proposal->source_type = SourceType::GOOGLE_SCHOLAR;
        proposal->source_reference = "DOI: " + doi;
        proposal->import_timestamp = std::chrono::system_clock::now();
        proposal->notes_for_human_review = std::string("JSON parse error: ") + e.what();
        return proposal;
    }
}

std::vector<std::unique_ptr<KnowledgeProposal>> ScholarImporter::search_papers(
    const std::string& query,
    int limit
) {
    std::vector<std::unique_ptr<KnowledgeProposal>> results;

    std::string url = "https://api.semanticscholar.org/graph/v1/paper/search?query=" +
        HttpClient::url_encode(query) +
        "&limit=" + std::to_string(limit) +
        "&fields=title,abstract,authors,year,venue,isOpenAccess";

    auto resp = HttpClient::http_get(url);
    if (!resp.success) {
        auto proposal = std::make_unique<KnowledgeProposal>();
        proposal->proposal_id = next_proposal_id_++;
        proposal->source_type = SourceType::GOOGLE_SCHOLAR;
        proposal->import_timestamp = std::chrono::system_clock::now();
        proposal->notes_for_human_review = "SEARCH FAILED: " + resp.error;
        results.push_back(std::move(proposal));
        return results;
    }

    try {
        auto j = json::parse(resp.body);

        if (!j.contains("data") || !j["data"].is_array()) {
            return results;
        }

        for (auto& paper : j["data"]) {
            std::string title = paper.value("title", "Unknown");
            std::string abstract_text = paper.value("abstract", "");
            std::string year = paper.contains("year") && !paper["year"].is_null()
                ? std::to_string(paper["year"].get<int>()) : "";
            std::string venue = paper.value("venue", "");

            std::vector<std::string> authors;
            if (paper.contains("authors") && paper["authors"].is_array()) {
                for (auto& a : paper["authors"]) {
                    authors.push_back(a.value("name", ""));
                }
            }

            bool is_preprint = venue.empty() || venue.find("arXiv") != std::string::npos;
            std::string full_text = abstract_text.empty() ? title : abstract_text;

            auto proposal = parse_paper_text(title, full_text, authors, year, venue, is_preprint);
            results.push_back(std::move(proposal));
        }

    } catch (const std::exception& e) {
        auto proposal = std::make_unique<KnowledgeProposal>();
        proposal->proposal_id = next_proposal_id_++;
        proposal->source_type = SourceType::GOOGLE_SCHOLAR;
        proposal->import_timestamp = std::chrono::system_clock::now();
        proposal->notes_for_human_review = std::string("JSON parse error: ") + e.what();
        results.push_back(std::move(proposal));
    }

    return results;
}

std::unique_ptr<KnowledgeProposal> ScholarImporter::import_paper_from_url(
    const std::string& url
) {
    auto proposal = std::make_unique<KnowledgeProposal>();
    proposal->proposal_id = next_proposal_id_++;
    proposal->source_type = SourceType::GOOGLE_SCHOLAR;
    proposal->source_reference = url;
    proposal->import_timestamp = std::chrono::system_clock::now();
    
    return proposal;
}

std::unique_ptr<KnowledgeProposal> ScholarImporter::parse_paper_text(
    const std::string& title,
    const std::string& text,
    const std::vector<std::string>& authors,
    const std::string& year,
    const std::string& venue,
    bool is_preprint
) {
    auto proposal = std::make_unique<KnowledgeProposal>();
    proposal->proposal_id = next_proposal_id_++;
    proposal->source_type = SourceType::GOOGLE_SCHOLAR;
    proposal->title = title;
    proposal->authors = authors;
    proposal->publication_year = year;
    proposal->publication_venue = venue;
    proposal->is_preprint = is_preprint;
    proposal->import_timestamp = std::chrono::system_clock::now();
    
    // Extract sections
    std::string abstract = extract_abstract(text);
    std::string conclusion = extract_conclusion(text);
    
    // Store abstract as primary extracted text
    proposal->extracted_text = abstract;
    
    // Extract research concepts
    proposal->suggested_concepts = extract_research_concepts(abstract);
    
    // Extract claims as relations
    proposal->suggested_relations = extract_claims(abstract + " " + conclusion);
    
    // Determine epistemic type based on language
    // CRITICAL: This is a SUGGESTION only, NOT an assignment
    // Importers MUST NOT assign actual EpistemicType or Trust
    // Human must explicitly decide during LTM ingestion
    if (contains_uncertainty_language(abstract)) {
        proposal->suggested_epistemic_type = SuggestedEpistemicType::HYPOTHESIS_CANDIDATE;
    } else {
        proposal->suggested_epistemic_type = SuggestedEpistemicType::THEORY_CANDIDATE;
    }
    
    // Build human review notes
    std::ostringstream notes;
    notes << "Research paper: " << title << ". ";
    if (is_preprint) {
        notes << "⚠ PREPRINT (not peer-reviewed). ";
    }
    notes << "Published in " << venue << " (" << year << "). ";
    notes << "Authors: ";
    for (size_t i = 0; i < authors.size() && i < 3; i++) {
        notes << authors[i];
        if (i < authors.size() - 1) notes << ", ";
    }
    if (authors.size() > 3) notes << " et al.";
    notes << ". Contains " << proposal->suggested_concepts.size() << " research concepts. ";
    
    std::string hedging = extract_hedging_language(abstract);
    if (!hedging.empty()) {
        notes << "Note: Contains uncertainty language (\"" << hedging << "\"). ";
    }
    
    notes << "Requires expert human review before acceptance.";
    
    proposal->notes_for_human_review = notes.str();
    
    return proposal;
}

std::string ScholarImporter::extract_abstract(const std::string& text) const {
    // Simple heuristic: look for "Abstract" section
    std::regex abstract_regex("Abstract[:\\.\\s]+(.*?)(?=\\n\\n|Introduction|$)", 
                             std::regex::icase);
    std::smatch match;
    
    if (std::regex_search(text, match, abstract_regex)) {
        return match[1].str();
    }
    
    // Fallback: first 500 chars
    return text.substr(0, std::min(text.length(), size_t(500)));
}

std::string ScholarImporter::extract_conclusion(const std::string& text) const {
    // Look for "Conclusion" section
    std::regex conclusion_regex("Conclusion[:\\.\\s]+(.*?)(?=\\n\\n|References|$)", 
                               std::regex::icase);
    std::smatch match;
    
    if (std::regex_search(text, match, conclusion_regex)) {
        return match[1].str();
    }
    
    return "";
}

std::vector<SuggestedConcept> ScholarImporter::extract_research_concepts(
    const std::string& text
) const {
    std::vector<SuggestedConcept> concepts;
    
    // Extract technical terms (capitalized, multi-word phrases)
    std::regex tech_regex("\\b[A-Z][a-z]+(?:\\s+[A-Z][a-z]+)+\\b");
    std::smatch match;
    
    std::string::const_iterator searchStart(text.cbegin());
    while (std::regex_search(searchStart, text.cend(), match, tech_regex)) {
        std::string concept_str = match[0];
        
        // Get context
        size_t pos = match.position() + std::distance(text.cbegin(), searchStart);
        size_t ctx_start = (pos > 30) ? (pos - 30) : 0;
        size_t ctx_end = std::min(pos + concept_str.length() + 30, text.length());
        std::string context = text.substr(ctx_start, ctx_end - ctx_start);
        
        concepts.emplace_back(concept_str, context);
        
        searchStart = match.suffix().first;
        
        if (concepts.size() >= 15) break;
    }
    
    return concepts;
}

std::vector<SuggestedRelation> ScholarImporter::extract_claims(
    const std::string& text
) const {
    std::vector<SuggestedRelation> relations;
    
    // Look for causal claims: "X causes Y", "X leads to Y"
    std::regex causal_regex("([A-Z][a-z]+(?:\\s+[a-z]+)*)\\s+(?:causes|leads to)\\s+([a-z]+(?:\\s+[a-z]+)*)");
    std::smatch match;
    
    std::string::const_iterator searchStart(text.cbegin());
    while (std::regex_search(searchStart, text.cend(), match, causal_regex)) {
        std::string source = match[1];
        std::string target = match[2];
        std::string evidence = match[0];
        
        relations.emplace_back(source, target, "causes", evidence);
        
        searchStart = match.suffix().first;
        
        if (relations.size() >= 10) break;
    }
    
    return relations;
}

std::vector<std::string> ScholarImporter::extract_methods(const std::string& text) const {
    std::vector<std::string> methods;
    
    // Look for methodology keywords
    std::vector<std::string> method_keywords = {
        "experiment", "survey", "analysis", "model", "simulation", "measurement"
    };
    
    for (const auto& keyword : method_keywords) {
        if (text.find(keyword) != std::string::npos) {
            methods.push_back(keyword);
        }
    }
    
    return methods;
}

std::vector<std::string> ScholarImporter::extract_citations(const std::string& text) const {
    std::vector<std::string> citations;
    
    // Simple citation pattern: [Author, Year]
    std::regex citation_regex("\\[([A-Z][a-z]+(?:\\s+et al\\.)?),?\\s+(\\d{4})\\]");
    std::smatch match;
    
    std::string::const_iterator searchStart(text.cbegin());
    while (std::regex_search(searchStart, text.cend(), match, citation_regex)) {
        citations.push_back(match[0]);
        searchStart = match.suffix().first;
        
        if (citations.size() >= 20) break;
    }
    
    return citations;
}

bool ScholarImporter::contains_uncertainty_language(const std::string& text) const {
    std::vector<std::string> uncertainty_markers = {
        "may", "might", "could", "possibly", "likely", "suggest", 
        "indicate", "appear", "seem", "hypothesis", "preliminary"
    };
    
    std::string lower_text = text;
    std::transform(lower_text.begin(), lower_text.end(), lower_text.begin(), ::tolower);
    
    for (const auto& marker : uncertainty_markers) {
        if (lower_text.find(marker) != std::string::npos) {
            return true;
        }
    }
    
    return false;
}

std::string ScholarImporter::extract_hedging_language(const std::string& text) const {
    std::vector<std::string> hedges = {
        "may", "might", "could", "possibly", "likely", "suggest"
    };
    
    std::string lower_text = text;
    std::transform(lower_text.begin(), lower_text.end(), lower_text.begin(), ::tolower);
    
    for (const auto& hedge : hedges) {
        if (lower_text.find(hedge) != std::string::npos) {
            return hedge;
        }
    }
    
    return "";
}

} // namespace brain19
