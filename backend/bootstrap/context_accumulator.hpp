#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <cstddef>

namespace brain19 {

// ContextAccumulator: Tracks knowledge growth and improves suggestions
//
// As concepts accumulate, the accumulator:
// - Tracks how often each concept is referenced (frequency)
// - Detects underrepresented domains (knowledge gaps)
// - Provides progress metrics per domain
// - Suggests what to learn next based on coverage
class ContextAccumulator {
public:
    // Pre-defined knowledge domains
    static constexpr const char* DOMAINS[] = {
        "ontology", "geography", "biology", "physics", "chemistry",
        "mathematics", "technology", "social", "humanities", "general"
    };
    static constexpr size_t DOMAIN_COUNT = 10;

    ContextAccumulator();

    // Record that a concept was added or referenced
    void record_concept(const std::string& label, const std::string& domain = "general");

    // Record text processed (for frequency / gap analysis)
    void record_text_processed(const std::string& text);

    // Domain coverage: fraction of total concepts in each domain
    struct DomainStats {
        std::string domain;
        size_t concept_count;
        double coverage_score;  // 0.0–1.0 relative to best domain
    };
    std::vector<DomainStats> get_domain_stats() const;

    // Find underrepresented domains
    std::vector<std::string> find_knowledge_gaps() const;

    // Suggest types for an entity based on accumulated knowledge
    std::vector<std::string> suggest_types(const std::string& entity_name) const;

    // Total concepts tracked
    size_t total_concepts() const;

    // Texts processed
    size_t texts_processed() const;

    // Concept frequency (how many times referenced)
    size_t concept_frequency(const std::string& label) const;

private:
    std::unordered_map<std::string, size_t> concept_freq_;   // label → count
    std::unordered_map<std::string, std::string> concept_domain_; // label → domain
    std::unordered_map<std::string, size_t> domain_counts_;  // domain → concept count
    size_t texts_processed_;

    std::string classify_domain(const std::string& label) const;
};

} // namespace brain19
