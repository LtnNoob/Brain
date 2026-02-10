#include "context_accumulator.hpp"
#include <algorithm>
#include <cctype>

namespace brain19 {

ContextAccumulator::ContextAccumulator()
    : texts_processed_(0)
{
    for (size_t i = 0; i < DOMAIN_COUNT; ++i) {
        domain_counts_[DOMAINS[i]] = 0;
    }
}

void ContextAccumulator::record_concept(const std::string& label,
                                         const std::string& domain) {
    concept_freq_[label]++;

    if (concept_domain_.find(label) == concept_domain_.end()) {
        std::string d = domain.empty() ? classify_domain(label) : domain;
        concept_domain_[label] = d;
        domain_counts_[d]++;
    }
}

void ContextAccumulator::record_text_processed(const std::string& /*text*/) {
    texts_processed_++;
}

std::vector<ContextAccumulator::DomainStats>
ContextAccumulator::get_domain_stats() const {
    size_t max_count = 0;
    for (auto& [d, c] : domain_counts_) {
        if (c > max_count) max_count = c;
    }

    std::vector<DomainStats> result;
    result.reserve(DOMAIN_COUNT);
    for (size_t i = 0; i < DOMAIN_COUNT; ++i) {
        auto it = domain_counts_.find(DOMAINS[i]);
        size_t count = (it != domain_counts_.end()) ? it->second : 0;
        double score = (max_count > 0) ? static_cast<double>(count) / max_count : 0.0;
        result.push_back({DOMAINS[i], count, score});
    }
    return result;
}

std::vector<std::string> ContextAccumulator::find_knowledge_gaps() const {
    auto stats = get_domain_stats();
    std::vector<std::string> gaps;
    for (const auto& s : stats) {
        if (s.coverage_score < 0.3) {
            gaps.push_back(s.domain);
        }
    }
    return gaps;
}

std::vector<std::string>
ContextAccumulator::suggest_types(const std::string& entity_name) const {
    std::vector<std::string> types;

    // Simple heuristic: check if entity name contains domain keywords
    std::string lower;
    lower.reserve(entity_name.size());
    for (char c : entity_name) lower += static_cast<char>(std::tolower(c));

    // Biology signals
    if (lower.find("cell") != std::string::npos ||
        lower.find("gene") != std::string::npos ||
        lower.find("protein") != std::string::npos ||
        lower.find("organism") != std::string::npos ||
        lower.find("species") != std::string::npos)
        types.push_back("Organism");

    // Place signals
    if (lower.find("city") != std::string::npos ||
        lower.find("country") != std::string::npos ||
        lower.find("river") != std::string::npos ||
        lower.find("mountain") != std::string::npos ||
        lower.find("land") != std::string::npos)
        types.push_back("Place");

    // Person signals
    if (lower.find("person") != std::string::npos ||
        lower.find("scientist") != std::string::npos ||
        lower.find("author") != std::string::npos)
        types.push_back("Person");

    // Science signals
    if (lower.find("theory") != std::string::npos ||
        lower.find("law") != std::string::npos ||
        lower.find("equation") != std::string::npos ||
        lower.find("theorem") != std::string::npos)
        types.push_back("Science");

    // Technology signals
    if (lower.find("algorithm") != std::string::npos ||
        lower.find("software") != std::string::npos ||
        lower.find("protocol") != std::string::npos ||
        lower.find("system") != std::string::npos)
        types.push_back("Technology");

    if (types.empty()) types.push_back("Entity");
    return types;
}

size_t ContextAccumulator::total_concepts() const {
    size_t total = 0;
    for (auto& [d, c] : domain_counts_) total += c;
    return total;
}

size_t ContextAccumulator::texts_processed() const {
    return texts_processed_;
}

size_t ContextAccumulator::concept_frequency(const std::string& label) const {
    auto it = concept_freq_.find(label);
    return (it != concept_freq_.end()) ? it->second : 0;
}

std::string ContextAccumulator::classify_domain(const std::string& label) const {
    std::string lower;
    lower.reserve(label.size());
    for (char c : label) lower += static_cast<char>(std::tolower(c));

    // Keyword-based classification
    static const std::vector<std::pair<std::string, std::vector<std::string>>> domain_keywords = {
        {"biology",    {"cell", "dna", "rna", "gene", "protein", "enzyme", "organism", "species",
                        "evolution", "mutation", "bacteria", "virus", "tissue", "organ", "neuron",
                        "brain", "photosynthesis", "respiration", "metabolism", "heredity",
                        "immune", "nervous", "plant", "animal"}},
        {"physics",    {"atom", "electron", "proton", "neutron", "photon", "force", "energy",
                        "mass", "gravity", "light", "wave", "quantum", "relativity", "field",
                        "particle", "spectrum", "electromagnetism", "thermodynamics",
                        "momentum", "velocity", "acceleration", "pressure", "temperature",
                        "frequency", "wavelength", "amplitude", "density", "speed", "heat",
                        "electricity", "magnetism", "sound"}},
        {"chemistry",  {"molecule", "compound", "ion", "bond", "reaction", "catalyst",
                        "solution", "acid", "base", "oxidation", "reduction", "element",
                        "entropy", "nucleus"}},
        {"mathematics",{"number", "integer", "real", "complex", "variable", "equation",
                        "function", "limit", "derivative", "integral", "proof", "theorem",
                        "axiom", "algebra", "geometry", "calculus", "topology", "graph",
                        "matrix", "vector", "scalar", "infinity", "zero", "pi", "set",
                        "probability", "statistics", "logic", "inequality"}},
        {"technology", {"computer", "software", "hardware", "algorithm", "network", "internet",
                        "database", "turing", "complexity", "recursion", "boolean", "bit",
                        "byte", "encryption", "operating", "programming", "artificial",
                        "machine_learning", "neural_network"}},
        {"geography",  {"place", "city", "country", "region", "continent", "ocean", "mountain",
                        "river", "earth", "climate", "weather", "ecosystem"}},
        {"social",     {"society", "government", "law", "culture", "education", "religion",
                        "economics", "sociology", "psychology", "person", "communication"}},
        {"humanities", {"language", "art", "music", "literature", "history", "philosophy",
                        "symbol", "emotion", "consciousness", "perception", "reasoning"}},
        {"ontology",   {"entity", "object", "action", "property", "relation", "event", "state",
                        "process", "abstract", "concrete", "physical", "mental", "social",
                        "temporal", "spatial", "category", "type", "instance", "concept",
                        "definition", "cause", "effect", "system", "structure", "pattern",
                        "change", "quantity", "quality", "value", "truth", "knowledge",
                        "information", "meaning", "evidence", "rule", "principle", "model",
                        "theory", "hypothesis", "fact", "measurement", "unit", "dimension",
                        "boundary", "context", "scope", "identity", "existence"}},
    };

    for (const auto& [domain, keywords] : domain_keywords) {
        for (const auto& kw : keywords) {
            if (lower.find(kw) != std::string::npos) {
                return domain;
            }
        }
    }
    return "general";
}

} // namespace brain19
