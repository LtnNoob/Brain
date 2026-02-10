#include "bootstrap_interface.hpp"
#include <algorithm>
#include <cctype>
#include <sstream>
#include <unordered_set>

namespace brain19 {

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

BootstrapInterface::BootstrapInterface(LongTermMemory& ltm)
    : ltm_(ltm)
    , foundation_loaded_(false)
{}

// ---------------------------------------------------------------------------
// Foundation
// ---------------------------------------------------------------------------

void BootstrapInterface::initialize_foundation() {
    if (foundation_loaded_) return;

    FoundationConcepts::seed_all(ltm_);
    foundation_loaded_ = true;

    // Rebuild label index from LTM
    rebuild_label_index();

    // Record all foundation concepts in the accumulator
    for (const auto& [label, id] : label_index_) {
        accumulator_.record_concept(label);
    }
}

bool BootstrapInterface::is_foundation_loaded() const {
    return foundation_loaded_;
}

// ---------------------------------------------------------------------------
// Text Processing
// ---------------------------------------------------------------------------

std::vector<BootstrapProposal>
BootstrapInterface::process_text(const std::string& text) {
    accumulator_.record_text_processed(text);

    auto candidates = extract_candidate_entities(text);
    std::vector<BootstrapProposal> proposals;

    for (const auto& entity : candidates) {
        // Skip if already known or already rejected
        if (is_known(entity) || rejected_.count(entity)) continue;

        // Skip if already pending
        bool already_pending = false;
        for (const auto& p : pending_) {
            if (p.entity_name == entity) { already_pending = true; break; }
        }
        if (already_pending) continue;

        BootstrapProposal prop;
        prop.entity_name = entity;
        prop.context_text = text;
        prop.suggested_types = accumulator_.suggest_types(entity);
        prop.similar_concepts = find_similar(entity);
        prop.suggested_trust = 0.85;  // Default suggestion for new concepts
        prop.auto_description = generate_description(entity, text);

        proposals.push_back(std::move(prop));
    }

    // Add to pending
    for (const auto& p : proposals) {
        pending_.push_back(p);
    }

    return proposals;
}

// ---------------------------------------------------------------------------
// Human Review
// ---------------------------------------------------------------------------

void BootstrapInterface::accept_proposal(const BootstrapProposal& p,
                                          const std::string& human_description,
                                          EpistemicType type, double trust) {
    // Store in LTM
    auto id = ltm_.store_concept(
        p.entity_name,
        human_description.empty() ? p.auto_description : human_description,
        EpistemicMetadata(type, EpistemicStatus::ACTIVE, trust));

    // Update indices
    label_index_[p.entity_name] = id;
    accumulator_.record_concept(p.entity_name);

    // Remove from pending
    pending_.erase(
        std::remove_if(pending_.begin(), pending_.end(),
            [&](const BootstrapProposal& bp) { return bp.entity_name == p.entity_name; }),
        pending_.end());
}

void BootstrapInterface::reject_proposal(const BootstrapProposal& p,
                                          const std::string& /*reason*/) {
    rejected_.insert(p.entity_name);

    pending_.erase(
        std::remove_if(pending_.begin(), pending_.end(),
            [&](const BootstrapProposal& bp) { return bp.entity_name == p.entity_name; }),
        pending_.end());
}

// ---------------------------------------------------------------------------
// Stats
// ---------------------------------------------------------------------------

size_t BootstrapInterface::known_concepts() const {
    return label_index_.size();
}

size_t BootstrapInterface::pending_proposals() const {
    return pending_.size();
}

// ---------------------------------------------------------------------------
// Progressive Complexity
// ---------------------------------------------------------------------------

std::vector<std::string> BootstrapInterface::suggest_next_topics() const {
    auto gaps = accumulator_.find_knowledge_gaps();

    // Map domain gaps to concrete topic suggestions
    static const std::unordered_map<std::string, std::vector<std::string>> domain_topics = {
        {"biology",    {"Genetics", "Ecology", "Cell Biology", "Anatomy", "Microbiology"}},
        {"physics",    {"Mechanics", "Optics", "Nuclear Physics", "Astrophysics", "Fluid Dynamics"}},
        {"chemistry",  {"Organic Chemistry", "Inorganic Chemistry", "Biochemistry", "Materials Science"}},
        {"mathematics",{"Number Theory", "Linear Algebra", "Differential Equations", "Combinatorics"}},
        {"technology", {"Operating Systems", "Cryptography", "Distributed Systems", "Compilers"}},
        {"geography",  {"Countries of Europe", "World Rivers", "Tectonic Plates", "Biomes"}},
        {"social",     {"Political Systems", "Economic Theory", "Social Psychology", "Anthropology"}},
        {"humanities", {"Ancient History", "Linguistics", "Music Theory", "Modern Art"}},
        {"ontology",   {"Mereology", "Modal Logic", "Set Theory", "Category Theory"}},
        {"general",    {"Current Events", "Famous People", "World Cultures", "Applied Sciences"}},
    };

    std::vector<std::string> suggestions;
    for (const auto& gap : gaps) {
        auto it = domain_topics.find(gap);
        if (it != domain_topics.end()) {
            for (const auto& topic : it->second) {
                suggestions.push_back(topic);
                if (suggestions.size() >= 10) return suggestions;
            }
        }
    }

    // If few gaps, suggest deeper learning in existing domains
    if (suggestions.empty()) {
        suggestions.push_back("Advanced topics in existing domains");
        suggestions.push_back("Cross-domain connections");
        suggestions.push_back("Historical context for known concepts");
    }

    return suggestions;
}

// ---------------------------------------------------------------------------
// Internal Helpers
// ---------------------------------------------------------------------------

void BootstrapInterface::rebuild_label_index() {
    label_index_.clear();
    auto all_ids = ltm_.get_all_concept_ids();
    for (auto id : all_ids) {
        auto info = ltm_.retrieve_concept(id);
        if (info) {
            label_index_[info->label] = id;
        }
    }
}

std::vector<std::string>
BootstrapInterface::find_similar(const std::string& name) const {
    std::vector<std::string> similar;
    std::string lower_name;
    for (char c : name) lower_name += static_cast<char>(std::tolower(c));

    for (const auto& [label, id] : label_index_) {
        std::string lower_label;
        for (char c : label) lower_label += static_cast<char>(std::tolower(c));

        // Substring match in either direction
        if (lower_name.size() >= 3 && lower_label.find(lower_name) != std::string::npos) {
            similar.push_back(label);
        } else if (lower_label.size() >= 3 && lower_name.find(lower_label) != std::string::npos) {
            similar.push_back(label);
        }
        if (similar.size() >= 5) break;
    }
    return similar;
}

std::vector<std::string>
BootstrapInterface::extract_candidate_entities(const std::string& text) const {
    // Simple heuristic: extract capitalised words/phrases as candidates
    std::vector<std::string> candidates;
    std::istringstream stream(text);
    std::string word;
    std::unordered_set<std::string> seen;

    while (stream >> word) {
        // Strip trailing punctuation
        while (!word.empty() && std::ispunct(static_cast<unsigned char>(word.back()))) {
            word.pop_back();
        }
        if (word.empty()) continue;

        // Capitalised words (not at sentence start — simple heuristic)
        if (std::isupper(static_cast<unsigned char>(word[0])) && word.size() >= 2) {
            // Skip very common English words
            std::string lower;
            for (char c : word) lower += static_cast<char>(std::tolower(c));
            static const std::unordered_set<std::string> stopwords = {
                "the", "a", "an", "is", "are", "was", "were", "has", "have",
                "had", "be", "been", "being", "do", "does", "did", "will",
                "would", "could", "should", "may", "might", "can", "this",
                "that", "these", "those", "it", "its", "they", "them",
                "their", "we", "our", "you", "your", "he", "she", "his",
                "her", "i", "my", "me", "not", "no", "yes", "if", "or",
                "and", "but", "so", "for", "of", "in", "on", "at", "to",
                "by", "with", "from", "as", "into", "about", "than",
                "after", "before", "between", "through", "during",
                "each", "every", "all", "both", "few", "more", "most",
                "other", "some", "such", "only", "own", "same", "also",
                "just", "because", "however", "therefore", "thus",
            };
            if (stopwords.count(lower)) continue;

            if (seen.insert(word).second) {
                candidates.push_back(word);
            }
        }
    }
    return candidates;
}

std::string BootstrapInterface::generate_description(const std::string& entity,
                                                       const std::string& context) const {
    // Extract the sentence containing the entity as a basic description
    std::string::size_type pos = context.find(entity);
    if (pos == std::string::npos) {
        return "A concept extracted from processed text.";
    }

    // Find sentence boundaries
    auto start = context.rfind('.', pos);
    start = (start == std::string::npos) ? 0 : start + 1;

    auto end = context.find('.', pos);
    end = (end == std::string::npos) ? context.size() : end + 1;

    std::string sentence = context.substr(start, end - start);

    // Trim whitespace
    auto first = sentence.find_first_not_of(" \t\n\r");
    if (first == std::string::npos) return "A concept extracted from processed text.";
    auto last = sentence.find_last_not_of(" \t\n\r");
    return sentence.substr(first, last - first + 1);
}

bool BootstrapInterface::is_known(const std::string& label) const {
    return label_index_.count(label) > 0;
}

} // namespace brain19
