#include "bpe_tokenizer.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>

namespace brain19 {

// =============================================================================
// Construction
// =============================================================================

BPETokenizer::BPETokenizer() {
    init_base_vocab();
}

void BPETokenizer::init_base_vocab() {
    id_to_token_.clear();
    token_to_id_.clear();

    // Special tokens (0-4)
    id_to_token_.push_back("[PAD]");    // 0
    id_to_token_.push_back("[BOS]");    // 1
    id_to_token_.push_back("[EOS]");    // 2
    id_to_token_.push_back("[UNK]");    // 3
    id_to_token_.push_back("[SEP]");    // 4

    for (size_t i = 0; i < id_to_token_.size(); ++i) {
        token_to_id_[id_to_token_[i]] = static_cast<uint16_t>(i);
    }

    // Reserve slots 5-1004 for concept tokens (filled by build_from_ltm)
    // Slots 1005+ for BPE merge tokens
}

// =============================================================================
// Build from LTM
// =============================================================================

void BPETokenizer::build_from_ltm(const LongTermMemory& ltm) {
    init_base_vocab();
    token_concept_map_.clear();
    concept_token_map_.clear();

    // Map concept labels to tokens 5-1004
    auto all_ids = ltm.get_all_concept_ids();
    uint16_t token_id = LanguageConfig::CONCEPT_TOKEN_START;

    for (auto cid : all_ids) {
        if (token_id > LanguageConfig::CONCEPT_TOKEN_END) break;

        auto info = ltm.retrieve_concept(cid);
        if (!info) continue;

        // Pad id_to_token_ up to token_id
        while (id_to_token_.size() <= token_id) {
            id_to_token_.push_back("[RESERVED]");
        }

        id_to_token_[token_id] = info->label;
        token_to_id_[info->label] = token_id;
        token_concept_map_[token_id] = cid;
        concept_token_map_[cid] = token_id;
        ++token_id;
    }

    // Fill remaining concept slots with reserved
    while (id_to_token_.size() <= LanguageConfig::CONCEPT_TOKEN_END) {
        id_to_token_.push_back("[RESERVED]");
    }

    // Add base byte tokens (256 single-byte tokens) starting at BPE_TOKEN_START
    while (id_to_token_.size() < LanguageConfig::BPE_TOKEN_START) {
        id_to_token_.push_back("[RESERVED]");
    }

    for (int b = 0; b < 256; ++b) {
        std::string byte_tok(1, static_cast<char>(b));
        uint16_t bid = static_cast<uint16_t>(id_to_token_.size());
        if (bid >= LanguageConfig::VOCAB_SIZE) break;
        id_to_token_.push_back(byte_tok);
        // Only add printable/useful bytes to lookup
        if (!token_to_id_.count(byte_tok)) {
            token_to_id_[byte_tok] = bid;
        }
    }
}

// =============================================================================
// BPE Training
// =============================================================================

void BPETokenizer::train(const std::vector<std::string>& corpus,
                          const LongTermMemory& ltm,
                          size_t target_vocab_size) {
    build_from_ltm(ltm);

    // Tokenize corpus into byte sequences (word-level)
    std::vector<std::vector<std::string>> tokenized;
    for (const auto& text : corpus) {
        auto words = split_words(text);
        for (const auto& w : words) {
            std::vector<std::string> chars;
            for (char c : w) {
                chars.push_back(std::string(1, c));
            }
            if (!chars.empty()) {
                tokenized.push_back(std::move(chars));
            }
        }
    }

    // Iterative BPE merge
    while (id_to_token_.size() < target_vocab_size) {
        auto pairs = count_pairs(tokenized);
        if (pairs.empty()) break;

        // Find most frequent pair
        std::string best_pair;
        size_t best_count = 0;
        for (const auto& [pair_key, count] : pairs) {
            if (count > best_count) {
                best_count = count;
                best_pair = pair_key;
            }
        }
        if (best_count < 2) break;  // no pair occurs more than once

        // Split pair key back into two tokens
        size_t sep = best_pair.find('\x01');
        if (sep == std::string::npos) break;
        std::string left = best_pair.substr(0, sep);
        std::string right = best_pair.substr(sep + 1);
        std::string merged = left + right;

        merges_.push_back({left, right});

        uint16_t new_id = static_cast<uint16_t>(id_to_token_.size());
        id_to_token_.push_back(merged);
        token_to_id_[merged] = new_id;

        // Apply merge to all tokenized words
        for (auto& word : tokenized) {
            for (size_t i = 0; i + 1 < word.size(); ) {
                if (word[i] == left && word[i + 1] == right) {
                    word[i] = merged;
                    word.erase(word.begin() + static_cast<long>(i) + 1);
                } else {
                    ++i;
                }
            }
        }
    }
}

BPETokenizer::PairCount BPETokenizer::count_pairs(
    const std::vector<std::vector<std::string>>& tokenized) const {
    PairCount counts;
    for (const auto& word : tokenized) {
        for (size_t i = 0; i + 1 < word.size(); ++i) {
            std::string key = word[i] + '\x01' + word[i + 1];
            counts[key]++;
        }
    }
    return counts;
}

// =============================================================================
// Encode
// =============================================================================

std::vector<uint16_t> BPETokenizer::encode(const std::string& text) const {
    std::vector<uint16_t> result;
    auto words = split_words(text);

    for (const auto& word : words) {
        // First try: exact match as concept token
        auto it = token_to_id_.find(word);
        if (it != token_to_id_.end() && is_concept_token(it->second)) {
            result.push_back(it->second);
            continue;
        }

        // Try case-insensitive concept match
        std::string lower_word = word;
        for (auto& c : lower_word) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        bool found_concept = false;
        for (const auto& [tok, tid] : token_to_id_) {
            if (!is_concept_token(tid)) continue;
            std::string lower_tok = tok;
            for (auto& c : lower_tok) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            if (lower_word == lower_tok) {
                result.push_back(tid);
                found_concept = true;
                break;
            }
        }
        if (found_concept) continue;

        // Apply BPE
        auto bpe_tokens = bpe_encode_word(word);
        for (const auto& tok : bpe_tokens) {
            auto tok_it = token_to_id_.find(tok);
            if (tok_it != token_to_id_.end()) {
                result.push_back(tok_it->second);
            } else {
                // Fall back to byte-level encoding
                for (char c : tok) {
                    std::string byte_tok(1, c);
                    auto byte_it = token_to_id_.find(byte_tok);
                    if (byte_it != token_to_id_.end()) {
                        result.push_back(byte_it->second);
                    } else {
                        result.push_back(LanguageConfig::UNK_TOKEN);
                    }
                }
            }
        }
    }

    // Truncate to max seq length
    if (result.size() > LanguageConfig::MAX_SEQ_LEN) {
        result.resize(LanguageConfig::MAX_SEQ_LEN);
    }
    return result;
}

std::vector<std::string> BPETokenizer::bpe_encode_word(const std::string& word) const {
    // Start with character-level tokens
    std::vector<std::string> tokens;
    for (char c : word) {
        tokens.push_back(std::string(1, c));
    }

    // Apply merges in order
    for (const auto& [left, right] : merges_) {
        for (size_t i = 0; i + 1 < tokens.size(); ) {
            if (tokens[i] == left && tokens[i + 1] == right) {
                tokens[i] = left + right;
                tokens.erase(tokens.begin() + static_cast<long>(i) + 1);
            } else {
                ++i;
            }
        }
    }
    return tokens;
}

// =============================================================================
// Decode
// =============================================================================

std::string BPETokenizer::decode(const std::vector<uint16_t>& tokens) const {
    std::string result;
    for (auto tid : tokens) {
        if (tid == LanguageConfig::PAD_TOKEN ||
            tid == LanguageConfig::BOS_TOKEN ||
            tid == LanguageConfig::EOS_TOKEN) {
            continue;
        }
        if (tid == LanguageConfig::SEP_TOKEN) {
            result += " ";
            continue;
        }
        if (tid < id_to_token_.size()) {
            const auto& tok = id_to_token_[tid];
            if (tok != "[RESERVED]" && tok != "[UNK]") {
                if (!result.empty() && result.back() != ' ' &&
                    is_concept_token(tid)) {
                    result += " ";
                }
                result += tok;
            }
        }
    }
    return result;
}

// =============================================================================
// Concept Token Mapping
// =============================================================================

std::optional<ConceptId> BPETokenizer::token_to_concept(uint16_t token) const {
    auto it = token_concept_map_.find(token);
    if (it != token_concept_map_.end()) return it->second;
    return std::nullopt;
}

std::optional<uint16_t> BPETokenizer::concept_to_token(ConceptId cid) const {
    auto it = concept_token_map_.find(cid);
    if (it != concept_token_map_.end()) return it->second;
    return std::nullopt;
}

bool BPETokenizer::is_concept_token(uint16_t token) const {
    return token >= LanguageConfig::CONCEPT_TOKEN_START &&
           token <= LanguageConfig::CONCEPT_TOKEN_END &&
           token_concept_map_.count(token) > 0;
}

uint16_t BPETokenizer::get_token_id(const std::string& token) const {
    auto it = token_to_id_.find(token);
    return it != token_to_id_.end() ? it->second : LanguageConfig::UNK_TOKEN;
}

// =============================================================================
// Word Splitting
// =============================================================================

std::vector<std::string> BPETokenizer::split_words(const std::string& text) const {
    std::vector<std::string> words;
    std::string current;

    for (char c : text) {
        if (std::isspace(static_cast<unsigned char>(c))) {
            if (!current.empty()) {
                words.push_back(current);
                current.clear();
            }
        } else if (std::ispunct(static_cast<unsigned char>(c)) && c != '\'') {
            if (!current.empty()) {
                words.push_back(current);
                current.clear();
            }
            words.push_back(std::string(1, c));
        } else {
            current += c;
        }
    }
    if (!current.empty()) {
        words.push_back(current);
    }
    return words;
}

// =============================================================================
// Persistence
// =============================================================================

void BPETokenizer::save(const std::string& path) const {
    std::ofstream f(path);
    if (!f) return;

    // Write merges
    f << merges_.size() << "\n";
    for (const auto& [left, right] : merges_) {
        f << left << "\t" << right << "\n";
    }

    // Write concept mappings
    f << token_concept_map_.size() << "\n";
    for (const auto& [tid, cid] : token_concept_map_) {
        f << tid << "\t" << cid << "\t" << id_to_token_[tid] << "\n";
    }
}

bool BPETokenizer::load(const std::string& path) {
    std::ifstream f(path);
    if (!f) return false;

    init_base_vocab();
    merges_.clear();
    token_concept_map_.clear();
    concept_token_map_.clear();

    // Read merges
    size_t num_merges;
    f >> num_merges;
    f.ignore();
    for (size_t i = 0; i < num_merges; ++i) {
        std::string line;
        std::getline(f, line);
        size_t tab = line.find('\t');
        if (tab == std::string::npos) continue;
        std::string left = line.substr(0, tab);
        std::string right = line.substr(tab + 1);
        merges_.push_back({left, right});

        std::string merged = left + right;
        uint16_t new_id = static_cast<uint16_t>(id_to_token_.size());
        id_to_token_.push_back(merged);
        token_to_id_[merged] = new_id;
    }

    // Read concept mappings
    size_t num_concepts;
    f >> num_concepts;
    f.ignore();
    for (size_t i = 0; i < num_concepts; ++i) {
        uint16_t tid;
        ConceptId cid;
        std::string label;
        f >> tid >> cid;
        f.ignore();
        std::getline(f, label);

        while (id_to_token_.size() <= tid) {
            id_to_token_.push_back("[RESERVED]");
        }
        id_to_token_[tid] = label;
        token_to_id_[label] = tid;
        token_concept_map_[tid] = cid;
        concept_token_map_[cid] = tid;
    }

    return true;
}

} // namespace brain19
