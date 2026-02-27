// ============================================================================
// Graph Structure Analysis: Diagnosing Reasoning Chain Domain Drift
// ============================================================================
//
// Hypothesis: The graph structure itself — hub properties, polysemous bridge
// nodes, and undifferentiated generic concepts — causes reasoning chains to
// drift into unrelated domains, independent of traversal algorithms.
//
// This program measures:
//   1. Full neighborhood of key seed concepts
//   2. Universal bridge properties (Volume, Mass, Density, etc.)
//   3. Polysemous concept analysis (Replication, Base, etc.)
//   4. Hub node census and polysemous bridge node count
//
#include "ltm/long_term_memory.hpp"
#include "bootstrap/foundation_concepts.hpp"
#include "hybrid/domain_manager.hpp"
#include "memory/relation_type_registry.hpp"
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <set>

using namespace brain19;

// ============================================================================
// Helpers
// ============================================================================

static const char* rel_name(RelationType t) {
    return RelationTypeRegistry::instance().get_name(t).c_str();
}

static std::string concept_label(const LongTermMemory& ltm, ConceptId id) {
    auto c = ltm.retrieve_concept(id);
    return c ? c->label : ("?" + std::to_string(id));
}

// Print all relations (outgoing + incoming) for a concept
static void print_full_neighborhood(const LongTermMemory& ltm, ConceptId cid) {
    auto info = ltm.retrieve_concept(cid);
    if (!info) { std::cout << "  [concept not found]\n"; return; }

    std::cout << "  ID=" << cid << "  Label=\"" << info->label << "\"\n";
    std::cout << "  Definition: " << info->definition.substr(0, 120) << "\n";

    auto out = ltm.get_outgoing_relations(cid);
    auto in  = ltm.get_incoming_relations(cid);
    std::cout << "  Outgoing: " << out.size() << "  Incoming: " << in.size()
              << "  Total degree: " << (out.size() + in.size()) << "\n";

    if (!out.empty()) {
        std::cout << "  --- OUTGOING ---\n";
        for (const auto& r : out) {
            std::cout << "    -> " << std::left << std::setw(30)
                      << concept_label(ltm, r.target)
                      << "  [" << std::setw(16) << rel_name(r.type) << "]"
                      << "  w=" << std::fixed << std::setprecision(3) << r.weight
                      << "\n";
        }
    }
    if (!in.empty()) {
        std::cout << "  --- INCOMING ---\n";
        for (const auto& r : in) {
            std::cout << "    <- " << std::left << std::setw(30)
                      << concept_label(ltm, r.source)
                      << "  [" << std::setw(16) << rel_name(r.type) << "]"
                      << "  w=" << std::fixed << std::setprecision(3) << r.weight
                      << "\n";
        }
    }
}

// Detect domain using DomainManager
static DomainType detect(const DomainManager& dm, ConceptId cid, const LongTermMemory& ltm) {
    return dm.detect_domain(cid, ltm);
}

// ============================================================================
// MAIN
// ============================================================================
int main() {
    std::cout << "================================================================\n"
              << " Brain19 Graph Structure Analysis — Domain Drift Diagnostics\n"
              << "================================================================\n\n";

    LongTermMemory ltm;
    bool ok = FoundationConcepts::seed_from_file(ltm, "../data/foundation_full.json");
    if (!ok) {
        std::cerr << "FATAL: Could not load foundation_full.json\n";
        return 1;
    }

    auto all_ids = ltm.get_all_concept_ids();
    std::cout << "Loaded: " << all_ids.size() << " concepts, "
              << ltm.total_relation_count() << " relations\n\n";

    DomainManager dm;

    // =====================================================================
    // 1. Full neighborhood of key seed concepts
    // =====================================================================
    std::cout << "================================================================\n"
              << " SECTION 1: Full Neighborhood of Key Seed Concepts\n"
              << "================================================================\n\n";

    std::vector<std::string> seeds = {
        "Photosynthesis", "Primary Production", "Base", "Replication",
        "Volume", "Work", "DNA"
    };

    for (const auto& label : seeds) {
        std::cout << "--- " << label << " ---\n";
        auto ids = ltm.find_by_label(label);
        if (ids.empty()) {
            std::cout << "  NOT FOUND in label index!\n\n";
            continue;
        }
        for (auto cid : ids) {
            print_full_neighborhood(ltm, cid);
            auto d = detect(dm, cid, ltm);
            std::cout << "  Detected domain: " << domain_to_string(d) << "\n";
        }
        std::cout << "\n";
    }

    // =====================================================================
    // 2. Universal bridge properties: how many concepts connect to
    //    Volume, Mass, Density via HAS_PROPERTY?
    // =====================================================================
    std::cout << "================================================================\n"
              << " SECTION 2: Universal Bridge Properties\n"
              << "================================================================\n\n";

    std::vector<std::string> bridge_props = {
        "Volume", "Mass", "Density", "Energy", "Temperature",
        "Pressure", "Force", "Velocity", "Time", "Space",
        "Shape", "Color", "Size", "Weight", "Length"
    };

    struct BridgeStat {
        std::string label;
        ConceptId cid = 0;
        size_t has_property_incoming = 0;   // X --HAS_PROPERTY--> this
        size_t total_incoming = 0;
        size_t total_outgoing = 0;
        std::vector<std::string> connected_labels;  // Who connects via HAS_PROPERTY
    };

    std::vector<BridgeStat> bridge_stats;

    for (const auto& prop_label : bridge_props) {
        auto ids = ltm.find_by_label(prop_label);
        if (ids.empty()) continue;

        for (auto pid : ids) {
            BridgeStat bs;
            bs.label = prop_label;
            bs.cid = pid;

            auto in = ltm.get_incoming_relations(pid);
            auto out = ltm.get_outgoing_relations(pid);
            bs.total_incoming = in.size();
            bs.total_outgoing = out.size();

            for (const auto& r : in) {
                if (r.type == RelationType::HAS_PROPERTY) {
                    bs.has_property_incoming++;
                    bs.connected_labels.push_back(concept_label(ltm, r.source));
                }
            }

            bridge_stats.push_back(bs);
        }
    }

    // Sort by HAS_PROPERTY count descending
    std::sort(bridge_stats.begin(), bridge_stats.end(),
              [](const BridgeStat& a, const BridgeStat& b) {
                  return a.has_property_incoming > b.has_property_incoming;
              });

    std::cout << std::left
              << std::setw(20) << "Property"
              << std::setw(10) << "HP-In"
              << std::setw(10) << "Tot-In"
              << std::setw(10) << "Tot-Out"
              << "Connected Concepts (via HAS_PROPERTY)\n";
    std::cout << std::string(90, '-') << "\n";

    for (const auto& bs : bridge_stats) {
        std::cout << std::left
                  << std::setw(20) << bs.label
                  << std::setw(10) << bs.has_property_incoming
                  << std::setw(10) << bs.total_incoming
                  << std::setw(10) << bs.total_outgoing;
        for (size_t i = 0; i < std::min(bs.connected_labels.size(), (size_t)10); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << bs.connected_labels[i];
        }
        if (bs.connected_labels.size() > 10)
            std::cout << " ... (+" << (bs.connected_labels.size() - 10) << " more)";
        std::cout << "\n";
    }
    std::cout << "\n";

    // =====================================================================
    // 3. Polysemous concept deep dive: Replication, Base, Work, Volume
    // =====================================================================
    std::cout << "================================================================\n"
              << " SECTION 3: Polysemous Concept Deep Dive\n"
              << "================================================================\n\n";

    std::vector<std::string> polysemous_suspects = {
        "Base", "Replication", "Work", "Volume", "Cell",
        "Field", "Ring", "Group", "Order", "Power",
        "Function", "Pressure", "Resistance", "Current",
        "Charge", "Bond", "Medium", "Culture", "Light",
        "Spectrum", "Channel", "Protocol", "Model", "Network"
    };

    for (const auto& label : polysemous_suspects) {
        auto ids = ltm.find_by_label(label);
        if (ids.empty()) continue;

        std::cout << "--- " << label << " ---\n";
        for (auto cid : ids) {
            auto info = ltm.retrieve_concept(cid);
            if (!info) continue;

            auto out = ltm.get_outgoing_relations(cid);
            auto in  = ltm.get_incoming_relations(cid);

            // Detect domains of all neighbors
            std::unordered_map<DomainType, std::vector<std::string>, DomainTypeHash> neighbor_domains;
            std::unordered_set<ConceptId> seen;

            for (const auto& r : out) {
                if (seen.insert(r.target).second) {
                    auto d = detect(dm, r.target, ltm);
                    neighbor_domains[d].push_back(concept_label(ltm, r.target));
                }
            }
            for (const auto& r : in) {
                if (seen.insert(r.source).second) {
                    auto d = detect(dm, r.source, ltm);
                    neighbor_domains[d].push_back(concept_label(ltm, r.source));
                }
            }

            std::cout << "  ID=" << cid
                      << " Def: " << info->definition.substr(0, 100)
                      << "\n  Degree: out=" << out.size() << " in=" << in.size()
                      << " total=" << (out.size() + in.size())
                      << "\n  Neighbor domains (" << neighbor_domains.size() << " distinct):\n";

            for (const auto& [dom, names] : neighbor_domains) {
                std::cout << "    " << domain_to_string(dom) << " (" << names.size() << "): ";
                for (size_t i = 0; i < std::min(names.size(), (size_t)8); ++i) {
                    if (i) std::cout << ", ";
                    std::cout << names[i];
                }
                if (names.size() > 8) std::cout << " ...";
                std::cout << "\n";
            }
        }
        std::cout << "\n";
    }

    // =====================================================================
    // 4. Hub Census & Polysemous Bridge Nodes
    // =====================================================================
    std::cout << "================================================================\n"
              << " SECTION 4: Hub Census & Polysemous Bridge Nodes\n"
              << "================================================================\n\n";

    struct NodeProfile {
        ConceptId id;
        std::string label;
        size_t out_degree = 0;
        size_t in_degree = 0;
        size_t total_degree = 0;
        size_t distinct_neighbor_domains = 0;
        std::unordered_map<DomainType, size_t, DomainTypeHash> domain_counts;
    };

    std::vector<NodeProfile> profiles;
    profiles.reserve(all_ids.size());

    for (auto cid : all_ids) {
        NodeProfile np;
        np.id = cid;
        np.label = concept_label(ltm, cid);

        auto out = ltm.get_outgoing_relations(cid);
        auto in  = ltm.get_incoming_relations(cid);
        np.out_degree = out.size();
        np.in_degree  = in.size();
        np.total_degree = np.out_degree + np.in_degree;

        // Detect domains of all distinct neighbors
        std::unordered_set<ConceptId> seen;
        for (const auto& r : out) {
            if (seen.insert(r.target).second) {
                auto d = detect(dm, r.target, ltm);
                np.domain_counts[d]++;
            }
        }
        for (const auto& r : in) {
            if (seen.insert(r.source).second) {
                auto d = detect(dm, r.source, ltm);
                np.domain_counts[d]++;
            }
        }
        np.distinct_neighbor_domains = np.domain_counts.size();
        profiles.push_back(std::move(np));
    }

    // --- 4a: Top 30 by total degree ---
    std::sort(profiles.begin(), profiles.end(),
              [](const NodeProfile& a, const NodeProfile& b) {
                  return a.total_degree > b.total_degree;
              });

    std::cout << "--- Top 30 Nodes by Total Degree ---\n";
    std::cout << std::left
              << std::setw(6)  << "Rank"
              << std::setw(30) << "Concept"
              << std::setw(8)  << "Out"
              << std::setw(8)  << "In"
              << std::setw(8)  << "Total"
              << std::setw(10) << "Domains"
              << "Domain Distribution\n";
    std::cout << std::string(100, '-') << "\n";

    for (size_t i = 0; i < std::min(profiles.size(), (size_t)30); ++i) {
        const auto& p = profiles[i];
        std::cout << std::left
                  << std::setw(6)  << (i + 1)
                  << std::setw(30) << p.label.substr(0, 28)
                  << std::setw(8)  << p.out_degree
                  << std::setw(8)  << p.in_degree
                  << std::setw(8)  << p.total_degree
                  << std::setw(10) << p.distinct_neighbor_domains;
        for (const auto& [dom, cnt] : p.domain_counts) {
            std::cout << domain_to_string(dom) << ":" << cnt << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";

    // --- 4b: Polysemous bridge nodes (3+ distinct neighbor domains) ---
    std::vector<const NodeProfile*> bridges;
    for (const auto& p : profiles) {
        if (p.distinct_neighbor_domains >= 3) {
            bridges.push_back(&p);
        }
    }

    std::sort(bridges.begin(), bridges.end(),
              [](const NodeProfile* a, const NodeProfile* b) {
                  return a->total_degree > b->total_degree;
              });

    std::cout << "--- Polysemous Bridge Nodes (3+ distinct neighbor domains) ---\n";
    std::cout << "Total: " << bridges.size() << " out of " << all_ids.size()
              << " concepts (" << std::fixed << std::setprecision(1)
              << (100.0 * bridges.size() / all_ids.size()) << "%)\n\n";

    std::cout << std::left
              << std::setw(30) << "Concept"
              << std::setw(8)  << "Degree"
              << std::setw(10) << "Domains"
              << "Domain Distribution\n";
    std::cout << std::string(100, '-') << "\n";

    for (size_t i = 0; i < std::min(bridges.size(), (size_t)50); ++i) {
        const auto& p = *bridges[i];
        std::cout << std::left
                  << std::setw(30) << p.label.substr(0, 28)
                  << std::setw(8)  << p.total_degree
                  << std::setw(10) << p.distinct_neighbor_domains;
        for (const auto& [dom, cnt] : p.domain_counts) {
            std::cout << domain_to_string(dom) << ":" << cnt << " ";
        }
        std::cout << "\n";
    }
    if (bridges.size() > 50) {
        std::cout << "... and " << (bridges.size() - 50) << " more\n";
    }
    std::cout << "\n";

    // --- 4c: Degree distribution summary ---
    std::cout << "--- Degree Distribution Summary ---\n";
    size_t deg_1 = 0, deg_2_5 = 0, deg_6_10 = 0, deg_11_20 = 0, deg_21_50 = 0, deg_50p = 0;
    for (const auto& p : profiles) {
        if (p.total_degree <= 1) deg_1++;
        else if (p.total_degree <= 5) deg_2_5++;
        else if (p.total_degree <= 10) deg_6_10++;
        else if (p.total_degree <= 20) deg_11_20++;
        else if (p.total_degree <= 50) deg_21_50++;
        else deg_50p++;
    }
    std::cout << "  Degree  0-1:   " << deg_1 << "\n"
              << "  Degree  2-5:   " << deg_2_5 << "\n"
              << "  Degree  6-10:  " << deg_6_10 << "\n"
              << "  Degree 11-20:  " << deg_11_20 << "\n"
              << "  Degree 21-50:  " << deg_21_50 << "\n"
              << "  Degree 50+:    " << deg_50p << "\n\n";

    // --- 4d: Relation type distribution ---
    std::cout << "--- Relation Type Distribution ---\n";
    std::map<std::string, size_t> rel_type_counts;
    for (auto cid : all_ids) {
        for (const auto& r : ltm.get_outgoing_relations(cid)) {
            rel_type_counts[rel_name(r.type)]++;
        }
    }
    for (const auto& [name, count] : rel_type_counts) {
        std::cout << "  " << std::left << std::setw(20) << name << " " << count << "\n";
    }
    std::cout << "\n";

    // --- 4e: HAS_PROPERTY target frequency (top targets) ---
    std::cout << "--- HAS_PROPERTY Top Targets (most-connected properties) ---\n";
    std::unordered_map<ConceptId, size_t> hp_target_count;
    for (auto cid : all_ids) {
        for (const auto& r : ltm.get_outgoing_relations(cid)) {
            if (r.type == RelationType::HAS_PROPERTY) {
                hp_target_count[r.target]++;
            }
        }
    }
    // Sort
    std::vector<std::pair<ConceptId, size_t>> hp_sorted(hp_target_count.begin(), hp_target_count.end());
    std::sort(hp_sorted.begin(), hp_sorted.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    std::cout << std::left
              << std::setw(30) << "Property Target"
              << std::setw(10) << "Count"
              << "Sources (sample)\n";
    std::cout << std::string(90, '-') << "\n";

    for (size_t i = 0; i < std::min(hp_sorted.size(), (size_t)30); ++i) {
        auto [target_id, count] = hp_sorted[i];
        std::cout << std::left
                  << std::setw(30) << concept_label(ltm, target_id).substr(0, 28)
                  << std::setw(10) << count;

        // Show up to 5 source concepts
        size_t shown = 0;
        for (auto cid : all_ids) {
            if (shown >= 5) break;
            for (const auto& r : ltm.get_outgoing_relations(cid)) {
                if (r.type == RelationType::HAS_PROPERTY && r.target == target_id) {
                    if (shown > 0) std::cout << ", ";
                    std::cout << concept_label(ltm, cid);
                    shown++;
                    break;
                }
            }
        }
        std::cout << "\n";
    }
    std::cout << "\n";

    // =====================================================================
    // 5. Specific Drift Path Analysis
    // =====================================================================
    std::cout << "================================================================\n"
              << " SECTION 5: Drift Path Analysis\n"
              << "================================================================\n\n";

    // Show how Photosynthesis can reach unrelated domains in 2 hops
    std::cout << "--- 2-Hop Reachability from Photosynthesis ---\n";
    auto photo_ids = ltm.find_by_label("Photosynthesis");
    if (!photo_ids.empty()) {
        ConceptId photo = photo_ids[0];
        std::unordered_set<ConceptId> hop1_nodes;
        std::unordered_map<DomainType, std::unordered_set<ConceptId>, DomainTypeHash> hop2_by_domain;

        // Hop 1
        auto photo_out = ltm.get_outgoing_relations(photo);
        auto photo_in  = ltm.get_incoming_relations(photo);
        for (const auto& r : photo_out) hop1_nodes.insert(r.target);
        for (const auto& r : photo_in)  hop1_nodes.insert(r.source);

        std::cout << "  Hop-1 neighbors: " << hop1_nodes.size() << "\n";

        // Hop 2
        std::unordered_set<ConceptId> hop2_nodes;
        for (auto h1 : hop1_nodes) {
            auto h1_out = ltm.get_outgoing_relations(h1);
            auto h1_in  = ltm.get_incoming_relations(h1);
            for (const auto& r : h1_out) {
                if (r.target != photo && hop1_nodes.find(r.target) == hop1_nodes.end()) {
                    hop2_nodes.insert(r.target);
                    auto d = detect(dm, r.target, ltm);
                    hop2_by_domain[d].insert(r.target);
                }
            }
            for (const auto& r : h1_in) {
                if (r.source != photo && hop1_nodes.find(r.source) == hop1_nodes.end()) {
                    hop2_nodes.insert(r.source);
                    auto d = detect(dm, r.source, ltm);
                    hop2_by_domain[d].insert(r.source);
                }
            }
        }

        std::cout << "  Hop-2 unique nodes: " << hop2_nodes.size() << "\n";
        std::cout << "  Hop-2 by domain:\n";
        for (const auto& [dom, nodes] : hop2_by_domain) {
            std::cout << "    " << domain_to_string(dom) << ": " << nodes.size() << " concepts — ";
            size_t shown = 0;
            for (auto cid : nodes) {
                if (shown >= 8) { std::cout << "..."; break; }
                if (shown > 0) std::cout << ", ";
                std::cout << concept_label(ltm, cid);
                shown++;
            }
            std::cout << "\n";
        }
    }
    std::cout << "\n";

    // Show DNA's connections for biology vs tech ambiguity
    std::cout << "--- 2-Hop Reachability from DNA ---\n";
    auto dna_ids = ltm.find_by_label("DNA");
    if (!dna_ids.empty()) {
        ConceptId dna = dna_ids[0];
        std::unordered_set<ConceptId> hop1;
        auto dna_out = ltm.get_outgoing_relations(dna);
        auto dna_in  = ltm.get_incoming_relations(dna);
        for (const auto& r : dna_out) hop1.insert(r.target);
        for (const auto& r : dna_in)  hop1.insert(r.source);

        std::cout << "  Hop-1 neighbors: " << hop1.size() << "\n";

        std::unordered_map<DomainType, size_t, DomainTypeHash> hop2_domains;
        size_t hop2_total = 0;
        for (auto h1 : hop1) {
            auto h1_out = ltm.get_outgoing_relations(h1);
            auto h1_in  = ltm.get_incoming_relations(h1);
            for (const auto& r : h1_out) {
                if (r.target != dna && hop1.find(r.target) == hop1.end()) {
                    hop2_domains[detect(dm, r.target, ltm)]++;
                    hop2_total++;
                }
            }
            for (const auto& r : h1_in) {
                if (r.source != dna && hop1.find(r.source) == hop1.end()) {
                    hop2_domains[detect(dm, r.source, ltm)]++;
                    hop2_total++;
                }
            }
        }
        std::cout << "  Hop-2 (non-unique) edges: " << hop2_total << "\n";
        for (const auto& [dom, cnt] : hop2_domains) {
            std::cout << "    " << domain_to_string(dom) << ": " << cnt << "\n";
        }
    }
    std::cout << "\n";

    std::cout << "================================================================\n"
              << " Analysis Complete\n"
              << "================================================================\n";

    return 0;
}
