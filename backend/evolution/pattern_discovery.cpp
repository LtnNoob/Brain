#include "pattern_discovery.hpp"
#include <algorithm>
#include <queue>
#include <unordered_set>

namespace brain19 {

PatternDiscovery::PatternDiscovery(const LongTermMemory& ltm)
    : ltm_(ltm)
{
}

// =============================================================================
// Build adjacency — filters anti-knowledge, includes all relation types
// =============================================================================

PatternDiscovery::AdjacencyGraph PatternDiscovery::build_graph() const {
    AdjacencyGraph graph;
    auto all_ids = ltm_.get_active_concepts();

    // First pass: collect valid nodes (active AND not anti-knowledge)
    for (auto id : all_ids) {
        auto cinfo = ltm_.retrieve_concept(id);
        if (!cinfo) continue;
        if (cinfo->is_anti_knowledge) continue;  // Filter anti-knowledge
        if (!cinfo->epistemic.is_active()) continue;
        graph.nodes.push_back(id);
        graph.node_set.insert(id);
    }

    // Second pass: build adjacency from valid nodes only
    for (auto id : graph.nodes) {
        auto outgoing = ltm_.get_outgoing_relations(id);
        for (const auto& rel : outgoing) {
            // Only include edges to valid nodes
            if (graph.node_set.count(rel.target) == 0) continue;

            graph.adj[id].push_back(rel.target);
            if (rel.type == RelationType::IS_A) {
                graph.adj_typed[id].push_back(rel.target);
            }
        }
    }

    return graph;
}

// =============================================================================
// Average trust of involved concepts
// =============================================================================

double PatternDiscovery::avg_trust(const std::vector<ConceptId>& concepts) const {
    if (concepts.empty()) return 0.5;
    double sum = 0.0;
    size_t count = 0;
    for (ConceptId cid : concepts) {
        auto cinfo = ltm_.retrieve_concept(cid);
        if (cinfo) {
            sum += cinfo->epistemic.trust;
            ++count;
        }
    }
    return (count > 0) ? sum / static_cast<double>(count) : 0.5;
}

// =============================================================================
// Connected components via BFS
// =============================================================================

std::vector<std::vector<ConceptId>> PatternDiscovery::find_components(
    const AdjacencyGraph& graph) const
{
    std::vector<std::vector<ConceptId>> components;
    std::unordered_set<ConceptId> visited;

    // Build undirected adjacency
    std::unordered_map<ConceptId, std::unordered_set<ConceptId>> undirected;
    for (const auto& [node, neighbors] : graph.adj) {
        for (auto n : neighbors) {
            undirected[node].insert(n);
            undirected[n].insert(node);
        }
    }

    for (auto node : graph.nodes) {
        if (visited.count(node)) continue;

        std::vector<ConceptId> component;
        std::queue<ConceptId> q;
        q.push(node);
        visited.insert(node);

        while (!q.empty()) {
            auto current = q.front();
            q.pop();
            component.push_back(current);

            if (undirected.count(current)) {
                for (auto neighbor : undirected.at(current)) {
                    if (!visited.count(neighbor)) {
                        visited.insert(neighbor);
                        q.push(neighbor);
                    }
                }
            }
        }

        components.push_back(std::move(component));
    }

    return components;
}

// =============================================================================
// Clusters — trust-weighted confidence
// =============================================================================

std::vector<DiscoveredPattern> PatternDiscovery::find_clusters(size_t min_size) {
    std::vector<DiscoveredPattern> patterns;
    auto graph = build_graph();
    auto components = find_components(graph);

    for (const auto& comp : components) {
        if (comp.size() < min_size) continue;

        // Compute density: edges / (n*(n-1))
        size_t edge_count = 0;
        std::unordered_set<ConceptId> comp_set(comp.begin(), comp.end());
        for (auto node : comp) {
            if (graph.adj.count(node)) {
                for (auto neighbor : graph.adj.at(node)) {
                    if (comp_set.count(neighbor)) {
                        ++edge_count;
                    }
                }
            }
        }

        size_t n = comp.size();
        double max_edges = static_cast<double>(n * (n - 1));
        double density = (max_edges > 0) ? static_cast<double>(edge_count) / max_edges : 0.0;

        // Confidence = density * average trust of cluster members
        double trust = avg_trust(comp);

        patterns.emplace_back(
            "Cluster of " + std::to_string(n) + " concepts (density=" +
            std::to_string(density) + ", trust=" + std::to_string(trust) + ")",
            comp,
            density * trust,
            "cluster"
        );
    }

    return patterns;
}

// =============================================================================
// Hierarchies — trust-weighted
// =============================================================================

std::vector<DiscoveredPattern> PatternDiscovery::find_hierarchies(size_t min_depth) {
    std::vector<DiscoveredPattern> patterns;
    auto graph = build_graph();

    // Follow IS_A chains from each node
    for (auto start : graph.nodes) {
        if (!graph.adj_typed.count(start)) continue;

        std::vector<ConceptId> chain;
        std::unordered_set<ConceptId> in_chain;
        chain.push_back(start);
        in_chain.insert(start);

        ConceptId current = start;
        while (graph.adj_typed.count(current) && !graph.adj_typed.at(current).empty()) {
            ConceptId next = graph.adj_typed.at(current)[0];
            if (in_chain.count(next)) break;  // Cycle in IS_A
            chain.push_back(next);
            in_chain.insert(next);
            current = next;
        }

        if (chain.size() >= min_depth) {
            double trust = avg_trust(chain);
            patterns.emplace_back(
                "IS_A hierarchy of depth " + std::to_string(chain.size()) +
                " (trust=" + std::to_string(trust) + ")",
                chain,
                trust,  // trust IS the confidence
                "hierarchy"
            );
        }
    }

    return patterns;
}

// =============================================================================
// Bridges — trust-weighted
// =============================================================================

std::vector<DiscoveredPattern> PatternDiscovery::find_bridges() {
    std::vector<DiscoveredPattern> patterns;
    auto graph = build_graph();
    auto components = find_components(graph);

    if (components.size() < 2) return patterns;

    // Build component membership map
    std::unordered_map<ConceptId, size_t> comp_index;
    for (size_t i = 0; i < components.size(); ++i) {
        for (auto node : components[i]) {
            comp_index[node] = i;
        }
    }

    // Find concepts that have relations to concepts in other components
    for (auto node : graph.nodes) {
        if (!graph.adj.count(node)) continue;

        std::unordered_set<size_t> connected_components;
        connected_components.insert(comp_index[node]);

        for (auto neighbor : graph.adj.at(node)) {
            if (comp_index.count(neighbor)) {
                connected_components.insert(comp_index[neighbor]);
            }
        }

        // Also check incoming
        auto incoming = ltm_.get_incoming_relations(node);
        for (const auto& rel : incoming) {
            if (comp_index.count(rel.source)) {
                connected_components.insert(comp_index[rel.source]);
            }
        }

        if (connected_components.size() >= 2) {
            auto cinfo = ltm_.retrieve_concept(node);
            double trust = cinfo ? cinfo->epistemic.trust : 0.5;

            patterns.emplace_back(
                "Bridge concept connecting " +
                std::to_string(connected_components.size()) + " clusters" +
                " (trust=" + std::to_string(trust) + ")",
                std::vector<ConceptId>{node},
                trust,  // bridge confidence = concept's own trust
                "bridge"
            );
        }
    }

    return patterns;
}

// =============================================================================
// Cycles — trust-weighted
// =============================================================================

bool PatternDiscovery::dfs_cycles(
    ConceptId node, ConceptId start,
    std::vector<ConceptId>& path,
    std::unordered_set<ConceptId>& visited,
    const AdjacencyGraph& graph,
    size_t max_length,
    std::vector<std::vector<ConceptId>>& found_cycles) const
{
    if (path.size() > max_length) return false;

    if (!graph.adj.count(node)) return false;

    for (auto neighbor : graph.adj.at(node)) {
        if (neighbor == start && path.size() >= 2) {
            found_cycles.push_back(path);
            return true;
        }

        if (!visited.count(neighbor) && path.size() < max_length) {
            visited.insert(neighbor);
            path.push_back(neighbor);
            dfs_cycles(neighbor, start, path, visited, graph,
                       max_length, found_cycles);
            path.pop_back();
            visited.erase(neighbor);
        }
    }

    return false;
}

std::vector<DiscoveredPattern> PatternDiscovery::find_cycles(size_t max_length) {
    std::vector<DiscoveredPattern> patterns;
    auto graph = build_graph();

    std::unordered_set<ConceptId> checked;

    for (auto start : graph.nodes) {
        if (checked.count(start)) continue;

        std::vector<ConceptId> path = {start};
        std::unordered_set<ConceptId> visited = {start};
        std::vector<std::vector<ConceptId>> found;

        dfs_cycles(start, start, path, visited, graph, max_length, found);

        for (auto& cycle : found) {
            double trust = avg_trust(cycle);
            patterns.emplace_back(
                "Cycle of length " + std::to_string(cycle.size()) +
                " (trust=" + std::to_string(trust) + ")",
                cycle,
                trust * 0.7,  // cycles are less certain, scale by 0.7
                "cycle"
            );
        }

        checked.insert(start);
    }

    return patterns;
}

// =============================================================================
// Gaps — trust-weighted, all relation types
// =============================================================================

std::vector<DiscoveredPattern> PatternDiscovery::find_gaps() {
    std::vector<DiscoveredPattern> patterns;
    auto graph = build_graph();

    for (auto id : graph.nodes) {
        auto outgoing = ltm_.get_outgoing_relations(id);

        // For each IS_A parent, check if siblings share relations we don't have
        for (const auto& rel : outgoing) {
            if (rel.type != RelationType::IS_A) continue;

            ConceptId parent = rel.target;
            if (graph.node_set.count(parent) == 0) continue;  // parent filtered out

            auto parent_incoming = ltm_.get_incoming_relations(parent);

            // Find siblings (other concepts that IS_A the same parent)
            std::vector<ConceptId> siblings;
            for (const auto& sibling_rel : parent_incoming) {
                if (sibling_rel.type == RelationType::IS_A &&
                    sibling_rel.source != id &&
                    graph.node_set.count(sibling_rel.source)) {  // sibling must be valid
                    siblings.push_back(sibling_rel.source);
                }
            }

            // Check what relations siblings have that we don't
            for (auto sibling : siblings) {
                auto sibling_rels = ltm_.get_outgoing_relations(sibling);
                for (const auto& srel : sibling_rels) {
                    if (srel.type == RelationType::IS_A) continue;
                    if (graph.node_set.count(srel.target) == 0) continue;  // target filtered

                    // Does id have a similar relation?
                    auto our_rels = ltm_.get_relations_between(id, srel.target);
                    if (our_rels.empty()) {
                        // Trust-weighted: use min trust of involved concepts
                        auto id_info = ltm_.retrieve_concept(id);
                        auto target_info = ltm_.retrieve_concept(srel.target);
                        double id_trust = id_info ? id_info->epistemic.trust : 0.5;
                        double target_trust = target_info ? target_info->epistemic.trust : 0.5;
                        double conf = std::min(id_trust, target_trust) * 0.5;

                        patterns.emplace_back(
                            "Gap: concept " + std::to_string(id) +
                            " may need " + relation_type_to_string(srel.type) +
                            " to " + std::to_string(srel.target) +
                            " (sibling " + std::to_string(sibling) + " has it)",
                            std::vector<ConceptId>{id, srel.target, sibling},
                            conf,
                            "gap",
                            srel.type  // Store the actual missing relation type
                        );
                    }
                }
            }
        }
    }

    return patterns;
}

// =============================================================================
// Full discovery
// =============================================================================

std::vector<DiscoveredPattern> PatternDiscovery::discover_all() {
    std::vector<DiscoveredPattern> all;

    auto clusters = find_clusters();
    all.insert(all.end(), clusters.begin(), clusters.end());

    auto hierarchies = find_hierarchies();
    all.insert(all.end(), hierarchies.begin(), hierarchies.end());

    auto bridges = find_bridges();
    all.insert(all.end(), bridges.begin(), bridges.end());

    auto cycles = find_cycles();
    all.insert(all.end(), cycles.begin(), cycles.end());

    auto gaps = find_gaps();
    all.insert(all.end(), gaps.begin(), gaps.end());

    return all;
}

} // namespace brain19
