#include "pattern_discovery.hpp"
#include <algorithm>
#include <queue>
#include <unordered_set>

namespace brain19 {

PatternDiscovery::PatternDiscovery(const LongTermMemory& ltm)
    : ltm_(ltm)
{
}

PatternDiscovery::AdjacencyGraph PatternDiscovery::build_graph() const {
    AdjacencyGraph graph;
    auto all_ids = ltm_.get_active_concepts();
    graph.nodes = all_ids;

    for (auto id : all_ids) {
        auto outgoing = ltm_.get_outgoing_relations(id);
        for (const auto& rel : outgoing) {
            // Only include relations to active concepts
            auto target = ltm_.retrieve_concept(rel.target);
            if (target && target->epistemic.is_active()) {
                graph.adj[id].push_back(rel.target);
                if (rel.type == RelationType::IS_A) {
                    graph.adj_typed[id].push_back(rel.target);
                }
            }
        }
    }

    return graph;
}

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

std::vector<DiscoveredPattern> PatternDiscovery::find_clusters(size_t min_size) {
    std::vector<DiscoveredPattern> patterns;
    auto graph = build_graph();
    auto components = find_components(graph);

    for (const auto& comp : components) {
        if (comp.size() < min_size) continue;

        // Compute density: edges / (n*(n-1))
        size_t edge_count = 0;
        for (auto node : comp) {
            if (graph.adj.count(node)) {
                std::unordered_set<ConceptId> comp_set(comp.begin(), comp.end());
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

        patterns.emplace_back(
            "Cluster of " + std::to_string(n) + " concepts (density=" +
            std::to_string(density) + ")",
            comp,
            density,
            "cluster"
        );
    }

    return patterns;
}

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
            patterns.emplace_back(
                "IS_A hierarchy of depth " + std::to_string(chain.size()),
                chain,
                0.9,
                "hierarchy"
            );
        }
    }

    return patterns;
}

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
    // (This catches near-bridges — concepts with weak cross-component links)
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
            patterns.emplace_back(
                "Bridge concept connecting " +
                std::to_string(connected_components.size()) + " clusters",
                std::vector<ConceptId>{node},
                0.8,
                "bridge"
            );
        }
    }

    return patterns;
}

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
            patterns.emplace_back(
                "Cycle of length " + std::to_string(cycle.size()),
                cycle,
                0.7,
                "cycle"
            );
        }

        checked.insert(start);
    }

    return patterns;
}

std::vector<DiscoveredPattern> PatternDiscovery::find_gaps() {
    std::vector<DiscoveredPattern> patterns;
    auto all_ids = ltm_.get_active_concepts();

    for (auto id : all_ids) {
        auto outgoing = ltm_.get_outgoing_relations(id);

        // For each IS_A parent, check if siblings share relations we don't have
        for (const auto& rel : outgoing) {
            if (rel.type != RelationType::IS_A) continue;

            ConceptId parent = rel.target;
            auto parent_incoming = ltm_.get_incoming_relations(parent);

            // Find siblings (other concepts that IS_A the same parent)
            std::vector<ConceptId> siblings;
            for (const auto& sibling_rel : parent_incoming) {
                if (sibling_rel.type == RelationType::IS_A &&
                    sibling_rel.source != id) {
                    siblings.push_back(sibling_rel.source);
                }
            }

            // Check what relations siblings have that we don't
            for (auto sibling : siblings) {
                auto sibling_rels = ltm_.get_outgoing_relations(sibling);
                for (const auto& srel : sibling_rels) {
                    if (srel.type == RelationType::IS_A) continue;

                    // Does id have a similar relation?
                    auto our_rels = ltm_.get_relations_between(id, srel.target);
                    if (our_rels.empty()) {
                        patterns.emplace_back(
                            "Gap: concept " + std::to_string(id) +
                            " may need " + relation_type_to_string(srel.type) +
                            " to " + std::to_string(srel.target) +
                            " (sibling " + std::to_string(sibling) + " has it)",
                            std::vector<ConceptId>{id, srel.target, sibling},
                            0.5,
                            "gap"
                        );
                    }
                }
            }
        }
    }

    return patterns;
}

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
