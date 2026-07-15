#include "h2h_graph.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>

#include "h2h_format.hpp"

namespace h2h {
namespace {

// 从一个起点遍历邻接表，返回可达节点标记。
std::vector<std::uint8_t> reachable_from(
    const std::vector<std::vector<std::uint32_t>>& adjacency,
    std::uint32_t start
) {
    std::vector<std::uint8_t> visited(adjacency.size(), 0);
    std::vector<std::uint32_t> stack{start};
    visited[start] = 1;
    while (!stack.empty()) {
        const auto node = stack.back();
        stack.pop_back();
        for (const auto neighbor : adjacency[node]) {
            if (!visited[neighbor]) {
                visited[neighbor] = 1;
                stack.push_back(neighbor);
            }
        }
    }
    return visited;
}

// 校验规范化有限弧形成强连通有向图。
void validate_strong_connectivity(const DirectedGraph& graph) {
    std::vector<std::vector<std::uint32_t>> forward(graph.node_count);
    std::vector<std::vector<std::uint32_t>> reverse(graph.node_count);
    for (std::uint32_t source = 0; source < graph.node_count; ++source) {
        forward[source].reserve(graph.out[source].size());
        for (const auto& [target, weight] : graph.out[source]) {
            (void)weight;
            forward[source].push_back(target);
            reverse[target].push_back(source);
        }
    }
    const auto forward_seen = reachable_from(forward, 0);
    const auto reverse_seen = reachable_from(reverse, 0);
    const auto forward_count = std::count(forward_seen.begin(), forward_seen.end(), 1);
    const auto reverse_count = std::count(reverse_seen.begin(), reverse_seen.end(), 1);
    if (forward_count != graph.node_count || reverse_count != graph.node_count) {
        throw std::runtime_error(
            "规范化图不是强连通图：从节点 0 正向可达 " + std::to_string(forward_count) +
            " 个节点，反向可达 " + std::to_string(reverse_count) + " 个节点。"
        );
    }
}

}  // namespace

DirectedGraph read_graph_binary(const std::filesystem::path& path) {
    // graph.bin 是阶段 3 builder 的唯一输入，必须在分配大数组前验证文件头。
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("无法打开规范化图文件：" + path.string());
    }
    GraphHeader header{};
    input.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!input) {
        throw std::runtime_error("规范化图文件头不完整：" + path.string());
    }
    if (std::memcmp(header.magic, kGraphMagic.data(), kGraphMagic.size()) != 0) {
        throw std::runtime_error("规范化图 magic 不匹配：" + path.string());
    }
    if (header.format_version != kGraphFormatVersion) {
        throw std::runtime_error("不支持的规范化图格式版本。 ");
    }
    if (header.endian_marker != kEndianMarker) {
        throw std::runtime_error("规范化图不是当前要求的小端序格式。 ");
    }
    if (header.node_count == 0) {
        throw std::runtime_error("规范化图不能包含 0 个节点。 ");
    }
    if (header.edge_count > (std::numeric_limits<std::uint64_t>::max() - sizeof(GraphHeader)) /
            sizeof(GraphEdgeRecord)) {
        throw std::runtime_error("规范化图边数导致文件大小溢出。 ");
    }
    const auto expected_size = static_cast<std::uint64_t>(sizeof(GraphHeader)) +
        header.edge_count * sizeof(GraphEdgeRecord);
    const auto actual_size = std::filesystem::file_size(path);
    if (actual_size != expected_size) {
        throw std::runtime_error(
            "规范化图文件大小不匹配：期望 " + std::to_string(expected_size) +
            "，实际 " + std::to_string(actual_size) + "。"
        );
    }

    DirectedGraph graph;
    graph.node_count = header.node_count;
    graph.out.resize(graph.node_count);
    for (std::uint64_t edge_index = 0; edge_index < header.edge_count; ++edge_index) {
        GraphEdgeRecord edge{};
        input.read(reinterpret_cast<char*>(&edge), sizeof(edge));
        if (!input) {
            throw std::runtime_error("读取规范化图边记录失败。 ");
        }
        if (edge.source >= graph.node_count || edge.target >= graph.node_count) {
            throw std::runtime_error("规范化图包含越界节点编号。 ");
        }
        if (!std::isfinite(edge.weight)) {
            throw std::runtime_error("规范化图包含非有限边权。 ");
        }
        if (edge.weight < 0.0) {
            throw std::runtime_error("规范化图包含负权边。 ");
        }
        // 自环不参与消元；查询相同节点始终直接返回 0。
        if (edge.source == edge.target) {
            continue;
        }
        auto& row = graph.out[edge.source];
        const auto existing = row.find(edge.target);
        if (existing == row.end() || edge.weight < existing->second) {
            row[edge.target] = edge.weight;
        }
    }

    for (const auto& row : graph.out) {
        graph.edge_count += row.size();
        for (const auto& [target, weight] : row) {
            (void)target;
            if (weight == 0.0) {
                ++graph.zero_weight_edges;
            }
        }
    }
    validate_strong_connectivity(graph);
    return graph;
}

}  // namespace h2h
