#pragma once

// 规范化有向图的读取、校验和最小平行边表示。

#include <cstdint>
#include <filesystem>
#include <unordered_map>
#include <vector>

namespace h2h {

struct DirectedGraph {
    // 节点固定连续编号为 [0, node_count)。
    std::uint32_t node_count = 0;
    // out[source][target] 保存规范化后的最小有限非负边权。
    std::vector<std::unordered_map<std::uint32_t, double>> out;
    std::uint64_t edge_count = 0;
    std::uint64_t zero_weight_edges = 0;
};

// 读取 graph.bin，拒绝非法编号、负权、非有限权和非强连通图。
DirectedGraph read_graph_binary(const std::filesystem::path& path);

}  // namespace h2h
