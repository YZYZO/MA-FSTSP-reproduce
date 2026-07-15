#include "h2h_index.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <queue>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace h2h {
namespace {

constexpr double kInfinity = std::numeric_limits<double>::infinity();

// 将 offset 向上对齐，使 mmap 后的固定宽度数组满足自然对齐要求。
std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment) {
    return (value + alignment - 1U) / alignment * alignment;
}

// 获取有限弧；不存在时返回正无穷。
double edge_or_infinity(
    const std::vector<std::unordered_map<std::uint32_t, double>>& out,
    std::uint32_t source,
    std::uint32_t target
) {
    const auto iterator = out[source].find(target);
    return iterator == out[source].end() ? kInfinity : iterator->second;
}

// 对构建期内存数组执行二进制提升 LCA。
std::uint32_t lca_index_data(
    const H2HIndexData& index,
    std::uint32_t first,
    std::uint32_t second
) {
    if (index.depth[first] < index.depth[second]) {
        std::swap(first, second);
    }
    auto difference = index.depth[first] - index.depth[second];
    for (std::uint32_t level = 0; level < index.level_count; ++level) {
        if ((difference & (1U << level)) != 0U) {
            first = index.up[static_cast<std::size_t>(level) * index.node_count + first];
        }
    }
    if (first == second) {
        return first;
    }
    for (std::uint32_t level = index.level_count; level-- > 0;) {
        const auto first_up = index.up[static_cast<std::size_t>(level) * index.node_count + first];
        const auto second_up = index.up[static_cast<std::size_t>(level) * index.node_count + second];
        if (first_up != second_up) {
            first = first_up;
            second = second_up;
        }
    }
    return index.parent[first];
}

// 在指定文件位置写一个连续向量；空向量不会解引用 data()。
template <typename T>
void write_vector_at(
    std::ofstream& output,
    std::uint64_t offset,
    const std::vector<T>& values,
    const char* section_name
) {
    output.seekp(static_cast<std::streamoff>(offset));
    if (!values.empty()) {
        output.write(
            reinterpret_cast<const char*>(values.data()),
            static_cast<std::streamsize>(values.size() * sizeof(T))
        );
    }
    if (!output) {
        throw std::runtime_error(std::string("写入索引段失败：") + section_name);
    }
}

}  // namespace

H2HIndexData build_h2h_index(
    const DirectedGraph& graph,
    const BuildLimits& limits,
    BuildStats* stats
) {
    const auto started_at = std::chrono::steady_clock::now();
    if (graph.node_count == 0 || graph.out.size() != graph.node_count) {
        throw std::runtime_error("构建 H2H 时收到空图或邻接表规模不一致。 ");
    }
    if (limits.max_nodes != 0 && graph.node_count > limits.max_nodes) {
        throw std::runtime_error(
            "图节点数 " + std::to_string(graph.node_count) + " 超过 --max-nodes " +
            std::to_string(limits.max_nodes) + "。"
        );
    }

    const auto node_count = graph.node_count;
    auto out = graph.out;
    // structural 只描述 tree decomposition 的无向邻接；不代表存在有限有向路径。
    std::vector<std::unordered_set<std::uint32_t>> structural(node_count);
    for (std::uint32_t source = 0; source < node_count; ++source) {
        for (const auto& [target, weight] : out[source]) {
            (void)weight;
            structural[source].insert(target);
            structural[target].insert(source);
        }
    }
    std::uint64_t structural_edge_count = 0;
    for (const auto& row : structural) {
        structural_edge_count += row.size();
    }
    structural_edge_count /= 2U;

    using HeapEntry = std::pair<std::size_t, std::uint32_t>;
    std::priority_queue<HeapEntry, std::vector<HeapEntry>, std::greater<HeapEntry>> heap;
    for (std::uint32_t node = 0; node < node_count; ++node) {
        heap.emplace(structural[node].size(), node);
    }
    std::vector<std::uint8_t> active(node_count, 1);
    std::vector<std::uint32_t> rank(node_count, 0);
    std::vector<std::vector<std::uint32_t>> bags(node_count);
    std::vector<std::vector<double>> star_out(node_count);
    std::vector<std::vector<double>> star_in(node_count);
    std::uint64_t structural_fill_edges = 0;
    std::uint64_t shortcut_arcs = 0;
    std::uint32_t treewidth = 0;

    for (std::uint32_t order = 0; order < node_count; ++order) {
        std::uint32_t node = 0;
        bool found = false;
        while (!heap.empty()) {
            const auto [degree, candidate] = heap.top();
            heap.pop();
            if (active[candidate] && degree == structural[candidate].size()) {
                node = candidate;
                found = true;
                break;
            }
        }
        if (!found) {
            throw std::runtime_error("动态最小度优先队列意外耗尽。 ");
        }

        auto neighbors = std::vector<std::uint32_t>(
            structural[node].begin(), structural[node].end()
        );
        std::sort(neighbors.begin(), neighbors.end());
        rank[node] = order;
        bags[node] = neighbors;
        treewidth = std::max(treewidth, static_cast<std::uint32_t>(neighbors.size()));
        star_out[node].reserve(neighbors.size());
        star_in[node].reserve(neighbors.size());
        for (const auto neighbor : neighbors) {
            star_out[node].push_back(edge_or_infinity(out, node, neighbor));
            star_in[node].push_back(edge_or_infinity(out, neighbor, node));
        }

        // 对结构 bag 补 clique；这一步不创造有限距离。
        for (std::size_t first = 0; first < neighbors.size(); ++first) {
            for (std::size_t second = first + 1; second < neighbors.size(); ++second) {
                const auto source = neighbors[first];
                const auto target = neighbors[second];
                if (structural[source].insert(target).second) {
                    structural[target].insert(source);
                    ++structural_fill_edges;
                    ++structural_edge_count;
                }
            }
        }
        if (limits.max_structural_edges != 0 &&
                structural_edge_count > limits.max_structural_edges) {
            throw std::runtime_error(
                "结构 fill-in 超过 --max-structural-edges=" +
                std::to_string(limits.max_structural_edges) + "。"
            );
        }

        // 只有真实有限的前驱/后继组合才新增或缩短有向 shortcut。
        for (const auto source : neighbors) {
            const auto incoming_iterator = out[source].find(node);
            if (incoming_iterator == out[source].end()) {
                continue;
            }
            for (const auto target : neighbors) {
                if (source == target) {
                    continue;
                }
                const auto outgoing_iterator = out[node].find(target);
                if (outgoing_iterator == out[node].end()) {
                    continue;
                }
                const auto candidate = incoming_iterator->second + outgoing_iterator->second;
                const auto existing = out[source].find(target);
                if (existing == out[source].end()) {
                    out[source][target] = candidate;
                    ++shortcut_arcs;
                } else if (candidate < existing->second) {
                    existing->second = candidate;
                }
            }
        }
        if (limits.max_shortcut_arcs != 0 && shortcut_arcs > limits.max_shortcut_arcs) {
            throw std::runtime_error(
                "新增 shortcut 数超过 --max-shortcut-arcs=" +
                std::to_string(limits.max_shortcut_arcs) + "。"
            );
        }

        active[node] = 0;
        for (const auto neighbor : neighbors) {
            structural[neighbor].erase(node);
            out[neighbor].erase(node);
        }
        structural_edge_count -= neighbors.size();
        structural[node].clear();
        out[node].clear();
        for (const auto neighbor : neighbors) {
            heap.emplace(structural[neighbor].size(), neighbor);
        }

        if (limits.progress_interval != 0 &&
                ((order + 1U) % limits.progress_interval == 0U || order + 1U == node_count)) {
            std::cerr << "[h2h] eliminated=" << (order + 1U) << '/' << node_count
                      << " active_structural_edges=" << structural_edge_count
                      << " fill_edges=" << structural_fill_edges
                      << " shortcut_arcs=" << shortcut_arcs
                      << " max_bag_neighbors=" << treewidth << '\n';
        }
    }

    // 父节点是 bag 中最早在之后被消元的节点；最后消元节点是唯一根。
    std::vector<std::uint32_t> parent(node_count, std::numeric_limits<std::uint32_t>::max());
    std::uint32_t root = std::numeric_limits<std::uint32_t>::max();
    std::uint32_t root_count = 0;
    for (std::uint32_t node = 0; node < node_count; ++node) {
        if (bags[node].empty()) {
            root = node;
            parent[node] = node;
            ++root_count;
            continue;
        }
        auto selected = bags[node].front();
        for (const auto candidate : bags[node]) {
            if (rank[candidate] <= rank[node]) {
                throw std::runtime_error("bag 中出现 rank 不高于当前节点的成员。 ");
            }
            if (rank[candidate] < rank[selected]) {
                selected = candidate;
            }
        }
        parent[node] = selected;
    }
    if (root_count != 1) {
        throw std::runtime_error("分解树根节点数量不是 1。 ");
    }

    std::vector<std::vector<std::uint32_t>> children(node_count);
    for (std::uint32_t node = 0; node < node_count; ++node) {
        if (node != root) {
            if (parent[node] >= node_count || rank[parent[node]] <= rank[node]) {
                throw std::runtime_error("分解树父节点非法或 rank 未提高。 ");
            }
            children[parent[node]].push_back(node);
        }
    }
    std::vector<std::uint32_t> depth(node_count, std::numeric_limits<std::uint32_t>::max());
    depth[root] = 0;
    std::vector<std::uint32_t> stack{root};
    while (!stack.empty()) {
        const auto node = stack.back();
        stack.pop_back();
        for (const auto child : children[node]) {
            if (depth[child] != std::numeric_limits<std::uint32_t>::max()) {
                throw std::runtime_error("分解树检测到环。 ");
            }
            depth[child] = depth[node] + 1U;
            stack.push_back(child);
        }
    }
    if (std::any_of(depth.begin(), depth.end(), [](const auto value) {
            return value == std::numeric_limits<std::uint32_t>::max();
        })) {
        throw std::runtime_error("分解树没有覆盖全部节点。 ");
    }

    std::vector<std::uint32_t> node_at_rank(node_count);
    for (std::uint32_t node = 0; node < node_count; ++node) {
        node_at_rank[rank[node]] = node;
    }
    std::vector<std::vector<std::uint32_t>> ancestors(node_count);
    std::vector<std::vector<double>> dis_out_rows(node_count);
    std::vector<std::vector<double>> dis_in_rows(node_count);
    std::vector<std::vector<std::uint32_t>> position_rows(node_count);

    // 按 rank 逆序即根到叶，复用祖先已构造标签完成 partial-label DP。
    for (std::uint32_t reverse_order = node_count; reverse_order-- > 0;) {
        const auto node = node_at_rank[reverse_order];
        if (node == root) {
            ancestors[node] = {node};
        } else {
            ancestors[node] = ancestors[parent[node]];
            ancestors[node].push_back(node);
        }
        if (ancestors[node].size() != static_cast<std::size_t>(depth[node]) + 1U) {
            throw std::runtime_error("祖先链长度与深度不一致。 ");
        }

        auto positions = std::vector<std::uint32_t>{depth[node]};
        for (const auto bag_node : bags[node]) {
            if (depth[bag_node] >= ancestors[node].size() ||
                    ancestors[node][depth[bag_node]] != bag_node) {
                throw std::runtime_error("bag 节点不能映射到当前祖先链。 ");
            }
            positions.push_back(depth[bag_node]);
        }
        std::sort(positions.begin(), positions.end());
        positions.erase(std::unique(positions.begin(), positions.end()), positions.end());
        position_rows[node] = std::move(positions);

        auto out_label = std::vector<double>(ancestors[node].size(), kInfinity);
        auto in_label = std::vector<double>(ancestors[node].size(), kInfinity);
        out_label[depth[node]] = 0.0;
        in_label[depth[node]] = 0.0;
        for (std::uint32_t target_position = 0; target_position < depth[node]; ++target_position) {
            const auto target = ancestors[node][target_position];
            for (std::size_t boundary_index = 0;
                    boundary_index < bags[node].size(); ++boundary_index) {
                const auto boundary = bags[node][boundary_index];
                const auto boundary_depth = depth[boundary];
                double boundary_to_target = 0.0;
                double target_to_boundary = 0.0;
                if (boundary_depth > target_position) {
                    boundary_to_target = dis_out_rows[boundary][target_position];
                    target_to_boundary = dis_in_rows[boundary][target_position];
                } else if (boundary_depth < target_position) {
                    boundary_to_target = dis_in_rows[target][boundary_depth];
                    target_to_boundary = dis_out_rows[target][boundary_depth];
                }
                out_label[target_position] = std::min(
                    out_label[target_position],
                    star_out[node][boundary_index] + boundary_to_target
                );
                in_label[target_position] = std::min(
                    in_label[target_position],
                    target_to_boundary + star_in[node][boundary_index]
                );
            }
        }
        if (std::any_of(out_label.begin(), out_label.end(), [](double value) {
                return !std::isfinite(value);
            }) || std::any_of(in_label.begin(), in_label.end(), [](double value) {
                return !std::isfinite(value);
            })) {
            throw std::runtime_error("强连通图的祖先标签出现无穷距离。 ");
        }
        dis_out_rows[node] = std::move(out_label);
        dis_in_rows[node] = std::move(in_label);
    }

    H2HIndexData index;
    index.node_count = node_count;
    index.parent = std::move(parent);
    index.depth = std::move(depth);
    index.treewidth = treewidth;
    index.treeheight = *std::max_element(index.depth.begin(), index.depth.end()) + 1U;
    const auto max_depth = index.treeheight - 1U;
    std::uint32_t depth_bits = 0;
    for (auto value = max_depth; value != 0; value >>= 1U) {
        ++depth_bits;
    }
    index.level_count = std::max(1U, depth_bits + 1U);
    index.up.resize(static_cast<std::size_t>(index.level_count) * node_count);
    for (std::uint32_t node = 0; node < node_count; ++node) {
        index.up[node] = index.parent[node];
    }
    for (std::uint32_t level = 1; level < index.level_count; ++level) {
        const auto previous_offset = static_cast<std::size_t>(level - 1U) * node_count;
        const auto current_offset = static_cast<std::size_t>(level) * node_count;
        for (std::uint32_t node = 0; node < node_count; ++node) {
            const auto middle = index.up[previous_offset + node];
            index.up[current_offset + node] = index.up[previous_offset + middle];
        }
    }

    index.label_offsets.reserve(static_cast<std::size_t>(node_count) + 1U);
    index.pos_offsets.reserve(static_cast<std::size_t>(node_count) + 1U);
    index.label_offsets.push_back(0);
    index.pos_offsets.push_back(0);
    for (std::uint32_t node = 0; node < node_count; ++node) {
        index.dis_out.insert(index.dis_out.end(), dis_out_rows[node].begin(), dis_out_rows[node].end());
        index.dis_in.insert(index.dis_in.end(), dis_in_rows[node].begin(), dis_in_rows[node].end());
        index.positions.insert(
            index.positions.end(), position_rows[node].begin(), position_rows[node].end()
        );
        index.label_offsets.push_back(index.dis_out.size());
        index.pos_offsets.push_back(index.positions.size());
    }

    if (stats != nullptr) {
        stats->structural_fill_edges = structural_fill_edges;
        stats->shortcut_arcs = shortcut_arcs;
        stats->treewidth = index.treewidth;
        stats->treeheight = index.treeheight;
        stats->label_count = index.dis_out.size();
        stats->position_count = index.positions.size();
        stats->elapsed_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started_at
        ).count();
    }
    return index;
}

double query_index_data(
    const H2HIndexData& index,
    std::uint32_t source,
    std::uint32_t target
) {
    if (source >= index.node_count || target >= index.node_count) {
        throw std::out_of_range("H2H 查询节点编号越界。 ");
    }
    if (source == target) {
        return 0.0;
    }
    const auto ancestor = lca_index_data(index, source, target);
    auto result = kInfinity;
    for (auto offset = index.pos_offsets[ancestor]; offset < index.pos_offsets[ancestor + 1U]; ++offset) {
        const auto position = index.positions[offset];
        const auto source_label = index.label_offsets[source] + position;
        const auto target_label = index.label_offsets[target] + position;
        result = std::min(result, index.dis_out[source_label] + index.dis_in[target_label]);
    }
    if (!std::isfinite(result)) {
        throw std::runtime_error("H2H 内存索引查询返回无穷距离。 ");
    }
    return result;
}

void write_h2h_index(
    const H2HIndexData& index,
    const std::filesystem::path& path,
    const std::array<std::uint8_t, 32>& metadata_hash
) {
    if (index.node_count == 0 || index.parent.size() != index.node_count ||
            index.depth.size() != index.node_count ||
            index.up.size() != static_cast<std::size_t>(index.level_count) * index.node_count ||
            index.label_offsets.size() != static_cast<std::size_t>(index.node_count) + 1U ||
            index.pos_offsets.size() != static_cast<std::size_t>(index.node_count) + 1U ||
            index.dis_out.size() != index.dis_in.size()) {
        throw std::runtime_error("拒绝写入内部数组规模不一致的 H2H 索引。 ");
    }

    IndexHeader header{};
    std::memcpy(header.magic, kIndexMagic.data(), kIndexMagic.size());
    header.format_version = kIndexFormatVersion;
    header.endian_marker = kEndianMarker;
    header.api_version = kApiVersion;
    header.header_size = sizeof(IndexHeader);
    header.node_count = index.node_count;
    header.level_count = index.level_count;
    header.treeheight = index.treeheight;
    header.treewidth = index.treewidth;
    header.label_count = index.dis_out.size();
    header.position_count = index.positions.size();
    std::copy(metadata_hash.begin(), metadata_hash.end(), header.metadata_hash);

    auto cursor = static_cast<std::uint64_t>(sizeof(IndexHeader));
    header.parent_offset = align_up(cursor, alignof(std::uint32_t));
    cursor = header.parent_offset + index.parent.size() * sizeof(std::uint32_t);
    header.depth_offset = align_up(cursor, alignof(std::uint32_t));
    cursor = header.depth_offset + index.depth.size() * sizeof(std::uint32_t);
    header.up_offset = align_up(cursor, alignof(std::uint32_t));
    cursor = header.up_offset + index.up.size() * sizeof(std::uint32_t);
    header.label_offsets_offset = align_up(cursor, alignof(std::uint64_t));
    cursor = header.label_offsets_offset + index.label_offsets.size() * sizeof(std::uint64_t);
    header.dis_out_offset = align_up(cursor, alignof(double));
    cursor = header.dis_out_offset + index.dis_out.size() * sizeof(double);
    header.dis_in_offset = align_up(cursor, alignof(double));
    cursor = header.dis_in_offset + index.dis_in.size() * sizeof(double);
    header.pos_offsets_offset = align_up(cursor, alignof(std::uint64_t));
    cursor = header.pos_offsets_offset + index.pos_offsets.size() * sizeof(std::uint64_t);
    header.positions_offset = align_up(cursor, alignof(std::uint32_t));
    cursor = header.positions_offset + index.positions.size() * sizeof(std::uint32_t);
    header.file_size = cursor;

    const auto parent_path = path.parent_path();
    if (!parent_path.empty()) {
        std::filesystem::create_directories(parent_path);
    }
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error("无法创建 H2H 索引文件：" + path.string());
    }
    output.write(reinterpret_cast<const char*>(&header), sizeof(header));
    write_vector_at(output, header.parent_offset, index.parent, "parent");
    write_vector_at(output, header.depth_offset, index.depth, "depth");
    write_vector_at(output, header.up_offset, index.up, "up");
    write_vector_at(output, header.label_offsets_offset, index.label_offsets, "label_offsets");
    write_vector_at(output, header.dis_out_offset, index.dis_out, "dis_out");
    write_vector_at(output, header.dis_in_offset, index.dis_in, "dis_in");
    write_vector_at(output, header.pos_offsets_offset, index.pos_offsets, "pos_offsets");
    write_vector_at(output, header.positions_offset, index.positions, "positions");
    output.flush();
    if (!output) {
        throw std::runtime_error("刷新 H2H 索引文件失败：" + path.string());
    }
    output.close();
    if (std::filesystem::file_size(path) != header.file_size) {
        throw std::runtime_error("写出的 H2H 索引文件大小与头部不一致。 ");
    }
}

}  // namespace h2h
