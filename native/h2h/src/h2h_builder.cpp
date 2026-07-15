#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <limits>
#include <queue>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#if defined(__linux__)
#include <sys/resource.h>
#endif

#include "h2h_format.hpp"
#include "h2h_graph.hpp"
#include "h2h_index.hpp"

namespace {

struct CommandLine {
    std::filesystem::path graph_path;
    std::filesystem::path output_path;
    h2h::BuildLimits limits;
    // Python 缓存层传入的 SHA-256；零值仅用于独立 smoke/格式测试。
    std::array<std::uint8_t, 32> metadata_hash{};
};

// 将十进制命令行参数转换为无符号整数并拒绝尾随字符。
std::uint64_t parse_unsigned(const std::string& text, const std::string& option) {
    std::size_t parsed = 0;
    const auto value = std::stoull(text, &parsed);
    if (parsed != text.size()) {
        throw std::runtime_error(option + " 必须是非负整数。 ");
    }
    return value;
}

// 将 64 个十六进制字符转换为 index.bin 头部的 32 字节 metadata 哈希。
std::array<std::uint8_t, 32> parse_hash(const std::string& text) {
    if (text.size() != 64) {
        throw std::runtime_error("--metadata-hash 必须恰好包含 64 个十六进制字符。 ");
    }
    std::array<std::uint8_t, 32> result{};
    for (std::size_t index = 0; index < result.size(); ++index) {
        const auto byte_text = text.substr(index * 2U, 2U);
        std::size_t parsed = 0;
        const auto value = std::stoul(byte_text, &parsed, 16);
        if (parsed != 2U || value > 255U) {
            throw std::runtime_error("--metadata-hash 包含非法十六进制字符。 ");
        }
        result[index] = static_cast<std::uint8_t>(value);
    }
    return result;
}

// 解析 builder 参数；所有路径参数均由 shell-free Python 列表传入。
CommandLine parse_arguments(int argc, char** argv) {
    CommandLine command;
    for (int index = 1; index < argc; ++index) {
        const std::string option = argv[index];
        const auto require_value = [&]() -> std::string {
            if (index + 1 >= argc) {
                throw std::runtime_error(option + " 缺少参数值。 ");
            }
            return argv[++index];
        };
        if (option == "--graph") {
            command.graph_path = std::filesystem::u8path(require_value());
        } else if (option == "--output") {
            command.output_path = std::filesystem::u8path(require_value());
        } else if (option == "--max-nodes") {
            const auto value = parse_unsigned(require_value(), option);
            if (value > std::numeric_limits<std::uint32_t>::max()) {
                throw std::runtime_error(option + " 超出 uint32 范围。 ");
            }
            command.limits.max_nodes = static_cast<std::uint32_t>(value);
        } else if (option == "--max-structural-edges") {
            command.limits.max_structural_edges = parse_unsigned(require_value(), option);
        } else if (option == "--max-shortcut-arcs") {
            command.limits.max_shortcut_arcs = parse_unsigned(require_value(), option);
        } else if (option == "--progress-interval") {
            const auto value = parse_unsigned(require_value(), option);
            if (value > std::numeric_limits<std::uint32_t>::max()) {
                throw std::runtime_error(option + " 超出 uint32 范围。 ");
            }
            command.limits.progress_interval = static_cast<std::uint32_t>(value);
        } else if (option == "--metadata-hash") {
            command.metadata_hash = parse_hash(require_value());
        } else if (option == "--version") {
            std::cout << "h2h_builder api=" << h2h::kApiVersion
                      << " index_format=" << h2h::kIndexFormatVersion << '\n';
            std::exit(0);
        } else if (option == "--help") {
            std::cout
                << "Usage: h2h_builder --graph graph.bin --output index.bin [options]\n"
                << "  --max-nodes N              节点数资源上限，0 表示不限制\n"
                << "  --max-structural-edges N   当前结构边上限，0 表示不限制\n"
                << "  --max-shortcut-arcs N      累计新增 shortcut 上限，0 表示不限制\n"
                << "  --progress-interval N      每 N 个消元点输出进度，0 表示关闭\n"
                << "  --metadata-hash HEX        写入索引头部的 64 位十六进制 SHA-256\n";
            std::exit(0);
        } else {
            throw std::runtime_error("未知参数：" + option);
        }
    }
    if (command.graph_path.empty() || command.output_path.empty()) {
        throw std::runtime_error("必须同时提供 --graph 和 --output。 ");
    }
    return command;
}

// 对小图执行单源 Dijkstra，仅用于 builder 写盘前内部正确性自检。
std::vector<double> dijkstra(const h2h::DirectedGraph& graph, std::uint32_t source) {
    std::vector<double> distance(graph.node_count, std::numeric_limits<double>::infinity());
    using QueueEntry = std::pair<double, std::uint32_t>;
    std::priority_queue<QueueEntry, std::vector<QueueEntry>, std::greater<QueueEntry>> queue;
    distance[source] = 0.0;
    queue.emplace(0.0, source);
    while (!queue.empty()) {
        const auto [current_distance, node] = queue.top();
        queue.pop();
        if (current_distance != distance[node]) {
            continue;
        }
        for (const auto& [target, weight] : graph.out[node]) {
            const auto candidate = current_distance + weight;
            if (candidate < distance[target]) {
                distance[target] = candidate;
                queue.emplace(candidate, target);
            }
        }
    }
    return distance;
}

// 节点数不超过 200 时执行全对自检，确保损坏索引不会写入文件。
void validate_small_index(const h2h::DirectedGraph& graph, const h2h::H2HIndexData& index) {
    if (graph.node_count > 200) {
        return;
    }
    for (std::uint32_t source = 0; source < graph.node_count; ++source) {
        const auto expected = dijkstra(graph, source);
        for (std::uint32_t target = 0; target < graph.node_count; ++target) {
            const auto actual = h2h::query_index_data(index, source, target);
            const auto absolute_error = std::abs(actual - expected[target]);
            const auto scale = std::max({1.0, std::abs(actual), std::abs(expected[target])});
            if (absolute_error > 1e-10 && absolute_error / scale > 1e-10) {
                throw std::runtime_error(
                    "builder 小图自检失败，节点对 " + std::to_string(source) + " -> " +
                    std::to_string(target) + " 误差为 " + std::to_string(absolute_error) + "。"
                );
            }
        }
    }
}

// 返回当前 builder 的峰值常驻内存；Linux 的 ru_maxrss 单位为 KiB。
// Windows 本机仅做中小图验收，返回 0；55k 服务器验收必须在 Linux 记录该值。
std::uint64_t peak_rss_bytes() noexcept {
#if defined(__linux__)
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) == 0 && usage.ru_maxrss > 0) {
        return static_cast<std::uint64_t>(usage.ru_maxrss) * 1024U;
    }
#endif
    return 0U;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto command = parse_arguments(argc, argv);
        const auto graph = h2h::read_graph_binary(command.graph_path);
        std::cerr << "[h2h] graph_nodes=" << graph.node_count
                  << " graph_arcs=" << graph.edge_count
                  << " zero_weight_arcs=" << graph.zero_weight_edges << '\n';
        h2h::BuildStats stats;
        const auto index = h2h::build_h2h_index(graph, command.limits, &stats);
        validate_small_index(graph, index);
        h2h::write_h2h_index(index, command.output_path, command.metadata_hash);
        std::cout << "H2H_BUILD_OK"
                  << " nodes=" << graph.node_count
                  << " treewidth=" << stats.treewidth
                  << " treeheight=" << stats.treeheight
                  << " fill_edges=" << stats.structural_fill_edges
                  << " shortcut_arcs=" << stats.shortcut_arcs
                  << " labels=" << stats.label_count
                  << " positions=" << stats.position_count
                  << " seconds=" << stats.elapsed_seconds
                  << " peak_rss_bytes=" << peak_rss_bytes()
                  << " index_bytes=" << std::filesystem::file_size(command.output_path)
                  << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "H2H_BUILD_ERROR: " << error.what() << '\n';
        return 1;
    }
}
