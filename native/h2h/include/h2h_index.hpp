#pragma once

// H2H 消元、分解树、标签、序列化与只读 mmap 查询接口。

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "h2h_format.hpp"
#include "h2h_graph.hpp"

namespace h2h {

struct BuildLimits {
    // 0 表示不限制；非零值用于服务器按实际资源设置保护阈值。
    std::uint32_t max_nodes = 0;
    std::uint64_t max_structural_edges = 0;
    std::uint64_t max_shortcut_arcs = 0;
    std::uint32_t progress_interval = 1000;
};

struct BuildStats {
    std::uint64_t structural_fill_edges = 0;
    std::uint64_t shortcut_arcs = 0;
    std::uint32_t treewidth = 0;
    std::uint32_t treeheight = 0;
    std::uint64_t label_count = 0;
    std::uint64_t position_count = 0;
    double elapsed_seconds = 0.0;
};

struct H2HIndexData {
    std::uint32_t node_count = 0;
    std::uint32_t level_count = 0;
    std::uint32_t treeheight = 0;
    std::uint32_t treewidth = 0;
    std::vector<std::uint32_t> parent;
    std::vector<std::uint32_t> depth;
    // level-major：up[level * node_count + node]。
    std::vector<std::uint32_t> up;
    std::vector<std::uint64_t> label_offsets;
    std::vector<double> dis_out;
    std::vector<double> dis_in;
    std::vector<std::uint64_t> pos_offsets;
    std::vector<std::uint32_t> positions;
};

// 按动态最小结构度构建精确有向 H2H 标签。
H2HIndexData build_h2h_index(
    const DirectedGraph& graph,
    const BuildLimits& limits,
    BuildStats* stats
);

// 以固定小端格式写出 index.bin；失败时抛出包含路径的异常。
void write_h2h_index(
    const H2HIndexData& index,
    const std::filesystem::path& path,
    const std::array<std::uint8_t, 32>& metadata_hash = {}
);

// 对构建期内存索引执行查询，用于 builder 自检及 Python 参考逐项对照。
double query_index_data(const H2HIndexData& index, std::uint32_t source, std::uint32_t target);

class MappedH2HIndex {
public:
    MappedH2HIndex() = default;
    ~MappedH2HIndex();
    MappedH2HIndex(const MappedH2HIndex&) = delete;
    MappedH2HIndex& operator=(const MappedH2HIndex&) = delete;

    // 只读内存映射并完整校验索引头、数组范围和节点编号。
    void open(const std::filesystem::path& path);
    // 释放映射；重复调用安全。
    void close() noexcept;
    // 查询单个有向节点对；越界或损坏时抛出异常。
    double query(std::uint32_t source, std::uint32_t target) const;
    // 批量查询等长数组，输出由调用方分配。
    void query_batch(
        const std::uint32_t* sources,
        const std::uint32_t* targets,
        std::size_t count,
        double* output
    ) const;
    std::uint32_t node_count() const noexcept;

private:
    std::uint32_t lca(std::uint32_t first, std::uint32_t second) const;
    void validate_mapping();

    const std::uint8_t* data_ = nullptr;
    std::size_t mapped_size_ = 0;
    const IndexHeader* header_ = nullptr;
    const std::uint32_t* parent_ = nullptr;
    const std::uint32_t* depth_ = nullptr;
    const std::uint32_t* up_ = nullptr;
    const std::uint64_t* label_offsets_ = nullptr;
    const double* dis_out_ = nullptr;
    const double* dis_in_ = nullptr;
    const std::uint64_t* pos_offsets_ = nullptr;
    const std::uint32_t* positions_ = nullptr;
#ifdef _WIN32
    void* file_handle_ = nullptr;
    void* mapping_handle_ = nullptr;
#else
    int file_descriptor_ = -1;
#endif
};

}  // namespace h2h
