#pragma once

// H2H 图文件与索引文件的固定宽度、小端序二进制格式。

#include <array>
#include <cstddef>
#include <cstdint>

namespace h2h {

inline constexpr std::array<char, 8> kGraphMagic{{'H', '2', 'H', 'G', 'R', 'P', 'H', '1'}};
inline constexpr std::array<char, 8> kIndexMagic{{'H', '2', 'H', 'I', 'D', 'X', '0', '1'}};
inline constexpr std::uint32_t kGraphFormatVersion = 1;
inline constexpr std::uint32_t kIndexFormatVersion = 1;
inline constexpr std::uint32_t kApiVersion = 1;
inline constexpr std::uint32_t kEndianMarker = 0x01020304U;

#pragma pack(push, 1)

// 规范化图头；后面紧跟 edge_count 条 GraphEdgeRecord。
struct GraphHeader {
    char magic[8];
    std::uint32_t format_version;
    std::uint32_t endian_marker;
    std::uint32_t node_count;
    std::uint64_t edge_count;
};

// 规范化有向简单边记录，权重为 IEEE 754 float64。
struct GraphEdgeRecord {
    std::uint32_t source;
    std::uint32_t target;
    double weight;
};

// 索引头中的 offset 均为相对 index.bin 起始位置的字节偏移。
struct IndexHeader {
    char magic[8];
    std::uint32_t format_version;
    std::uint32_t endian_marker;
    std::uint32_t api_version;
    std::uint32_t header_size;
    std::uint32_t node_count;
    std::uint32_t level_count;
    std::uint32_t treeheight;
    std::uint32_t treewidth;
    std::uint64_t label_count;
    std::uint64_t position_count;
    std::uint64_t parent_offset;
    std::uint64_t depth_offset;
    std::uint64_t up_offset;
    std::uint64_t label_offsets_offset;
    std::uint64_t dis_out_offset;
    std::uint64_t dis_in_offset;
    std::uint64_t pos_offsets_offset;
    std::uint64_t positions_offset;
    std::uint64_t file_size;
    // 阶段 4 写入图/metadata 哈希；阶段 3 保持全零并校验格式本身。
    std::uint8_t metadata_hash[32];
};

#pragma pack(pop)

static_assert(sizeof(GraphHeader) == 28, "GraphHeader 二进制布局发生变化");
static_assert(sizeof(GraphEdgeRecord) == 16, "GraphEdgeRecord 二进制布局发生变化");
static_assert(sizeof(IndexHeader) == 160, "IndexHeader 二进制布局发生变化");

}  // namespace h2h
