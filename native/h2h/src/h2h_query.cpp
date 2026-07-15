#include "h2h_index.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace h2h {
namespace {

// 校验一个固定宽度数组段完全位于 mmap 文件内部且乘法不溢出。
bool valid_segment(
    std::uint64_t offset,
    std::uint64_t count,
    std::uint64_t element_size,
    std::uint64_t file_size
) {
    if (offset > file_size || element_size == 0) {
        return false;
    }
    if (count > (std::numeric_limits<std::uint64_t>::max() - offset) / element_size) {
        return false;
    }
    return offset + count * element_size <= file_size;
}

#ifdef _WIN32
// C ABI 路径约定为 UTF-8；Windows mmap 前转换为 UTF-16，避免中文路径失败。
std::wstring utf8_to_wide(const std::string& value) {
    if (value.empty()) {
        return {};
    }
    const auto length = MultiByteToWideChar(
        CP_UTF8, MB_ERR_INVALID_CHARS, value.data(), static_cast<int>(value.size()), nullptr, 0
    );
    if (length <= 0) {
        throw std::runtime_error("索引路径不是有效 UTF-8。 ");
    }
    std::wstring result(static_cast<std::size_t>(length), L'\0');
    MultiByteToWideChar(
        CP_UTF8, MB_ERR_INVALID_CHARS, value.data(), static_cast<int>(value.size()),
        result.data(), length
    );
    return result;
}
#endif

}  // namespace

MappedH2HIndex::~MappedH2HIndex() {
    close();
}

void MappedH2HIndex::open(const std::filesystem::path& path) {
    close();
    try {
#ifdef _WIN32
        const auto wide_path = utf8_to_wide(path.u8string());
        const auto file_handle = CreateFileW(
            wide_path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
            nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr
        );
        if (file_handle == INVALID_HANDLE_VALUE) {
            throw std::runtime_error("无法打开 H2H 索引文件：" + path.string());
        }
        file_handle_ = file_handle;
        LARGE_INTEGER size{};
        if (!GetFileSizeEx(file_handle, &size) || size.QuadPart <= 0) {
            throw std::runtime_error("无法读取 H2H 索引文件大小。 ");
        }
        mapped_size_ = static_cast<std::size_t>(size.QuadPart);
        const auto mapping_handle = CreateFileMappingW(
            file_handle, nullptr, PAGE_READONLY, 0, 0, nullptr
        );
        if (mapping_handle == nullptr) {
            throw std::runtime_error("创建 H2H 只读文件映射失败。 ");
        }
        mapping_handle_ = mapping_handle;
        const auto view = MapViewOfFile(mapping_handle, FILE_MAP_READ, 0, 0, 0);
        if (view == nullptr) {
            throw std::runtime_error("映射 H2H 索引文件失败。 ");
        }
        data_ = static_cast<const std::uint8_t*>(view);
#else
        file_descriptor_ = ::open(path.c_str(), O_RDONLY);
        if (file_descriptor_ < 0) {
            throw std::runtime_error("无法打开 H2H 索引文件：" + path.string());
        }
        struct stat status {};
        if (fstat(file_descriptor_, &status) != 0 || status.st_size <= 0) {
            throw std::runtime_error("无法读取 H2H 索引文件大小。 ");
        }
        mapped_size_ = static_cast<std::size_t>(status.st_size);
        const auto view = mmap(nullptr, mapped_size_, PROT_READ, MAP_SHARED, file_descriptor_, 0);
        if (view == MAP_FAILED) {
            throw std::runtime_error("映射 H2H 索引文件失败。 ");
        }
        data_ = static_cast<const std::uint8_t*>(view);
#endif
        validate_mapping();
    } catch (...) {
        close();
        throw;
    }
}

void MappedH2HIndex::close() noexcept {
#ifdef _WIN32
    if (data_ != nullptr) {
        UnmapViewOfFile(data_);
    }
    if (mapping_handle_ != nullptr) {
        CloseHandle(static_cast<HANDLE>(mapping_handle_));
    }
    if (file_handle_ != nullptr) {
        CloseHandle(static_cast<HANDLE>(file_handle_));
    }
    mapping_handle_ = nullptr;
    file_handle_ = nullptr;
#else
    if (data_ != nullptr && mapped_size_ != 0) {
        munmap(const_cast<std::uint8_t*>(data_), mapped_size_);
    }
    if (file_descriptor_ >= 0) {
        ::close(file_descriptor_);
    }
    file_descriptor_ = -1;
#endif
    data_ = nullptr;
    mapped_size_ = 0;
    header_ = nullptr;
    parent_ = nullptr;
    depth_ = nullptr;
    up_ = nullptr;
    label_offsets_ = nullptr;
    dis_out_ = nullptr;
    dis_in_ = nullptr;
    pos_offsets_ = nullptr;
    positions_ = nullptr;
}

void MappedH2HIndex::validate_mapping() {
    if (data_ == nullptr || mapped_size_ < sizeof(IndexHeader)) {
        throw std::runtime_error("H2H 索引文件短于固定头部。 ");
    }
    header_ = reinterpret_cast<const IndexHeader*>(data_);
    if (std::memcmp(header_->magic, kIndexMagic.data(), kIndexMagic.size()) != 0) {
        throw std::runtime_error("H2H 索引 magic 不匹配。 ");
    }
    if (header_->format_version != kIndexFormatVersion || header_->api_version != kApiVersion) {
        throw std::runtime_error("H2H 索引或 API 版本不受支持。 ");
    }
    if (header_->endian_marker != kEndianMarker || header_->header_size != sizeof(IndexHeader)) {
        throw std::runtime_error("H2H 索引端序或头部大小不匹配。 ");
    }
    if (header_->file_size != mapped_size_ || header_->node_count == 0 ||
            header_->level_count == 0) {
        throw std::runtime_error("H2H 索引文件大小或基础计数非法。 ");
    }
    const auto node_count_value = static_cast<std::uint64_t>(header_->node_count);
    const auto up_count = node_count_value * header_->level_count;
    if (!valid_segment(header_->parent_offset, node_count_value, sizeof(std::uint32_t), mapped_size_) ||
            !valid_segment(header_->depth_offset, node_count_value, sizeof(std::uint32_t), mapped_size_) ||
            !valid_segment(header_->up_offset, up_count, sizeof(std::uint32_t), mapped_size_) ||
            !valid_segment(header_->label_offsets_offset, node_count_value + 1U,
                sizeof(std::uint64_t), mapped_size_) ||
            !valid_segment(header_->dis_out_offset, header_->label_count, sizeof(double), mapped_size_) ||
            !valid_segment(header_->dis_in_offset, header_->label_count, sizeof(double), mapped_size_) ||
            !valid_segment(header_->pos_offsets_offset, node_count_value + 1U,
                sizeof(std::uint64_t), mapped_size_) ||
            !valid_segment(header_->positions_offset, header_->position_count,
                sizeof(std::uint32_t), mapped_size_)) {
        throw std::runtime_error("H2H 索引数组 offset/长度越过文件边界。 ");
    }
    if (header_->label_offsets_offset % alignof(std::uint64_t) != 0 ||
            header_->dis_out_offset % alignof(double) != 0 ||
            header_->dis_in_offset % alignof(double) != 0 ||
            header_->pos_offsets_offset % alignof(std::uint64_t) != 0) {
        throw std::runtime_error("H2H 索引 float64/uint64 数组未自然对齐。 ");
    }

    parent_ = reinterpret_cast<const std::uint32_t*>(data_ + header_->parent_offset);
    depth_ = reinterpret_cast<const std::uint32_t*>(data_ + header_->depth_offset);
    up_ = reinterpret_cast<const std::uint32_t*>(data_ + header_->up_offset);
    label_offsets_ = reinterpret_cast<const std::uint64_t*>(data_ + header_->label_offsets_offset);
    dis_out_ = reinterpret_cast<const double*>(data_ + header_->dis_out_offset);
    dis_in_ = reinterpret_cast<const double*>(data_ + header_->dis_in_offset);
    pos_offsets_ = reinterpret_cast<const std::uint64_t*>(data_ + header_->pos_offsets_offset);
    positions_ = reinterpret_cast<const std::uint32_t*>(data_ + header_->positions_offset);

    if (label_offsets_[0] != 0 || label_offsets_[header_->node_count] != header_->label_count ||
            pos_offsets_[0] != 0 || pos_offsets_[header_->node_count] != header_->position_count) {
        throw std::runtime_error("H2H 索引 offset 端点与总计数不一致。 ");
    }
    std::uint32_t root_count = 0;
    for (std::uint32_t node = 0; node < header_->node_count; ++node) {
        if (parent_[node] >= header_->node_count) {
            throw std::runtime_error("H2H parent 数组含越界节点。 ");
        }
        if (parent_[node] == node) {
            ++root_count;
        }
        if (label_offsets_[node] > label_offsets_[node + 1U] ||
                pos_offsets_[node] > pos_offsets_[node + 1U]) {
            throw std::runtime_error("H2H offset 数组不是单调非降。 ");
        }
        if (label_offsets_[node + 1U] - label_offsets_[node] !=
                static_cast<std::uint64_t>(depth_[node]) + 1U) {
            throw std::runtime_error("H2H 标签行长度与节点深度不一致。 ");
        }
        for (auto offset = pos_offsets_[node]; offset < pos_offsets_[node + 1U]; ++offset) {
            if (positions_[offset] > depth_[node]) {
                throw std::runtime_error("H2H bag position 超过节点深度。 ");
            }
        }
        for (std::uint32_t level = 0; level < header_->level_count; ++level) {
            if (up_[static_cast<std::size_t>(level) * header_->node_count + node] >=
                    header_->node_count) {
                throw std::runtime_error("H2H up 数组含越界节点。 ");
            }
        }
    }
    if (root_count != 1) {
        throw std::runtime_error("H2H mmap 索引的根节点数量不是 1。 ");
    }
    for (std::uint64_t index = 0; index < header_->label_count; ++index) {
        if (!std::isfinite(dis_out_[index]) || !std::isfinite(dis_in_[index])) {
            throw std::runtime_error("强连通 H2H 索引包含无穷或 NaN 标签。 ");
        }
    }
}

std::uint32_t MappedH2HIndex::lca(std::uint32_t first, std::uint32_t second) const {
    if (depth_[first] < depth_[second]) {
        std::swap(first, second);
    }
    auto difference = depth_[first] - depth_[second];
    for (std::uint32_t level = 0; level < header_->level_count; ++level) {
        if ((difference & (1U << level)) != 0U) {
            first = up_[static_cast<std::size_t>(level) * header_->node_count + first];
        }
    }
    if (first == second) {
        return first;
    }
    for (std::uint32_t level = header_->level_count; level-- > 0;) {
        const auto first_up = up_[static_cast<std::size_t>(level) * header_->node_count + first];
        const auto second_up = up_[static_cast<std::size_t>(level) * header_->node_count + second];
        if (first_up != second_up) {
            first = first_up;
            second = second_up;
        }
    }
    return parent_[first];
}

double MappedH2HIndex::query(std::uint32_t source, std::uint32_t target) const {
    if (header_ == nullptr) {
        throw std::runtime_error("H2H 索引尚未打开。 ");
    }
    if (source >= header_->node_count || target >= header_->node_count) {
        throw std::out_of_range("H2H 查询节点编号越界。 ");
    }
    if (source == target) {
        return 0.0;
    }
    const auto ancestor = lca(source, target);
    auto result = std::numeric_limits<double>::infinity();
    for (auto offset = pos_offsets_[ancestor]; offset < pos_offsets_[ancestor + 1U]; ++offset) {
        const auto position = positions_[offset];
        const auto source_label = label_offsets_[source] + position;
        const auto target_label = label_offsets_[target] + position;
        if (source_label >= label_offsets_[source + 1U] ||
                target_label >= label_offsets_[target + 1U]) {
            throw std::runtime_error("H2H 查询 position 超出源/目标标签行。 ");
        }
        result = std::min(result, dis_out_[source_label] + dis_in_[target_label]);
    }
    if (!std::isfinite(result)) {
        throw std::runtime_error("H2H mmap 查询返回无穷距离。 ");
    }
    return result;
}

void MappedH2HIndex::query_batch(
    const std::uint32_t* sources,
    const std::uint32_t* targets,
    std::size_t count,
    double* output
) const {
    if (count != 0 && (sources == nullptr || targets == nullptr || output == nullptr)) {
        throw std::invalid_argument("H2H 批量查询收到空数组指针。 ");
    }
    for (std::size_t index = 0; index < count; ++index) {
        output[index] = query(sources[index], targets[index]);
    }
}

std::uint32_t MappedH2HIndex::node_count() const noexcept {
    return header_ == nullptr ? 0U : header_->node_count;
}

}  // namespace h2h
