#include "h2h_c_api.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>

#include "h2h_format.hpp"
#include "h2h_index.hpp"

namespace {

// 将错误文本安全截断到调用方缓冲区，并始终补零终止符。
void write_error(char* buffer, std::size_t buffer_size, const std::string& message) noexcept {
    if (buffer == nullptr || buffer_size == 0) {
        return;
    }
    const auto copy_size = std::min(buffer_size - 1U, message.size());
    std::memcpy(buffer, message.data(), copy_size);
    buffer[copy_size] = '\0';
}

}  // namespace

extern "C" {

void* h2h_open(const char* index_path, char* error_buffer, size_t error_buffer_size) {
    if (error_buffer != nullptr && error_buffer_size != 0) {
        error_buffer[0] = '\0';
    }
    if (index_path == nullptr || index_path[0] == '\0') {
        write_error(error_buffer, error_buffer_size, "index_path 不能为空。 ");
        return nullptr;
    }
    try {
        auto* index = new h2h::MappedH2HIndex();
        try {
            index->open(std::filesystem::u8path(index_path));
        } catch (...) {
            delete index;
            throw;
        }
        return index;
    } catch (const std::exception& error) {
        write_error(error_buffer, error_buffer_size, error.what());
        return nullptr;
    } catch (...) {
        write_error(error_buffer, error_buffer_size, "打开 H2H 索引时发生未知异常。 ");
        return nullptr;
    }
}

double h2h_query(void* handle, uint32_t source, uint32_t target) {
    if (handle == nullptr) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    try {
        return static_cast<h2h::MappedH2HIndex*>(handle)->query(source, target);
    } catch (...) {
        return std::numeric_limits<double>::quiet_NaN();
    }
}

int h2h_query_batch(
    void* handle,
    const uint32_t* sources,
    const uint32_t* targets,
    size_t count,
    double* output
) {
    if (handle == nullptr) {
        return -1;
    }
    try {
        static_cast<h2h::MappedH2HIndex*>(handle)->query_batch(
            sources, targets, count, output
        );
        return 0;
    } catch (...) {
        return -2;
    }
}

void h2h_close(void* handle) {
    delete static_cast<h2h::MappedH2HIndex*>(handle);
}

uint32_t h2h_api_version(void) {
    return h2h::kApiVersion;
}

}  // extern "C"
