#pragma once

// 稳定 C ABI：不跨 DLL 边界暴露任何 C++ STL 类型。

#include <stddef.h>
#include <stdint.h>

#ifdef _WIN32
#define H2H_EXPORT __declspec(dllexport)
#else
#define H2H_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

// 打开并校验索引；失败返回 NULL，并尽量写入 UTF-8 错误信息。
H2H_EXPORT void* h2h_open(const char* index_path, char* error_buffer, size_t error_buffer_size);

// 查询单个有向节点对；句柄或节点非法时返回 NaN。
H2H_EXPORT double h2h_query(void* handle, uint32_t source, uint32_t target);

// 批量查询成功返回 0；参数或任一节点非法时返回非零。
H2H_EXPORT int h2h_query_batch(
    void* handle,
    const uint32_t* sources,
    const uint32_t* targets,
    size_t count,
    double* output
);

// 关闭索引并释放句柄；NULL 安全。
H2H_EXPORT void h2h_close(void* handle);

// 返回原生 ABI/索引实现版本。
H2H_EXPORT uint32_t h2h_api_version(void);

#ifdef __cplusplus
}
#endif
