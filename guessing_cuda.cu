#include "guessing_cuda.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

using std::cerr;
using std::endl;
using std::string;
using std::vector;

// Each thread generates one candidate password: output[idx] = prefix + values[idx]
__global__ void generate_password_kernel(
    const char *prefix,
    int prefix_len,
    const char *values,
    const int *value_lens,
    int max_value_len,
    char *output,
    int max_out_len,
    int total
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    char *out = output + idx * max_out_len;
    const char *val = values + idx * max_value_len;
    int val_len = value_lens[idx];

    int pos = 0;

    // Copy prefix
    for (int i = 0; i < prefix_len && pos < max_out_len - 1; ++i) {
        out[pos++] = prefix[i];
    }

    // Copy values[idx]
    for (int i = 0; i < val_len && pos < max_out_len - 1; ++i) {
        out[pos++] = val[i];
    }

    out[pos] = '\0';
}

static void generate_on_cpu_fallback(
    const string &prefix,
    const vector<string> &values,
    vector<string> &guesses,
    int old_size
) {
    int n = static_cast<int>(values.size());
    for (int i = 0; i < n; ++i) {
        guesses[old_size + i] = prefix + values[i];
    }
}

static bool check_cuda(cudaError_t ret, const char *msg) {
    if (ret != cudaSuccess) {
        cerr << "[CUDA] " << msg << " failed: " << cudaGetErrorString(ret) << endl;
        return false;
    }
    return true;
}

template <typename T>
static bool ensure_device_capacity(T **ptr, size_t &capacity_bytes, size_t needed_bytes, const char *name) {
    if (needed_bytes == 0) {
        needed_bytes = sizeof(T);
    }

    if (*ptr != nullptr && capacity_bytes >= needed_bytes) {
        return true;
    }

    if (*ptr != nullptr) {
        cudaFree(*ptr);
        *ptr = nullptr;
        capacity_bytes = 0;
    }

    if (!check_cuda(cudaMalloc(reinterpret_cast<void **>(ptr), needed_bytes), name)) {
        return false;
    }

    capacity_bytes = needed_bytes;
    return true;
}

struct GpuGenerateBuffers {
    char *d_prefix = nullptr;
    char *d_values = nullptr;
    char *d_output = nullptr;
    int *d_value_lens = nullptr;

    size_t prefix_capacity = 0;
    size_t values_capacity = 0;
    size_t output_capacity = 0;
    size_t value_lens_capacity = 0;

    ~GpuGenerateBuffers() {
        if (d_prefix) cudaFree(d_prefix);
        if (d_values) cudaFree(d_values);
        if (d_output) cudaFree(d_output);
        if (d_value_lens) cudaFree(d_value_lens);
    }
};

static GpuGenerateBuffers &gpu_buffers() {
    static GpuGenerateBuffers buffers;
    return buffers;
}

void generate_on_gpu(
    const string &prefix,
    const vector<string> &values,
    vector<string> &guesses,
    int old_size
) {
    int total = static_cast<int>(values.size());
    if (total <= 0) return;

    // Ensure guesses has enough space
    if (static_cast<int>(guesses.size()) < old_size + total) {
        guesses.resize(old_size + total);
    }

    int prefix_len = static_cast<int>(prefix.size());

    int max_value_len = 1;
    for (int i = 0; i < total; ++i) {
        int len = static_cast<int>(values[i].size());
        if (len > max_value_len) max_value_len = len;
    }

    int max_out_len = prefix_len + max_value_len + 1;  // +1 for '\0'

    // ---- Pack host data into flat arrays ----
    size_t prefix_bytes = static_cast<size_t>(std::max(1, prefix_len)) * sizeof(char);
    vector<char> h_prefix(prefix_bytes, 0);
    if (prefix_len > 0) {
        std::memcpy(h_prefix.data(), prefix.data(), prefix_len);
    }

    size_t values_bytes = static_cast<size_t>(total) * max_value_len;
    vector<char> h_values(values_bytes);
    vector<int> h_value_lens(total, 0);

    for (int i = 0; i < total; ++i) {
        int len = static_cast<int>(values[i].size());
        h_value_lens[i] = len;
        if (len > 0) {
            std::memcpy(
                h_values.data() + static_cast<size_t>(i) * max_value_len,
                values[i].data(),
                len
            );
        }
    }

    size_t output_bytes = static_cast<size_t>(total) * max_out_len;
    vector<char> h_output(output_bytes);

    // ---- Reuse device memory across single-PT launches ----
    GpuGenerateBuffers &buffers = gpu_buffers();
    bool alloc_ok = true;

    alloc_ok = alloc_ok && ensure_device_capacity(
        &buffers.d_prefix,
        buffers.prefix_capacity,
        prefix_bytes,
        "cudaMalloc d_prefix"
    );

    alloc_ok = alloc_ok && ensure_device_capacity(
        &buffers.d_values,
        buffers.values_capacity,
        values_bytes * sizeof(char),
        "cudaMalloc d_values"
    );

    alloc_ok = alloc_ok && ensure_device_capacity(
        &buffers.d_value_lens,
        buffers.value_lens_capacity,
        static_cast<size_t>(total) * sizeof(int),
        "cudaMalloc d_value_lens"
    );

    alloc_ok = alloc_ok && ensure_device_capacity(
        &buffers.d_output,
        buffers.output_capacity,
        output_bytes * sizeof(char),
        "cudaMalloc d_output"
    );

    if (!alloc_ok) {
        generate_on_cpu_fallback(prefix, values, guesses, old_size);
        return;
    }

    // ---- Copy data to device ----
    bool copy_ok = true;

    copy_ok = copy_ok && check_cuda(
        cudaMemcpy(buffers.d_prefix, h_prefix.data(),
                   prefix_bytes,
                   cudaMemcpyHostToDevice),
        "cudaMemcpy prefix H2D"
    );

    copy_ok = copy_ok && check_cuda(
        cudaMemcpy(buffers.d_values, h_values.data(),
                   values_bytes * sizeof(char),
                   cudaMemcpyHostToDevice),
        "cudaMemcpy values H2D"
    );

    copy_ok = copy_ok && check_cuda(
        cudaMemcpy(buffers.d_value_lens, h_value_lens.data(),
                   static_cast<size_t>(total) * sizeof(int),
                   cudaMemcpyHostToDevice),
        "cudaMemcpy value_lens H2D"
    );

    if (!copy_ok) {
        generate_on_cpu_fallback(prefix, values, guesses, old_size);
        return;
    }

    // ---- Launch kernel ----
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;

    generate_password_kernel<<<grid_size, block_size>>>(
        buffers.d_prefix,
        prefix_len,
        buffers.d_values,
        buffers.d_value_lens,
        max_value_len,
        buffers.d_output,
        max_out_len,
        total
    );

    bool kernel_ok = true;
    kernel_ok = kernel_ok && check_cuda(cudaGetLastError(), "kernel launch");

    if (kernel_ok) {
        kernel_ok = kernel_ok && check_cuda(
            cudaMemcpy(h_output.data(), buffers.d_output,
                       output_bytes * sizeof(char),
                       cudaMemcpyDeviceToHost),
            "cudaMemcpy output D2H"
        );
    }

    // ---- Copy results back to guesses ----
    if (kernel_ok) {
        for (int i = 0; i < total; ++i) {
            const char *begin = h_output.data() + static_cast<size_t>(i) * max_out_len;
            guesses[old_size + i].assign(begin, prefix_len + h_value_lens[i]);
        }
    } else {
        generate_on_cpu_fallback(prefix, values, guesses, old_size);
    }
}
