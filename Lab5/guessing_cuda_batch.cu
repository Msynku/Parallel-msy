#include "guessing_cuda_batch.h"

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

__global__ void generate_password_batch_kernel(
    const char *prefix_chars,
    const int *prefix_offsets,
    const int *prefix_lens,
    const char *value_chars,
    const int *value_offsets,
    const int *value_lens,
    const int *candidate_task_ids,
    char *output,
    int max_out_len,
    int total_candidates
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_candidates) return;

    int task_id = candidate_task_ids[idx];
    const char *prefix = prefix_chars + prefix_offsets[task_id];
    const char *value = value_chars + value_offsets[idx];
    int prefix_len = prefix_lens[task_id];
    int value_len = value_lens[idx];
    char *out = output + static_cast<size_t>(idx) * max_out_len;

    int pos = 0;
    for (int i = 0; i < prefix_len && pos < max_out_len - 1; ++i) {
        out[pos++] = prefix[i];
    }
    for (int i = 0; i < value_len && pos < max_out_len - 1; ++i) {
        out[pos++] = value[i];
    }
    out[pos] = '\0';
}

static bool check_cuda(cudaError_t ret, const char *msg) {
    if (ret != cudaSuccess) {
        cerr << "[CUDA batch] " << msg << " failed: " << cudaGetErrorString(ret) << endl;
        return false;
    }
    return true;
}

template <typename T>
static bool ensure_device_capacity(T **ptr, size_t &capacity_bytes, size_t needed_bytes, const char *name) {
    if (needed_bytes == 0) needed_bytes = sizeof(T);
    if (*ptr != nullptr && capacity_bytes >= needed_bytes) return true;

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

struct BatchGpuBuffers {
    char *d_prefix_chars = nullptr;
    char *d_value_chars = nullptr;
    char *d_output = nullptr;
    int *d_prefix_offsets = nullptr;
    int *d_prefix_lens = nullptr;
    int *d_value_offsets = nullptr;
    int *d_value_lens = nullptr;
    int *d_candidate_task_ids = nullptr;

    size_t prefix_chars_capacity = 0;
    size_t value_chars_capacity = 0;
    size_t output_capacity = 0;
    size_t prefix_offsets_capacity = 0;
    size_t prefix_lens_capacity = 0;
    size_t value_offsets_capacity = 0;
    size_t value_lens_capacity = 0;
    size_t candidate_task_ids_capacity = 0;

    ~BatchGpuBuffers() {
        if (d_prefix_chars) cudaFree(d_prefix_chars);
        if (d_value_chars) cudaFree(d_value_chars);
        if (d_output) cudaFree(d_output);
        if (d_prefix_offsets) cudaFree(d_prefix_offsets);
        if (d_prefix_lens) cudaFree(d_prefix_lens);
        if (d_value_offsets) cudaFree(d_value_offsets);
        if (d_value_lens) cudaFree(d_value_lens);
        if (d_candidate_task_ids) cudaFree(d_candidate_task_ids);
    }
};

static BatchGpuBuffers &batch_buffers() {
    static BatchGpuBuffers buffers;
    return buffers;
}

static void generate_batch_on_cpu_fallback(
    const vector<GpuBatchTask> &tasks,
    vector<string> &guesses,
    int old_size
) {
    int out = old_size;
    for (const GpuBatchTask &task : tasks) {
        if (task.values == nullptr) continue;
        for (const string &value : *task.values) {
            guesses[out++] = task.prefix + value;
        }
    }
}

void generate_batch_on_gpu(
    const vector<GpuBatchTask> &tasks,
    vector<string> &guesses,
    int old_size
) {
    int task_count = static_cast<int>(tasks.size());
    if (task_count <= 0) return;

    int total_candidates = 0;
    int max_out_len = 1;
    size_t prefix_chars_bytes = 0;
    size_t value_chars_bytes = 0;

    for (const GpuBatchTask &task : tasks) {
        if (task.values == nullptr) continue;
        int prefix_len = static_cast<int>(task.prefix.size());
        prefix_chars_bytes += task.prefix.size();
        for (const string &value : *task.values) {
            ++total_candidates;
            value_chars_bytes += value.size();
            max_out_len = std::max(max_out_len, prefix_len + static_cast<int>(value.size()) + 1);
        }
    }

    if (total_candidates <= 0) return;

    if (static_cast<int>(guesses.size()) < old_size + total_candidates) {
        guesses.resize(old_size + total_candidates);
    }

    vector<char> h_prefix_chars(std::max<size_t>(1, prefix_chars_bytes), 0);
    vector<int> h_prefix_offsets(task_count, 0);
    vector<int> h_prefix_lens(task_count, 0);
    vector<char> h_value_chars(std::max<size_t>(1, value_chars_bytes), 0);
    vector<int> h_value_offsets(total_candidates, 0);
    vector<int> h_value_lens(total_candidates, 0);
    vector<int> h_candidate_task_ids(total_candidates, 0);

    size_t prefix_pos = 0;
    size_t value_pos = 0;
    int candidate_pos = 0;
    for (int task_id = 0; task_id < task_count; ++task_id) {
        const GpuBatchTask &task = tasks[task_id];
        h_prefix_offsets[task_id] = static_cast<int>(prefix_pos);
        h_prefix_lens[task_id] = static_cast<int>(task.prefix.size());
        if (!task.prefix.empty()) {
            std::memcpy(h_prefix_chars.data() + prefix_pos, task.prefix.data(), task.prefix.size());
            prefix_pos += task.prefix.size();
        }

        if (task.values == nullptr) continue;
        for (const string &value : *task.values) {
            h_value_offsets[candidate_pos] = static_cast<int>(value_pos);
            h_value_lens[candidate_pos] = static_cast<int>(value.size());
            h_candidate_task_ids[candidate_pos] = task_id;
            if (!value.empty()) {
                std::memcpy(h_value_chars.data() + value_pos, value.data(), value.size());
                value_pos += value.size();
            }
            ++candidate_pos;
        }
    }

    size_t output_bytes = static_cast<size_t>(total_candidates) * max_out_len;
    vector<char> h_output(output_bytes, 0);

    BatchGpuBuffers &buffers = batch_buffers();
    bool alloc_ok = true;
    alloc_ok = alloc_ok && ensure_device_capacity(&buffers.d_prefix_chars, buffers.prefix_chars_capacity, h_prefix_chars.size(), "cudaMalloc d_prefix_chars");
    alloc_ok = alloc_ok && ensure_device_capacity(&buffers.d_prefix_offsets, buffers.prefix_offsets_capacity, static_cast<size_t>(task_count) * sizeof(int), "cudaMalloc d_prefix_offsets");
    alloc_ok = alloc_ok && ensure_device_capacity(&buffers.d_prefix_lens, buffers.prefix_lens_capacity, static_cast<size_t>(task_count) * sizeof(int), "cudaMalloc d_prefix_lens");
    alloc_ok = alloc_ok && ensure_device_capacity(&buffers.d_value_chars, buffers.value_chars_capacity, h_value_chars.size(), "cudaMalloc d_value_chars");
    alloc_ok = alloc_ok && ensure_device_capacity(&buffers.d_value_offsets, buffers.value_offsets_capacity, static_cast<size_t>(total_candidates) * sizeof(int), "cudaMalloc d_value_offsets");
    alloc_ok = alloc_ok && ensure_device_capacity(&buffers.d_value_lens, buffers.value_lens_capacity, static_cast<size_t>(total_candidates) * sizeof(int), "cudaMalloc d_value_lens");
    alloc_ok = alloc_ok && ensure_device_capacity(&buffers.d_candidate_task_ids, buffers.candidate_task_ids_capacity, static_cast<size_t>(total_candidates) * sizeof(int), "cudaMalloc d_candidate_task_ids");
    alloc_ok = alloc_ok && ensure_device_capacity(&buffers.d_output, buffers.output_capacity, output_bytes, "cudaMalloc d_output");

    if (!alloc_ok) {
        generate_batch_on_cpu_fallback(tasks, guesses, old_size);
        return;
    }

    bool copy_ok = true;
    copy_ok = copy_ok && check_cuda(cudaMemcpy(buffers.d_prefix_chars, h_prefix_chars.data(), h_prefix_chars.size(), cudaMemcpyHostToDevice), "cudaMemcpy prefix_chars H2D");
    copy_ok = copy_ok && check_cuda(cudaMemcpy(buffers.d_prefix_offsets, h_prefix_offsets.data(), static_cast<size_t>(task_count) * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy prefix_offsets H2D");
    copy_ok = copy_ok && check_cuda(cudaMemcpy(buffers.d_prefix_lens, h_prefix_lens.data(), static_cast<size_t>(task_count) * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy prefix_lens H2D");
    copy_ok = copy_ok && check_cuda(cudaMemcpy(buffers.d_value_chars, h_value_chars.data(), h_value_chars.size(), cudaMemcpyHostToDevice), "cudaMemcpy value_chars H2D");
    copy_ok = copy_ok && check_cuda(cudaMemcpy(buffers.d_value_offsets, h_value_offsets.data(), static_cast<size_t>(total_candidates) * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy value_offsets H2D");
    copy_ok = copy_ok && check_cuda(cudaMemcpy(buffers.d_value_lens, h_value_lens.data(), static_cast<size_t>(total_candidates) * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy value_lens H2D");
    copy_ok = copy_ok && check_cuda(cudaMemcpy(buffers.d_candidate_task_ids, h_candidate_task_ids.data(), static_cast<size_t>(total_candidates) * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy candidate_task_ids H2D");

    if (!copy_ok) {
        generate_batch_on_cpu_fallback(tasks, guesses, old_size);
        return;
    }

    int block_size = 256;
    int grid_size = (total_candidates + block_size - 1) / block_size;
    generate_password_batch_kernel<<<grid_size, block_size>>>(
        buffers.d_prefix_chars,
        buffers.d_prefix_offsets,
        buffers.d_prefix_lens,
        buffers.d_value_chars,
        buffers.d_value_offsets,
        buffers.d_value_lens,
        buffers.d_candidate_task_ids,
        buffers.d_output,
        max_out_len,
        total_candidates
    );

    bool kernel_ok = check_cuda(cudaGetLastError(), "kernel launch");
    if (kernel_ok) {
        kernel_ok = check_cuda(cudaMemcpy(h_output.data(), buffers.d_output, output_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy output D2H");
    }

    if (!kernel_ok) {
        generate_batch_on_cpu_fallback(tasks, guesses, old_size);
        return;
    }

    for (int i = 0; i < total_candidates; ++i) {
        int task_id = h_candidate_task_ids[i];
        int out_len = h_prefix_lens[task_id] + h_value_lens[i];
        const char *begin = h_output.data() + static_cast<size_t>(i) * max_out_len;
        guesses[old_size + i].assign(begin, out_len);
    }
}
