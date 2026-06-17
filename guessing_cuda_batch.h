#pragma once

#include <string>
#include <vector>

struct GpuBatchTask {
    std::string prefix;
    const std::vector<std::string> *values = nullptr;
};

// Generate candidates for many PTs in one GPU launch.
// For each task t, candidates are task.prefix + task.values[i].
void generate_batch_on_gpu(
    const std::vector<GpuBatchTask> &tasks,
    std::vector<std::string> &guesses,
    int old_size
);
