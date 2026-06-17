#pragma once

#include <string>
#include <vector>

// Generate guesses on GPU: guesses[old_size + i] = prefix + values[i]
// prefix: already-instantiated prefix (empty string for single-segment case)
// values: all candidate values of the last segment (a->ordered_values)
// guesses: result array in PriorityQueue, already resized before call
// old_size: length of guesses before this batch
void generate_on_gpu(
    const std::string &prefix,
    const std::vector<std::string> &values,
    std::vector<std::string> &guesses,
    int old_size
);
