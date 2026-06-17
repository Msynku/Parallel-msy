#include "PCFG.h"
#include "guessing_cuda_batch.h"

#include <algorithm>
#include <string>
#include <vector>

using namespace std;

int PopNextBatch(PriorityQueue &q, int max_pts, int max_guesses);

void PriorityQueue::CalProb(PT &pt)
{
    pt.prob = pt.preterm_prob;

    int index = 0;
    for (int idx : pt.curr_indices)
    {
        if (pt.content[index].type == 1)
        {
            pt.prob *= m.letters[m.FindLetter(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.letters[m.FindLetter(pt.content[index])].total_freq;
        }
        if (pt.content[index].type == 2)
        {
            pt.prob *= m.digits[m.FindDigit(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.digits[m.FindDigit(pt.content[index])].total_freq;
        }
        if (pt.content[index].type == 3)
        {
            pt.prob *= m.symbols[m.FindSymbol(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.symbols[m.FindSymbol(pt.content[index])].total_freq;
        }
        index += 1;
    }
}

void PriorityQueue::init()
{
    for (PT pt : m.ordered_pts)
    {
        for (segment seg : pt.content)
        {
            if (seg.type == 1)
            {
                pt.max_indices.emplace_back(static_cast<int>(m.letters[m.FindLetter(seg)].ordered_values.size()));
            }
            if (seg.type == 2)
            {
                pt.max_indices.emplace_back(static_cast<int>(m.digits[m.FindDigit(seg)].ordered_values.size()));
            }
            if (seg.type == 3)
            {
                pt.max_indices.emplace_back(static_cast<int>(m.symbols[m.FindSymbol(seg)].ordered_values.size()));
            }
        }

        pt.preterm_prob = float(m.preterm_freq[m.FindPT(pt)]) / m.total_preterm;
        CalProb(pt);
        priority.emplace_back(pt);
    }
}

vector<PT> PT::NewPTs()
{
    vector<PT> res;

    if (content.size() == 1)
    {
        return res;
    }

    int init_pivot = pivot;
    for (int i = pivot; i < static_cast<int>(curr_indices.size()) - 1; i += 1)
    {
        curr_indices[i] += 1;

        if (curr_indices[i] < max_indices[i])
        {
            pivot = i;
            res.emplace_back(*this);
        }

        curr_indices[i] -= 1;
    }

    pivot = init_pivot;
    return res;
}

static const vector<string> *ordered_values_for(PriorityQueue &q, const segment &seg)
{
    if (seg.type == 1)
    {
        return &q.m.letters[q.m.FindLetter(seg)].ordered_values;
    }
    if (seg.type == 2)
    {
        return &q.m.digits[q.m.FindDigit(seg)].ordered_values;
    }
    if (seg.type == 3)
    {
        return &q.m.symbols[q.m.FindSymbol(seg)].ordered_values;
    }
    return nullptr;
}

static string build_prefix(PriorityQueue &q, const PT &pt)
{
    if (pt.content.size() <= 1)
    {
        return "";
    }

    string prefix;
    int seg_idx = 0;
    for (int idx : pt.curr_indices)
    {
        if (seg_idx == static_cast<int>(pt.content.size()) - 1)
        {
            break;
        }

        const vector<string> *values = ordered_values_for(q, pt.content[seg_idx]);
        if (values != nullptr && idx >= 0 && idx < static_cast<int>(values->size()))
        {
            prefix += (*values)[idx];
        }
        seg_idx += 1;
    }

    return prefix;
}

static bool make_task(PriorityQueue &q, const PT &pt, GpuBatchTask &task, int &candidate_count)
{
    if (pt.content.empty() || pt.max_indices.empty())
    {
        candidate_count = 0;
        return false;
    }

    int last = static_cast<int>(pt.content.size()) - 1;
    const vector<string> *values = ordered_values_for(q, pt.content[last]);
    if (values == nullptr)
    {
        candidate_count = 0;
        return false;
    }

    candidate_count = pt.max_indices[last];
    task.prefix = build_prefix(q, pt);
    task.values = values;
    return candidate_count > 0;
}

static void insert_by_prob(PriorityQueue &q, const PT &pt)
{
    auto iter = q.priority.begin();
    while (iter != q.priority.end() && iter->prob >= pt.prob)
    {
        ++iter;
    }
    q.priority.insert(iter, pt);
}

void PriorityQueue::Generate(PT pt)
{
    GpuBatchTask task;
    int candidate_count = 0;
    if (!make_task(*this, pt, task, candidate_count))
    {
        return;
    }

    int old_size = static_cast<int>(guesses.size());
    guesses.resize(old_size + candidate_count);
    vector<GpuBatchTask> tasks;
    tasks.emplace_back(task);
    generate_batch_on_gpu(tasks, guesses, old_size);
    total_guesses += candidate_count;
}

void PriorityQueue::PopNext()
{
    PopNextBatch(*this, 1, 0);
}

int PopNextBatch(PriorityQueue &q, int max_pts, int max_guesses)
{
    if (max_pts <= 0)
    {
        max_pts = 1;
    }

    vector<GpuBatchTask> tasks;
    tasks.reserve(max_pts);
    int batch_candidates = 0;
    int old_size = static_cast<int>(q.guesses.size());

    while (!q.priority.empty() && static_cast<int>(tasks.size()) < max_pts)
    {
        PT pt = q.priority.front();
        GpuBatchTask task;
        int pt_candidates = 0;
        bool ok = make_task(q, pt, task, pt_candidates);

        if (ok && max_guesses > 0 && !tasks.empty() && batch_candidates + pt_candidates > max_guesses)
        {
            break;
        }

        q.priority.erase(q.priority.begin());

        vector<PT> new_pts = pt.NewPTs();
        for (PT next_pt : new_pts)
        {
            q.CalProb(next_pt);
            insert_by_prob(q, next_pt);
        }

        if (ok)
        {
            tasks.emplace_back(task);
            batch_candidates += pt_candidates;
        }
    }

    if (batch_candidates <= 0)
    {
        return 0;
    }

    q.guesses.resize(old_size + batch_candidates);
    generate_batch_on_gpu(tasks, q.guesses, old_size);
    q.total_guesses += batch_candidates;
    return batch_candidates;
}
