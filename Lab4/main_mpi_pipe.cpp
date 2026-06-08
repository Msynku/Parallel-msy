#include "PCFG.h"
#include "md5.h"

#include <mpi.h>
#include <chrono>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <string>
#include <iostream>
#include <cstring>
#include <cstdlib>
#include <algorithm>

using namespace std;
using namespace chrono;

static const int TAG_CMD  = 100;
static const int TAG_SIZE = 101;
static const int TAG_DATA = 102;
static const int TAG_DONE = 103;

static const int CMD_HASH = 1;
static const int CMD_STOP = 2;

// 把 vector<string> 打包成二进制 buffer
// 格式：
// [int 字符串个数]
// [int len1][内容1]
// [int len2][内容2]
// ...
vector<char> pack_passwords(const vector<string>& passwords, int l, int r)
{
    vector<char> buf;

    int cnt = r - l;
    buf.resize(sizeof(int));
    memcpy(buf.data(), &cnt, sizeof(int));

    for (int i = l; i < r; i++)
    {
        int len = (int)passwords[i].size();

        size_t old_size = buf.size();
        buf.resize(old_size + sizeof(int) + len);

        memcpy(buf.data() + old_size, &len, sizeof(int));
        memcpy(buf.data() + old_size + sizeof(int), passwords[i].data(), len);
    }

    return buf;
}

// 从二进制 buffer 还原 vector<string>
vector<string> unpack_passwords(const vector<char>& buf)
{
    vector<string> passwords;

    if (buf.empty())
    {
        return passwords;
    }

    size_t pos = 0;
    int cnt = 0;

    memcpy(&cnt, buf.data() + pos, sizeof(int));
    pos += sizeof(int);

    passwords.reserve(cnt);

    for (int i = 0; i < cnt; i++)
    {
        int len = 0;
        memcpy(&len, buf.data() + pos, sizeof(int));
        pos += sizeof(int);

        string s(buf.data() + pos, buf.data() + pos + len);
        pos += len;

        passwords.push_back(s);
    }

    return passwords;
}
struct PendingHashTask
{
    bool active = false;
    int worker_count = 0;
    double start_time = 0.0;
};

// 只把任务发给 worker，不等待结果
// 0号进程不参与哈希，专心继续生成下一批
void start_hash_workers_only(const vector<string>& passwords,
                             int world_size,
                             PendingHashTask& task)
{
    int total = (int)passwords.size();

    if (total == 0 || world_size <= 1)
    {
        task.active = false;
        task.worker_count = 0;
        return;
    }

    int workers = world_size - 1;

    for (int rank = 1; rank < world_size; rank++)
    {
        int worker_id = rank - 1;

        int l = total * worker_id / workers;
        int r = total * (worker_id + 1) / workers;

        int cmd = CMD_HASH;
        MPI_Send(&cmd, 1, MPI_INT, rank, TAG_CMD, MPI_COMM_WORLD);

        vector<char> buf = pack_passwords(passwords, l, r);
        int nbytes = (int)buf.size();

        MPI_Send(&nbytes, 1, MPI_INT, rank, TAG_SIZE, MPI_COMM_WORLD);

        if (nbytes > 0)
        {
            MPI_Send(buf.data(), nbytes, MPI_CHAR, rank, TAG_DATA, MPI_COMM_WORLD);
        }
    }

    task.active = true;
    task.worker_count = workers;
    task.start_time = MPI_Wtime();
}
double wait_hash_workers(PendingHashTask& task)
{
    if (!task.active)
    {
        return 0.0;
    }

    for (int i = 0; i < task.worker_count; i++)
    {
        int done = 0;
        MPI_Recv(&done, 1, MPI_INT, MPI_ANY_SOURCE, TAG_DONE,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    double end_time = MPI_Wtime();
    double elapsed = end_time - task.start_time;

    task.active = false;
    task.worker_count = 0;

    return elapsed;
}
// 0号进程调用：把一批口令分给所有 MPI 进程做哈希
void hash_batch_mpi_master(const vector<string>& passwords, int world_size)
{
    int total = (int)passwords.size();

    if (total == 0)
    {
        return;
    }

    // 只有1个进程时，退化成原来的本地 SIMD 哈希
    if (world_size == 1)
    {
        MD5HashBatchSIMD(passwords);
        return;
    }

    // 给 1 ~ world_size-1 号进程发送任务
    for (int rank = 1; rank < world_size; rank++)
    {
        int l = total * rank / world_size;
        int r = total * (rank + 1) / world_size;

        int cmd = CMD_HASH;
        MPI_Send(&cmd, 1, MPI_INT, rank, TAG_CMD, MPI_COMM_WORLD);

        vector<char> buf = pack_passwords(passwords, l, r);
        int nbytes = (int)buf.size();

        MPI_Send(&nbytes, 1, MPI_INT, rank, TAG_SIZE, MPI_COMM_WORLD);

        if (nbytes > 0)
        {
            MPI_Send(buf.data(), nbytes, MPI_CHAR, rank, TAG_DATA, MPI_COMM_WORLD);
        }
    }

    // 0号进程自己也做一份任务
    int l0 = 0;
    int r0 = total * 1 / world_size;

    if (r0 > l0)
    {
        vector<string> local_part(passwords.begin() + l0, passwords.begin() + r0);
        MD5HashBatchSIMD(local_part);
    }

    // 等待其他进程完成
    for (int rank = 1; rank < world_size; rank++)
    {
        int done = 0;
        MPI_Recv(&done, 1, MPI_INT, rank, TAG_DONE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}

// 非0号进程调用：等待0号进程分发哈希任务
void worker_loop()
{
    while (true)
    {
        int cmd = 0;
        MPI_Recv(&cmd, 1, MPI_INT, 0, TAG_CMD, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        if (cmd == CMD_STOP)
        {
            break;
        }

        if (cmd == CMD_HASH)
        {
            int nbytes = 0;
            MPI_Recv(&nbytes, 1, MPI_INT, 0, TAG_SIZE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            vector<char> buf(nbytes);

            if (nbytes > 0)
            {
                MPI_Recv(buf.data(), nbytes, MPI_CHAR, 0, TAG_DATA, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }

            vector<string> passwords = unpack_passwords(buf);

            if (!passwords.empty())
            {
                MD5HashBatchSIMD(passwords);
            }

            int done = (int)passwords.size();
            MPI_Send(&done, 1, MPI_INT, 0, TAG_DONE, MPI_COMM_WORLD);
        }
    }
}

// 0号进程调用：通知其他进程退出
void stop_workers(int world_size)
{
    for (int rank = 1; rank < world_size; rank++)
    {
        int cmd = CMD_STOP;
        MPI_Send(&cmd, 1, MPI_INT, rank, TAG_CMD, MPI_COMM_WORLD);
    }
}

// 保留你原 main.cpp 里的单条 MD5 正确性测试
bool TestMD5HashSingle()
{
    cout << "Testing MD5Hash correctness..." << endl;

    string test_pws[8] = {
        "123456", "password", "12345678", "qwerty",
        "123456789", "12345", "1234", "111111"
    };

    string test_hashes[8] = {
        "e10adc3949ba59abbe56e057f20f883e",
        "5f4dcc3b5aa765d61d8327deb882cf99",
        "25d55ad283aa400af464c76d713c07ad",
        "d8578edf8458ce06fbc5bb76a58c5ca4",
        "25f9e794323b453885f5181f1b624d0b",
        "827ccb0eea8a706c4c34a16891f84e7b",
        "81dc9bdb52d04dc20036dbd8313ed055",
        "96e79218965eb72c92a549dd5a330112"
    };

    for (int i = 0; i < 8; i++)
    {
        bit32 state[4];
        MD5Hash(test_pws[i], state);

        stringstream ss;
        for (int j = 0; j < 4; j++)
        {
            ss << setw(8) << setfill('0') << hex << state[j];
        }

        if (ss.str() != test_hashes[i])
        {
            cout << "MD5Hash test failed for " << test_pws[i] << "!" << endl;
            cout << "Expected: " << test_hashes[i] << endl;
            cout << "Got:      " << ss.str() << endl;
            return false;
        }
    }

    cout << "MD5Hash test passed!" << endl;
    return true;
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int world_size = 1;
    int rank = 0;

    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // 非0号进程不训练、不生成，只等待0号进程分配哈希任务
    if (rank != 0)
    {
        worker_loop();
        MPI_Finalize();
        return 0;
    }

    // 下面只有0号进程执行
    cout << "MPI processes: " << world_size << endl;

    int generate_limit = 10000000;
    if (argc >= 2)
    {
        generate_limit = atoi(argv[1]);
    }

    cout << "Generate limit: " << generate_limit << endl;

    if (!TestMD5HashSingle())
    {
        stop_workers(world_size);
        MPI_Finalize();
        return 1;
    }

    if (!TestMD5HashSIMD())
    {
        stop_workers(world_size);
        MPI_Finalize();
        return 1;
    }

    double time_hash = 0.0;
    double time_guess = 0.0;
    double time_train = 0.0;

    PriorityQueue q;

    double train_start = MPI_Wtime();

    q.m.train("/guessdata/Rockyou-singleLined-full.txt");
    q.m.order();

    double train_end = MPI_Wtime();
    time_train = train_end - train_start;

    q.init();

    cout << "Priority queue initialized." << endl;

    const int HASH_FLUSH_LIMIT = 1000000;
    const int REPORT_STEP = 100000;

    long long history = 0;
    long long next_report = REPORT_STEP;
    PendingHashTask pending_task;
    double total_start = MPI_Wtime();

    while (!q.priority.empty() && history < generate_limit)
    {
        q.PopNext();

        // 每生成10万条左右，输出一次进度
        while (history + (long long)q.guesses.size() >= next_report)
        {
            cout << "Guesses generated: " << next_report << endl;
            next_report += REPORT_STEP;
        }

        // 如果超过总生成上限，就把最后一批截断到刚好够
        if (history + (long long)q.guesses.size() >= generate_limit)
        {
            long long need = generate_limit - history;

            if (need < (long long)q.guesses.size())
            {
                q.guesses.resize((size_t)need);
            }

            if (!q.guesses.empty())
            {
                double hash_start = MPI_Wtime();
                hash_batch_mpi_master(q.guesses, world_size);
                double hash_end = MPI_Wtime();

                time_hash += hash_end - hash_start;

                history += q.guesses.size();
                q.guesses.clear();
            }

            break;
        }

        // 缓存达到一定数量后，进行一次 MPI 并行哈希
        if ((int)q.guesses.size() >= HASH_FLUSH_LIMIT)
{
    // 如果上一批还没哈希完，先等待上一批完成
    time_hash += wait_hash_workers(pending_task);

    // 记录已经生成的口令数量
    history += q.guesses.size();

    // 启动当前批次的 worker 哈希，但不等待
    start_hash_workers_only(q.guesses, world_size, pending_task);

    // 0号进程立刻清空缓存，继续生成下一批
    q.guesses.clear();
}
    }
    time_hash += wait_hash_workers(pending_task);
    double total_end = MPI_Wtime();

    // 总时间 = 生成 + 哈希 + 通信
    // 这里沿用你原来的输出格式：Guess time 输出非哈希部分
    time_guess = (total_end - total_start) - time_hash;

    cout << "Guess time:" << time_guess << "seconds" << endl;
    cout << "Hash time:" << time_hash << "seconds" << endl;
    cout << "Train time:" << time_train << "seconds" << endl;

    stop_workers(world_size);

    MPI_Finalize();
    return 0;
}
