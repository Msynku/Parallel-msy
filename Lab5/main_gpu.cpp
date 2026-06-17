#include "PCFG.h"
#include <chrono>
#include <fstream>
#include <sstream>
#include "md5.h"
#include <iomanip>
#include <cstdlib>
using namespace std;
using namespace chrono;

// ============================================================
// GPU 版本编译指令（本地 x86 + CUDA）：
//   nvcc main_gpu.cpp train.cpp guessing_gpu.cpp guessing_cuda.cu md5_x86.cpp -O2 -std=c++17 -o main_gpu.exe
//
// 如果没有 CUDA，先用 g++ 编译验证 C++ 侧逻辑（会走 CPU fallback）：
//   g++ main_gpu.cpp train.cpp guessing_gpu.cpp md5_x86.cpp -O2 -std=c++17 -o main_gpu_cpu_fallback.exe
// ============================================================

int main(int argc, char **argv)
{
    // ====== 1. MD5 正确性测试 ======
    cout << "Testing MD5Hash correctness..." << endl;
    string test_pws[8] = {"123456", "password", "12345678", "qwerty", "123456789", "12345", "1234", "111111"};
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
    for (int i = 0; i < 8; i++) {
        bit32 state[4];
        MD5Hash(test_pws[i], state);
        stringstream ss;
        for (int i1 = 0; i1 < 4; i1 += 1) {
            ss << std::setw(8) << std::setfill('0') << hex << state[i1];
        }
        if (ss.str() != test_hashes[i]) {
            cout << "MD5Hash test failed for " << test_pws[i] << "!" << endl;
            cout << "Expected: " << test_hashes[i] << "\nGot:      " << ss.str() << endl;
            return 1;
        }
    }
    cout << "MD5Hash test passed!" << endl;

    double time_hash = 0;
    double time_guess = 0;
    double time_train = 0;

    if (!TestMD5HashSIMD()) {
        return 1;
    }

    // ====== 2. 训练模型 ======
    PriorityQueue q;
    auto start_train = system_clock::now();

    // ★ 本地训练数据路径（修改这里指向你的口令文件）
    //    完整 RockYou 数据集请从服务器下载
    const string TRAIN_PATH = "./test_passwords.txt";
    cout << "Training from: " << TRAIN_PATH << endl;

    q.m.train(TRAIN_PATH);
    q.m.order();

    auto end_train = system_clock::now();
    auto duration_train = duration_cast<microseconds>(end_train - start_train);
    time_train = double(duration_train.count()) * microseconds::period::num / microseconds::period::den;

    // ====== 3. GPU 加速的口令生成 ======
    q.init();
    cout << "Priority queue initialized, starting GPU-accelerated generation..." << endl;

    int curr_num = 0;
    auto start = system_clock::now();
    int history = 0;
    int generate_limit = 100000000;
    if (argc >= 2) {
        int parsed_limit = atoi(argv[1]);
        if (parsed_limit > 0) {
            generate_limit = parsed_limit;
        }
    }
    cout << "Generate limit: " << generate_limit << endl;

    while (!q.priority.empty())
    {
        q.PopNext();
        q.total_guesses = q.guesses.size();

        if (q.total_guesses - curr_num >= 1000 || q.priority.empty())
        {
            curr_num = q.total_guesses;
            if (curr_num > 0) {
                cout << "Guesses buffered: " << history + q.total_guesses << endl;
            }
            if (history + q.total_guesses > generate_limit)
            {
                break;
            }
        }

        if (curr_num > 10000)
        {
            auto start_hash = system_clock::now();
            MD5HashBatchSIMD(q.guesses);
            auto end_hash = system_clock::now();
            auto duration = duration_cast<microseconds>(end_hash - start_hash);
            time_hash += double(duration.count()) * microseconds::period::num / microseconds::period::den;
            history += curr_num;
            curr_num = 0;
            q.guesses.clear();
        }
    }

    if (!q.guesses.empty())
    {
        auto start_hash = system_clock::now();
        MD5HashBatchSIMD(q.guesses);
        auto end_hash = system_clock::now();
        auto duration = duration_cast<microseconds>(end_hash - start_hash);
        time_hash += double(duration.count()) * microseconds::period::num / microseconds::period::den;
        history += q.guesses.size();
    }

    auto end = system_clock::now();
    auto duration = duration_cast<microseconds>(end - start);
    time_guess = double(duration.count()) * microseconds::period::num / microseconds::period::den;

    cout << "\n=== Final Results ===" << endl;
    cout << "Total guesses generated: " << history << endl;
    cout << "Guess time: " << time_guess - time_hash << " seconds" << endl;
    cout << "Hash time: " << time_hash << " seconds" << endl;
    cout << "Train time: " << time_train << " seconds" << endl;

    return 0;
}
