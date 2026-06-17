#include "PCFG.h"
#include "md5.h"

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

using namespace std;
using namespace chrono;

int PopNextBatch(PriorityQueue &q, int max_pts, int max_guesses);

int main(int argc, char **argv)
{
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
        for (int j = 0; j < 4; j++) {
            ss << setw(8) << setfill('0') << hex << state[j];
        }
        if (ss.str() != test_hashes[i]) {
            cout << "MD5Hash test failed for " << test_pws[i] << endl;
            cout << "Expected: " << test_hashes[i] << "\nGot:      " << ss.str() << endl;
            return 1;
        }
    }
    cout << "MD5Hash test passed!" << endl;

    if (!TestMD5HashSIMD()) {
        return 1;
    }

    double time_hash = 0;
    double time_guess = 0;
    double time_train = 0;

    int generate_limit = 100000000;
    int batch_pt_limit = 128;
    int batch_guess_limit = 262144;
    if (argc >= 2) {
        int parsed = atoi(argv[1]);
        if (parsed > 0) generate_limit = parsed;
    }
    if (argc >= 3) {
        int parsed = atoi(argv[2]);
        if (parsed > 0) batch_pt_limit = parsed;
    }
    if (argc >= 4) {
        int parsed = atoi(argv[3]);
        if (parsed > 0) batch_guess_limit = parsed;
    }

    PriorityQueue q;
    auto start_train = system_clock::now();

    const string TRAIN_PATH = "./test_passwords.txt";
    cout << "Training from: " << TRAIN_PATH << endl;
    q.m.train(TRAIN_PATH);
    q.m.order();

    auto end_train = system_clock::now();
    auto duration_train = duration_cast<microseconds>(end_train - start_train);
    time_train = double(duration_train.count()) * microseconds::period::num / microseconds::period::den;

    q.init();
    cout << "Priority queue initialized, starting batch GPU generation..." << endl;
    cout << "Generate limit: " << generate_limit << endl;
    cout << "Batch PT limit: " << batch_pt_limit << endl;
    cout << "Batch guess limit: " << batch_guess_limit << endl;

    int curr_num = 0;
    int history = 0;
    auto start = system_clock::now();

    while (!q.priority.empty())
    {
        int produced = PopNextBatch(q, batch_pt_limit, batch_guess_limit);
        if (produced <= 0) {
            break;
        }

        q.total_guesses = static_cast<int>(q.guesses.size());

        if (q.total_guesses - curr_num >= 10000 || q.priority.empty())
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
        history += static_cast<int>(q.guesses.size());
    }

    auto end = system_clock::now();
    auto duration = duration_cast<microseconds>(end - start);
    time_guess = double(duration.count()) * microseconds::period::num / microseconds::period::den;

    cout << "\n=== Batch GPU Final Results ===" << endl;
    cout << "Total guesses generated: " << history << endl;
    cout << "Guess time: " << time_guess - time_hash << " seconds" << endl;
    cout << "Hash time: " << time_hash << " seconds" << endl;
    cout << "Train time: " << time_train << " seconds" << endl;

    return 0;
}
