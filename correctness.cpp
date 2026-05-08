#include "PCFG.h"
#include <chrono>
#include <fstream>
#include <sstream>
#include "md5.h"
#include <iomanip>
using namespace std;
using namespace chrono;

// 编译指令如下：
// g++ correctness.cpp train.cpp guessing.cpp md5.cpp -o test.exe


// 通过这个函数，你可以验证你实现的SIMD哈希函数的正确性
int main()
{
    cout << "Testing serial MD5Hash correctness..." << endl;

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
            cout << "Serial MD5Hash test failed for " << test_pws[i] << endl;
            cout << "Expected: " << test_hashes[i] << endl;
            cout << "Got:      " << ss.str() << endl;
            return 1;
        }
    }

    cout << "Serial MD5Hash test passed!" << endl;

    // 测试 SIMD 版本
    if (!TestMD5HashSIMD())
    {
        return 1;
    }

    return 0;
}
