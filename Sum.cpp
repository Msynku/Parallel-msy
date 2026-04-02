#include <iostream>
#include <windows.h>

using namespace std;

const int N = 100000000;
float a[N];

LARGE_INTEGER freq;

// 初始化数据
void init() {
    QueryPerformanceFrequency(&freq);
    for (int i = 0; i < N; i++) {
        a[i] = i * 0.1f;
    }
}

// 算法 1：普通串行累加
float serial_sum() {
    float sum = 0;
    for (int i = 0; i < N; i++) {
        sum += a[i];
    }
    return sum;
}

// 算法 2：利用超标量特性的多路并行累加 (4路)
float multi_sum() {
    float sum1 = 0, sum2 = 0, sum3 = 0, sum4 = 0;
    // 每次处理 4 个元素，减少循环间的依赖
    for (int i = 0; i < N; i += 4) {
        sum1 += a[i];
        sum2 += a[i + 1];
        sum3 += a[i + 2];
        sum4 += a[i + 3];
    }
    return (sum1 + sum2) + (sum3 + sum4);
}

int main() {
    init();
    
    LARGE_INTEGER t0, t1;
    //测试串行累加
    QueryPerformanceCounter(&t0);
    serial_sum();
    QueryPerformanceCounter(&t1);
    double time_serial = (double)(t1.QuadPart - t0.QuadPart) / freq.QuadPart;
    cout << "Serial_Sum Time: " << time_serial << " s " << endl;

    // 测试超标量优化累加
    QueryPerformanceCounter(&t0);
     multi_sum();
    QueryPerformanceCounter(&t1);
    double time_super = (double)(t1.QuadPart - t0.QuadPart) / freq.QuadPart;
    cout << "<Multi_Sum  Time: " << time_super << " s " <<  endl;

   
    cout << "Speedup: " << time_serial / time_super << "x" << endl;

    return 0;
}
