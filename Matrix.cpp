#include <iostream>
#include <windows.h>
using namespace std;

const int N = 5000;

double A[N][N];
double b[N];
double result[N];

LARGE_INTEGER freq;

void init()
{
    QueryPerformanceFrequency(&freq);
    for (int i = 0; i < N; i++) {
        b[i] = i * 0.1;
        result[i] = 0;
        for (int j = 0; j < N; j++)
            A[i][j] = 1.0;
    }
}

void reset()
{
    for (int i = 0; i < N; i++)
        result[i] = 0;
}

void col()
{
    for (int j = 0; j < N; j++)
        for (int i = 0; i < N; i++)
            result[j] += A[i][j] * b[i];
}

void row()
{
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            result[j] += A[i][j] * b[i];
}

int main()
{
    init();
    cout << "Matrix Size: " << N << " x " << N << endl;

    LARGE_INTEGER t0, t1;

    //列算法测试
    reset();
    QueryPerformanceCounter(&t0);
    col(); 
    QueryPerformanceCounter(&t1);
    double t_col = (double)(t1.QuadPart - t0.QuadPart) / freq.QuadPart;
    cout << "Column Time: " << t_col << " seconds" << endl;

    //行算法测试
    reset();
    QueryPerformanceCounter(&t0);
    row(); 
    QueryPerformanceCounter(&t1);
    double t_row = (double)(t1.QuadPart - t0.QuadPart) / freq.QuadPart;
    cout << "Row Time: " << t_row << " seconds" << endl;

    cout << "Speedup (Col/Row): " << t_col / t_row << "x" << endl;

    return 0;
}
