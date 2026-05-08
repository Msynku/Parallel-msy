#include "md5.h"
#include <iomanip>
#include <assert.h>
#include <chrono>
#include <arm_neon.h>
#include <vector>
#include <sstream>

using namespace std;
using namespace chrono;

/**
 * StringProcess: 将单个输入字符串转换成MD5计算所需的消息数组
 * @param input 输入
 * @param[out] n_byte 用于给调用者传递额外的返回值，即最终Byte数组的长度
 * @return Byte消息数组
 */
Byte *StringProcess(string input, int *n_byte)
{
    // 将输入的字符串转换为Byte为单位的数组
    Byte *blocks = (Byte *)input.c_str();
    int length = input.length();

    // 计算原始消息长度（以比特为单位）
    int bitLength = length * 8;

    // paddingBits: 原始消息需要的padding长度（以bit为单位）
    // 对于给定的消息，将其补齐至length%512==448为止
    // 需要注意的是，即便给定的消息满足length%512==448，也需要再pad 512bits
    int paddingBits = bitLength % 512;
    if (paddingBits > 448)
    {
        paddingBits = 512 - (paddingBits - 448);
    }
    else if (paddingBits < 448)
    {
        paddingBits = 448 - paddingBits;
    }
    else if (paddingBits == 448)
    {
        paddingBits = 512;
    }

    // 原始消息需要的padding长度（以Byte为单位）
    int paddingBytes = paddingBits / 8;

    // 创建最终的字节数组
    // length + paddingBytes + 8:
    // 1. length为原始消息的长度（bits）
    // 2. paddingBytes为原始消息需要的padding长度（Bytes）
    // 3. 在pad到length%512==448之后，需要额外附加64bits的原始消息长度，即8个bytes
    int paddedLength = length + paddingBytes + 8;
    Byte *paddedMessage = new Byte[paddedLength];

    // 复制原始消息
    memcpy(paddedMessage, blocks, length);

    // 添加填充字节。填充时，第一位为1，后面的所有位均为0。
    // 所以第一个byte是0x80
    paddedMessage[length] = 0x80;                            // 添加一个0x80字节
    memset(paddedMessage + length + 1, 0, paddingBytes - 1);  // 填充0字节

    // 添加消息长度（64比特，小端格式）
    for (int i = 0; i < 8; ++i)
    {
        // 特别注意此处应当将bitLength转换为uint64_t
        // 这里的length是原始消息的长度
        paddedMessage[length + paddingBytes + i] =
            ((uint64_t)length * 8 >> (i * 8)) & 0xFF;
    }

    // 验证长度是否满足要求。此时长度应当是512bit的倍数
    int residual = 8 * paddedLength % 512;
    // assert(residual == 0);

    // 在填充+添加长度之后，消息被分为n_blocks个512bit的部分
    *n_byte = paddedLength;
    return paddedMessage;
}

/**
 * MD5Hash: 将单个输入字符串转换成MD5
 * @param input 输入
 * @param[out] state 用于给调用者传递额外的返回值，即最终的缓冲区，也就是MD5的结果
 * @return Byte消息数组
 */
void MD5Hash(string input, bit32 *state)
{
    Byte *paddedMessage;
    int *messageLength = new int[1];

    for (int i = 0; i < 1; i += 1)
    {
        paddedMessage = StringProcess(input, &messageLength[i]);
        // cout << messageLength[i] << endl;
        assert(messageLength[i] == messageLength[0]);
    }

    int n_blocks = messageLength[0] / 64;

    // bit32* state = new bit32[4];
    state[0] = 0x67452301;
    state[1] = 0xefcdab89;
    state[2] = 0x98badcfe;
    state[3] = 0x10325476;

    // 逐block地更新state
    for (int i = 0; i < n_blocks; i += 1)
    {
        bit32 x[16];

        // 下面的处理，在理解上较为复杂
        for (int i1 = 0; i1 < 16; ++i1)
        {
            x[i1] = (paddedMessage[4 * i1 + i * 64]) |
                    (paddedMessage[4 * i1 + 1 + i * 64] << 8) |
                    (paddedMessage[4 * i1 + 2 + i * 64] << 16) |
                    (paddedMessage[4 * i1 + 3 + i * 64] << 24);
        }

        bit32 a = state[0];
        bit32 b = state[1];
        bit32 c = state[2];
        bit32 d = state[3];

        auto start = system_clock::now();

        /* Round 1 */
        FF(a, b, c, d, x[0], s11, 0xd76aa478);
        FF(d, a, b, c, x[1], s12, 0xe8c7b756);
        FF(c, d, a, b, x[2], s13, 0x242070db);
        FF(b, c, d, a, x[3], s14, 0xc1bdceee);
        FF(a, b, c, d, x[4], s11, 0xf57c0faf);
        FF(d, a, b, c, x[5], s12, 0x4787c62a);
        FF(c, d, a, b, x[6], s13, 0xa8304613);
        FF(b, c, d, a, x[7], s14, 0xfd469501);
        FF(a, b, c, d, x[8], s11, 0x698098d8);
        FF(d, a, b, c, x[9], s12, 0x8b44f7af);
        FF(c, d, a, b, x[10], s13, 0xffff5bb1);
        FF(b, c, d, a, x[11], s14, 0x895cd7be);
        FF(a, b, c, d, x[12], s11, 0x6b901122);
        FF(d, a, b, c, x[13], s12, 0xfd987193);
        FF(c, d, a, b, x[14], s13, 0xa679438e);
        FF(b, c, d, a, x[15], s14, 0x49b40821);

        /* Round 2 */
        GG(a, b, c, d, x[1], s21, 0xf61e2562);
        GG(d, a, b, c, x[6], s22, 0xc040b340);
        GG(c, d, a, b, x[11], s23, 0x265e5a51);
        GG(b, c, d, a, x[0], s24, 0xe9b6c7aa);
        GG(a, b, c, d, x[5], s21, 0xd62f105d);
        GG(d, a, b, c, x[10], s22, 0x2441453);
        GG(c, d, a, b, x[15], s23, 0xd8a1e681);
        GG(b, c, d, a, x[4], s24, 0xe7d3fbc8);
        GG(a, b, c, d, x[9], s21, 0x21e1cde6);
        GG(d, a, b, c, x[14], s22, 0xc33707d6);
        GG(c, d, a, b, x[3], s23, 0xf4d50d87);
        GG(b, c, d, a, x[8], s24, 0x455a14ed);
        GG(a, b, c, d, x[13], s21, 0xa9e3e905);
        GG(d, a, b, c, x[2], s22, 0xfcefa3f8);
        GG(c, d, a, b, x[7], s23, 0x676f02d9);
        GG(b, c, d, a, x[12], s24, 0x8d2a4c8a);

        /* Round 3 */
        HH(a, b, c, d, x[5], s31, 0xfffa3942);
        HH(d, a, b, c, x[8], s32, 0x8771f681);
        HH(c, d, a, b, x[11], s33, 0x6d9d6122);
        HH(b, c, d, a, x[14], s34, 0xfde5380c);
        HH(a, b, c, d, x[1], s31, 0xa4beea44);
        HH(d, a, b, c, x[4], s32, 0x4bdecfa9);
        HH(c, d, a, b, x[7], s33, 0xf6bb4b60);
        HH(b, c, d, a, x[10], s34, 0xbebfbc70);
        HH(a, b, c, d, x[13], s31, 0x289b7ec6);
        HH(d, a, b, c, x[0], s32, 0xeaa127fa);
        HH(c, d, a, b, x[3], s33, 0xd4ef3085);
        HH(b, c, d, a, x[6], s34, 0x4881d05);
        HH(a, b, c, d, x[9], s31, 0xd9d4d039);
        HH(d, a, b, c, x[12], s32, 0xe6db99e5);
        HH(c, d, a, b, x[15], s33, 0x1fa27cf8);
        HH(b, c, d, a, x[2], s34, 0xc4ac5665);

        /* Round 4 */
        II(a, b, c, d, x[0], s41, 0xf4292244);
        II(d, a, b, c, x[7], s42, 0x432aff97);
        II(c, d, a, b, x[14], s43, 0xab9423a7);
        II(b, c, d, a, x[5], s44, 0xfc93a039);
        II(a, b, c, d, x[12], s41, 0x655b59c3);
        II(d, a, b, c, x[3], s42, 0x8f0ccc92);
        II(c, d, a, b, x[10], s43, 0xffeff47d);
        II(b, c, d, a, x[1], s44, 0x85845dd1);
        II(a, b, c, d, x[8], s41, 0x6fa87e4f);
        II(d, a, b, c, x[15], s42, 0xfe2ce6e0);
        II(c, d, a, b, x[6], s43, 0xa3014314);
        II(b, c, d, a, x[13], s44, 0x4e0811a1);
        II(a, b, c, d, x[4], s41, 0xf7537e82);
        II(d, a, b, c, x[11], s42, 0xbd3af235);
        II(c, d, a, b, x[2], s43, 0x2ad7d2bb);
        II(b, c, d, a, x[9], s44, 0xeb86d391);

        state[0] += a;
        state[1] += b;
        state[2] += c;
        state[3] += d;
    }

    // 下面的处理，在理解上较为复杂
    for (int i = 0; i < 4; i++)
    {
        uint32_t value = state[i];
        state[i] = ((value & 0xff) << 24) |        // 将最低字节移到最高位
                   ((value & 0xff00) << 8) |       // 将次低字节左移
                   ((value & 0xff0000) >> 8) |     // 将次高字节右移
                   ((value & 0xff000000) >> 24);   // 将最高字节移到最低位
    }

    // 输出最终的hash结果
    // for (int i1 = 0; i1 < 4; i1 += 1)
    // {
    //     cout << std::setw(8) << std::setfill('0') << hex << state[i1];
    // }
    // cout << endl;

    // 释放动态分配的内存
    // 实现SIMD并行算法的时候，也请记得及时回收内存！
    delete[] paddedMessage;
    delete[] messageLength;
}
volatile bit32 md5_simd_sink = 0;

static inline uint32x4_t F_neon(uint32x4_t x, uint32x4_t y, uint32x4_t z)
{
    return vorrq_u32(vandq_u32(x, y), vandq_u32(vmvnq_u32(x), z));
}

static inline uint32x4_t G_neon(uint32x4_t x, uint32x4_t y, uint32x4_t z)
{
    return vorrq_u32(vandq_u32(x, z), vandq_u32(y, vmvnq_u32(z)));
}

static inline uint32x4_t H_neon(uint32x4_t x, uint32x4_t y, uint32x4_t z)
{
    return veorq_u32(veorq_u32(x, y), z);
}

static inline uint32x4_t I_neon(uint32x4_t x, uint32x4_t y, uint32x4_t z)
{
    return veorq_u32(y, vorrq_u32(x, vmvnq_u32(z)));
}

static inline uint32x4_t ROTATELEFT_neon(uint32x4_t x, int n)
{
    // vshlq_u32: 正数左移，负数右移
    uint32x4_t left = vshlq_u32(x, vdupq_n_s32(n));
    uint32x4_t right = vshlq_u32(x, vdupq_n_s32(n - 32));
    return vorrq_u32(left, right);
}

static inline void FF_neon(uint32x4_t &a, uint32x4_t b, uint32x4_t c, uint32x4_t d,
                           uint32x4_t x, int s, bit32 ac)
{
    a = vaddq_u32(a, F_neon(b, c, d));
    a = vaddq_u32(a, x);
    a = vaddq_u32(a, vdupq_n_u32(ac));
    a = ROTATELEFT_neon(a, s);
    a = vaddq_u32(a, b);
}

static inline void GG_neon(uint32x4_t &a, uint32x4_t b, uint32x4_t c, uint32x4_t d,
                           uint32x4_t x, int s, bit32 ac)
{
    a = vaddq_u32(a, G_neon(b, c, d));
    a = vaddq_u32(a, x);
    a = vaddq_u32(a, vdupq_n_u32(ac));
    a = ROTATELEFT_neon(a, s);
    a = vaddq_u32(a, b);
}

static inline void HH_neon(uint32x4_t &a, uint32x4_t b, uint32x4_t c, uint32x4_t d,
                           uint32x4_t x, int s, bit32 ac)
{
    a = vaddq_u32(a, H_neon(b, c, d));
    a = vaddq_u32(a, x);
    a = vaddq_u32(a, vdupq_n_u32(ac));
    a = ROTATELEFT_neon(a, s);
    a = vaddq_u32(a, b);
}

static inline void II_neon(uint32x4_t &a, uint32x4_t b, uint32x4_t c, uint32x4_t d,
                           uint32x4_t x, int s, bit32 ac)
{
    a = vaddq_u32(a, I_neon(b, c, d));
    a = vaddq_u32(a, x);
    a = vaddq_u32(a, vdupq_n_u32(ac));
    a = ROTATELEFT_neon(a, s);
    a = vaddq_u32(a, b);
}

static inline bit32 bswap32_local(bit32 value)
{
    return ((value & 0xff) << 24) |
           ((value & 0xff00) << 8) |
           ((value & 0xff0000) >> 8) |
           ((value & 0xff000000) >> 24);
}

// 处理4个短口令，每个口令长度必须 <= 55
static void MD5Hash4_NEON(const string& s0, const string& s1,
                          const string& s2, const string& s3,
                          bit32 output[4][4] = nullptr)
{
    const string* ss[4] = {&s0, &s1, &s2, &s3};

    // words[lane][word_index]
    bit32 words[4][16];

    for (int lane = 0; lane < 4; lane++)
    {
        for (int j = 0; j < 16; j++)
        {
            words[lane][j] = 0;
        }

        const string& input = *ss[lane];
        int len = input.length();

        // 拷贝原始字符串
        for (int i = 0; i < len; i++)
        {
            int word_id = i / 4;
            int shift = (i % 4) * 8;
            words[lane][word_id] |= ((bit32)(unsigned char)input[i]) << shift;
        }

        // 添加 0x80
        int word_id = len / 4;
        int shift = (len % 4) * 8;
        words[lane][word_id] |= ((bit32)0x80) << shift;

        // 长度，单位是bit。短口令只需要一个512-bit block
        words[lane][14] = (bit32)(len * 8);
        words[lane][15] = 0;
    }

    uint32x4_t x[16];

    for (int j = 0; j < 16; j++)
    {
        x[j] = vdupq_n_u32(0);
        x[j] = vsetq_lane_u32(words[0][j], x[j], 0);
        x[j] = vsetq_lane_u32(words[1][j], x[j], 1);
        x[j] = vsetq_lane_u32(words[2][j], x[j], 2);
        x[j] = vsetq_lane_u32(words[3][j], x[j], 3);
    }

    uint32x4_t A0 = vdupq_n_u32(0x67452301);
    uint32x4_t B0 = vdupq_n_u32(0xefcdab89);
    uint32x4_t C0 = vdupq_n_u32(0x98badcfe);
    uint32x4_t D0 = vdupq_n_u32(0x10325476);

    uint32x4_t a = A0;
    uint32x4_t b = B0;
    uint32x4_t c = C0;
    uint32x4_t d = D0;

    // Round 1
    FF_neon(a, b, c, d, x[0],  s11, 0xd76aa478);
    FF_neon(d, a, b, c, x[1],  s12, 0xe8c7b756);
    FF_neon(c, d, a, b, x[2],  s13, 0x242070db);
    FF_neon(b, c, d, a, x[3],  s14, 0xc1bdceee);
    FF_neon(a, b, c, d, x[4],  s11, 0xf57c0faf);
    FF_neon(d, a, b, c, x[5],  s12, 0x4787c62a);
    FF_neon(c, d, a, b, x[6],  s13, 0xa8304613);
    FF_neon(b, c, d, a, x[7],  s14, 0xfd469501);
    FF_neon(a, b, c, d, x[8],  s11, 0x698098d8);
    FF_neon(d, a, b, c, x[9],  s12, 0x8b44f7af);
    FF_neon(c, d, a, b, x[10], s13, 0xffff5bb1);
    FF_neon(b, c, d, a, x[11], s14, 0x895cd7be);
    FF_neon(a, b, c, d, x[12], s11, 0x6b901122);
    FF_neon(d, a, b, c, x[13], s12, 0xfd987193);
    FF_neon(c, d, a, b, x[14], s13, 0xa679438e);
    FF_neon(b, c, d, a, x[15], s14, 0x49b40821);

    // Round 2
    GG_neon(a, b, c, d, x[1],  s21, 0xf61e2562);
    GG_neon(d, a, b, c, x[6],  s22, 0xc040b340);
    GG_neon(c, d, a, b, x[11], s23, 0x265e5a51);
    GG_neon(b, c, d, a, x[0],  s24, 0xe9b6c7aa);
    GG_neon(a, b, c, d, x[5],  s21, 0xd62f105d);
    GG_neon(d, a, b, c, x[10], s22, 0x02441453);
    GG_neon(c, d, a, b, x[15], s23, 0xd8a1e681);
    GG_neon(b, c, d, a, x[4],  s24, 0xe7d3fbc8);
    GG_neon(a, b, c, d, x[9],  s21, 0x21e1cde6);
    GG_neon(d, a, b, c, x[14], s22, 0xc33707d6);
    GG_neon(c, d, a, b, x[3],  s23, 0xf4d50d87);
    GG_neon(b, c, d, a, x[8],  s24, 0x455a14ed);
    GG_neon(a, b, c, d, x[13], s21, 0xa9e3e905);
    GG_neon(d, a, b, c, x[2],  s22, 0xfcefa3f8);
    GG_neon(c, d, a, b, x[7],  s23, 0x676f02d9);
    GG_neon(b, c, d, a, x[12], s24, 0x8d2a4c8a);

    // Round 3
    HH_neon(a, b, c, d, x[5],  s31, 0xfffa3942);
    HH_neon(d, a, b, c, x[8],  s32, 0x8771f681);
    HH_neon(c, d, a, b, x[11], s33, 0x6d9d6122);
    HH_neon(b, c, d, a, x[14], s34, 0xfde5380c);
    HH_neon(a, b, c, d, x[1],  s31, 0xa4beea44);
    HH_neon(d, a, b, c, x[4],  s32, 0x4bdecfa9);
    HH_neon(c, d, a, b, x[7],  s33, 0xf6bb4b60);
    HH_neon(b, c, d, a, x[10], s34, 0xbebfbc70);
    HH_neon(a, b, c, d, x[13], s31, 0x289b7ec6);
    HH_neon(d, a, b, c, x[0],  s32, 0xeaa127fa);
    HH_neon(c, d, a, b, x[3],  s33, 0xd4ef3085);
    HH_neon(b, c, d, a, x[6],  s34, 0x04881d05);
    HH_neon(a, b, c, d, x[9],  s31, 0xd9d4d039);
    HH_neon(d, a, b, c, x[12], s32, 0xe6db99e5);
    HH_neon(c, d, a, b, x[15], s33, 0x1fa27cf8);
    HH_neon(b, c, d, a, x[2],  s34, 0xc4ac5665);

    // Round 4
    II_neon(a, b, c, d, x[0],  s41, 0xf4292244);
    II_neon(d, a, b, c, x[7],  s42, 0x432aff97);
    II_neon(c, d, a, b, x[14], s43, 0xab9423a7);
    II_neon(b, c, d, a, x[5],  s44, 0xfc93a039);
    II_neon(a, b, c, d, x[12], s41, 0x655b59c3);
    II_neon(d, a, b, c, x[3],  s42, 0x8f0ccc92);
    II_neon(c, d, a, b, x[10], s43, 0xffeff47d);
    II_neon(b, c, d, a, x[1],  s44, 0x85845dd1);
    II_neon(a, b, c, d, x[8],  s41, 0x6fa87e4f);
    II_neon(d, a, b, c, x[15], s42, 0xfe2ce6e0);
    II_neon(c, d, a, b, x[6],  s43, 0xa3014314);
    II_neon(b, c, d, a, x[13], s44, 0x4e0811a1);
    II_neon(a, b, c, d, x[4],  s41, 0xf7537e82);
    II_neon(d, a, b, c, x[11], s42, 0xbd3af235);
    II_neon(c, d, a, b, x[2],  s43, 0x2ad7d2bb);
    II_neon(b, c, d, a, x[9],  s44, 0xeb86d391);

    a = vaddq_u32(a, A0);
    b = vaddq_u32(b, B0);
    c = vaddq_u32(c, C0);
    d = vaddq_u32(d, D0);

    bit32 out_a[4], out_b[4], out_c[4], out_d[4];
    vst1q_u32(out_a, a);
    vst1q_u32(out_b, b);
    vst1q_u32(out_c, c);
    vst1q_u32(out_d, d);

    // 和串行版保持一致，最后进行字节序翻转
    for (int i = 0; i < 4; i++)
    {
        bit32 state0 = bswap32_local(out_a[i]);
        bit32 state1 = bswap32_local(out_b[i]);
        bit32 state2 = bswap32_local(out_c[i]);
        bit32 state3 = bswap32_local(out_d[i]);

        if (output != nullptr)
        {
            output[i][0] = state0;
            output[i][1] = state1;
            output[i][2] = state2;
            output[i][3] = state3;
        }

        md5_simd_sink ^= state0;
        md5_simd_sink ^= state1;
        md5_simd_sink ^= state2;
        md5_simd_sink ^= state3;
    }
}

static string MD5StateToHex(const bit32 state[4])
{
    stringstream ss;
    for (int i = 0; i < 4; i++)
    {
        ss << setw(8) << setfill('0') << hex << state[i];
    }
    return ss.str();
}

static void MD5HashBatchSIMDStates(const vector<string>& passwords,
                                   vector<vector<bit32> >& hashes)
{
    hashes.assign(passwords.size(), vector<bit32>(4, 0));

    size_t i = 0;
    bit32 state[4];
    bit32 simd_states[4][4];

    for (; i + 3 < passwords.size(); i += 4)
    {
        if (passwords[i].length() <= 55 &&
            passwords[i + 1].length() <= 55 &&
            passwords[i + 2].length() <= 55 &&
            passwords[i + 3].length() <= 55)
        {
            MD5Hash4_NEON(passwords[i], passwords[i + 1], passwords[i + 2], passwords[i + 3], simd_states);
            for (int lane = 0; lane < 4; lane++)
            {
                for (int part = 0; part < 4; part++)
                {
                    hashes[i + lane][part] = simd_states[lane][part];
                }
            }
        }
        else
        {
            for (int lane = 0; lane < 4; lane++)
            {
                MD5Hash(passwords[i + lane], state);
                for (int part = 0; part < 4; part++)
                {
                    hashes[i + lane][part] = state[part];
                    md5_simd_sink ^= state[part];
                }
            }
        }
    }

    for (; i < passwords.size(); i++)
    {
        MD5Hash(passwords[i], state);
        for (int part = 0; part < 4; part++)
        {
            hashes[i][part] = state[part];
            md5_simd_sink ^= state[part];
        }
    }
}

bool TestMD5HashSIMD()
{
    cout << "Testing parallel MD5HashBatchSIMD correctness..." << endl;

    vector<string> test_pws;
    test_pws.push_back("");
    test_pws.push_back("a");
    test_pws.push_back("abc");
    test_pws.push_back("123456");
    test_pws.push_back("password");
    test_pws.push_back("12345678");
    test_pws.push_back("qwerty");
    test_pws.push_back("123456789");
    test_pws.push_back("12345");
    test_pws.push_back("1234");
    test_pws.push_back("111111");
    test_pws.push_back("0123456789012345678901234567890123456789012345678901234");
    test_pws.push_back("01234567890123456789012345678901234567890123456789012345");

    vector<vector<bit32> > simd_hashes;
    MD5HashBatchSIMDStates(test_pws, simd_hashes);

    for (size_t i = 0; i < test_pws.size(); i++)
    {
        bit32 serial_state[4];
        bit32 simd_state[4];

        MD5Hash(test_pws[i], serial_state);
        for (int part = 0; part < 4; part++)
        {
            simd_state[part] = simd_hashes[i][part];
        }

        string serial_hex = MD5StateToHex(serial_state);
        string simd_hex = MD5StateToHex(simd_state);

        if (serial_hex != simd_hex)
        {
            cout << "[SIMD correctness] FAIL at case " << i << endl;
            cout << "Password: " << test_pws[i] << endl;
            cout << "Serial:   " << serial_hex << endl;
            cout << "Parallel: " << simd_hex << endl;
            return false;
        }
    }

    cout << "[SIMD correctness] Reference: serial MD5Hash" << endl;
    cout << "[SIMD correctness] Checked passwords: " << test_pws.size() << endl;
    cout << "[SIMD correctness] Result: PASS, parallel output matches serial output." << endl;
    return true;
}

void MD5HashBatchSIMD(const vector<string>& passwords)
{
    size_t i = 0;
    bit32 state[4];

    for (; i + 3 < passwords.size(); i += 4)
    {
        // SIMD版本只处理单block短口令
        // MD5中长度 <= 55 字节时，padding后只需要一个512-bit block
        if (passwords[i].length() <= 55 &&
            passwords[i + 1].length() <= 55 &&
            passwords[i + 2].length() <= 55 &&
            passwords[i + 3].length() <= 55)
        {
            MD5Hash4_NEON(passwords[i], passwords[i + 1], passwords[i + 2], passwords[i + 3]);
        }
        else
        {
            // 长口令回退到原串行版本，保证正确性
            MD5Hash(passwords[i], state);
            md5_simd_sink ^= state[0];

            MD5Hash(passwords[i + 1], state);
            md5_simd_sink ^= state[0];

            MD5Hash(passwords[i + 2], state);
            md5_simd_sink ^= state[0];

            MD5Hash(passwords[i + 3], state);
            md5_simd_sink ^= state[0];
        }
    }

    // 处理剩余不足4个的口令
    for (; i < passwords.size(); i++)
    {
        MD5Hash(passwords[i], state);
        md5_simd_sink ^= state[0];
    }
}
