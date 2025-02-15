/*
 * charset.cuh — Password candidate generation from CUDA thread index
 *
 * ═══════════════════════════════════════════════════════════════════════
 * THE KEY IDEA: INDEX → PASSWORD MAPPING
 * ═══════════════════════════════════════════════════════════════════════
 *
 * In a bruteforce attack, we need to try every possible password. With
 * CUDA, each thread gets a unique integer index. The question is: how
 * do we turn that integer into a password string?
 *
 * The answer: treat the index as a number in base N (where N = charset
 * size) and convert each "digit" to the corresponding character.
 *
 * It works exactly like counting in a different base:
 *
 *   Decimal (base 10):  0, 1, 2, ..., 9, 10, 11, ...
 *   Binary (base 2):    0, 1, 10, 11, 100, 101, ...
 *   Base-62 (our case):  a, b, c, ..., 9, ba, bb, bc, ...
 *
 * Example with a tiny charset "abc" (N=3), password length 2:
 *
 *   Index │ idx%3 │ idx/3%3 │ Password
 *   ──────┼───────┼─────────┼─────────
 *     0   │  0(a) │  0(a)   │  "aa"
 *     1   │  1(b) │  0(a)   │  "ba"
 *     2   │  2(c) │  0(a)   │  "ca"
 *     3   │  0(a) │  1(b)   │  "ab"
 *     4   │  1(b) │  1(b)   │  "bb"
 *     5   │  2(c) │  1(b)   │  "cb"
 *     6   │  0(a) │  2(c)   │  "ac"
 *     7   │  1(b) │  2(c)   │  "bc"
 *     8   │  2(c) │  2(c)   │  "cc"
 *
 *   Total: 3^2 = 9 candidates → covers the entire keyspace.
 *
 * With our real charset of 62 characters and length 6, thread index
 * 1,000,000 produces candidate "qerpa" + one more character — each
 * thread gets a unique, deterministic password with no overlap.
 *
 * ═══════════════════════════════════════════════════════════════════════
 * WHY __constant__ MEMORY FOR THE CHARSET?
 * ═══════════════════════════════════════════════════════════════════════
 *
 * The charset is accessed by every thread but never modified.
 * __constant__ memory is perfect for this:
 *   - Cached in each SM's constant cache (8 KB, very fast)
 *   - When all 32 threads in a warp access the same address, it's
 *     served in a single cycle (broadcast)
 *   - Even when threads access different addresses (like charset[idx%62]),
 *     the constant cache serializes but stays much faster than global mem
 *
 * Alternative: we could hardcode the charset as a local array, but then
 * it would consume 62 bytes of register space PER THREAD, reducing
 * occupancy. Constant memory is the better trade-off.
 */

#ifndef CHARSET_CUH
#define CHARSET_CUH

#include <cstdint>

/* ═══════════════════════════════════════════════════════════════════════
 * CHARACTER SET DEFINITION
 * ═══════════════════════════════════════════════════════════════════════
 *
 * We use alphanumeric characters: a-z (26) + A-Z (26) + 0-9 (10) = 62.
 *
 * Keyspace sizes for different password lengths:
 *   Length 1:  62
 *   Length 2:  3,844
 *   Length 3:  238,328
 *   Length 4:  14,776,336
 *   Length 5:  916,132,832
 *   Length 6:  56,800,235,584          (~57 billion)
 *   Length 7:  3,521,614,606,208       (~3.5 trillion)
 *   Length 8:  218,340,105,584,896     (~218 trillion)
 */
#define CHARSET_SIZE 62

__constant__ char d_charset[CHARSET_SIZE] = {
    /* Lowercase letters (indices 0–25) */
    'a','b','c','d','e','f','g','h','i','j','k','l','m',
    'n','o','p','q','r','s','t','u','v','w','x','y','z',
    /* Uppercase letters (indices 26–51) */
    'A','B','C','D','E','F','G','H','I','J','K','L','M',
    'N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
    /* Digits (indices 52–61) */
    '0','1','2','3','4','5','6','7','8','9'
};

/* ═══════════════════════════════════════════════════════════════════════
 * INDEX-TO-PASSWORD CONVERSION
 * ═══════════════════════════════════════════════════════════════════════
 *
 * __device__      : this function runs on the GPU only
 * __forceinline__ : the compiler MUST inline it at the call site
 *                   (eliminates function call overhead — important since
 *                   this is called billions of times)
 *
 * The algorithm is just repeated division:
 *   - Take idx modulo 62 → index into charset → first character
 *   - Divide idx by 62
 *   - Take modulo 62 again → second character
 *   - Repeat for each position
 *
 * This is O(length) with no memory allocation, no branches in the
 * inner loop (thanks to #pragma unroll), and reads from fast constant
 * memory cache.
 *
 * #pragma unroll tells the CUDA compiler to fully unroll the loop —
 * instead of a loop with a branch, it generates 16 sequential
 * if-guarded blocks. Since 'length' is uniform across all threads in
 * a warp, the if-guards don't cause warp divergence.
 */
__device__ __forceinline__
void index_to_password(uint64_t idx, int length, uint8_t *out) {
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        if (i < length) {
            out[i] = (uint8_t)d_charset[idx % CHARSET_SIZE];
            idx /= CHARSET_SIZE;
        }
    }
}

#endif /* CHARSET_CUH */
