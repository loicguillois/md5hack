/*
 * md5_gpu.cuh — Complete MD5 hash algorithm implemented as CUDA device code
 *
 * ═══════════════════════════════════════════════════════════════════════
 * WHAT IS MD5?
 * ═══════════════════════════════════════════════════════════════════════
 *
 * MD5 (Message-Digest algorithm 5) is a cryptographic hash function that
 * produces a 128-bit (16-byte) fingerprint from arbitrary input data.
 * Defined in RFC 1321 (1992) by Ronald Rivest.
 *
 * Although MD5 is cryptographically broken (collisions can be found in
 * seconds), it is still widely used for checksums and legacy password
 * storage — which is exactly what we're attacking here.
 *
 * ═══════════════════════════════════════════════════════════════════════
 * WHY GPU?
 * ═══════════════════════════════════════════════════════════════════════
 *
 * MD5 is a pure arithmetic/bitwise operation — no memory lookups, no
 * branches, no data dependencies between threads. This makes it a
 * perfect candidate for GPU parallelism:
 *
 *   - A CPU core computes ~50 million MD5/s
 *   - A GTX 1080 (2560 cores) computes ~1.5 billion MD5/s
 *   - That's a 30x speedup from GPU parallelism alone
 *
 * ═══════════════════════════════════════════════════════════════════════
 * GPU MEMORY OPTIMIZATION
 * ═══════════════════════════════════════════════════════════════════════
 *
 * CUDA offers several memory types with different performance:
 *
 *   Global memory  : large but slow (~400 cycles latency)
 *   Shared memory  : fast but limited (48 KB per SM)
 *   Constant memory: fast read-only (broadcast to all threads in a warp)
 *   Registers      : fastest (zero latency, private to each thread)
 *
 * Our MD5 implementation uses:
 *   - Registers for all computation (a, b, c, d state + M[16] message block)
 *   - __constant__ memory for the 64 sine constants (read by all threads)
 *   - Zero global/shared memory access during hashing
 *
 * This is critical because with 2 million threads running simultaneously,
 * any memory access would create massive contention.
 *
 * ═══════════════════════════════════════════════════════════════════════
 * MD5 ALGORITHM OVERVIEW
 * ═══════════════════════════════════════════════════════════════════════
 *
 * 1. PAD the message to a multiple of 512 bits (64 bytes)
 * 2. INITIALIZE four 32-bit state variables (a, b, c, d)
 * 3. PROCESS each 512-bit block through 64 rounds:
 *    - Rounds  0–15: use function F(x,y,z) = (x AND y) OR (NOT x AND z)
 *    - Rounds 16–31: use function G(x,y,z) = (x AND z) OR (y AND NOT z)
 *    - Rounds 32–47: use function H(x,y,z) = x XOR y XOR z
 *    - Rounds 48–63: use function I(x,y,z) = y XOR (x OR NOT z)
 *    Each round mixes one state variable with the function output,
 *    a message word, a sine constant, then rotates left.
 * 4. OUTPUT the final state as 4 concatenated 32-bit words (128 bits)
 *
 * For password bruteforcing, messages are always short (≤ 15 chars),
 * so we only ever need one 512-bit block — no loop required.
 */

#ifndef MD5_GPU_CUH
#define MD5_GPU_CUH

#include <cstdint>

/* ═══════════════════════════════════════════════════════════════════════
 * MD5 SINE-DERIVED CONSTANTS — T table
 * ═══════════════════════════════════════════════════════════════════════
 *
 * T[i] = floor(2^32 × |sin(i + 1)|)    for i = 0..63
 *
 * These constants were chosen by Rivest to provide "nothing up my sleeve"
 * numbers — they come from a well-known mathematical function (sine),
 * so there's no suspicion of a hidden backdoor in their choice.
 *
 * Stored in __constant__ memory: this is a read-only cache that sits
 * in each SM (Streaming Multiprocessor). When all threads in a warp
 * read the same constant address, it's broadcast in a single cycle —
 * as fast as a register read.
 */
__constant__ uint32_t T[64] = {
    /* Round 1 (F function) — T[0..15] */
    0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee,
    0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501,
    0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be,
    0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821,
    /* Round 2 (G function) — T[16..31] */
    0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa,
    0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8,
    0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed,
    0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a,
    /* Round 3 (H function) — T[32..47] */
    0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c,
    0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70,
    0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05,
    0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665,
    /* Round 4 (I function) — T[48..63] */
    0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039,
    0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
    0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1,
    0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391
};

/* ═══════════════════════════════════════════════════════════════════════
 * MD5 AUXILIARY FUNCTIONS
 * ═══════════════════════════════════════════════════════════════════════
 *
 * Each of the 4 rounds uses a different bitwise function to mix the
 * state variables. All four are branch-free (pure bitwise operations),
 * which is ideal for GPU execution — all threads in a warp execute
 * the same instruction with no divergence.
 *
 * __device__      : runs on GPU only
 * __forceinline__ : compiler MUST inline (no function call overhead)
 */

/*
 * F(x, y, z) = (x AND y) OR (NOT x AND z)
 *
 * Acts as a bitwise multiplexer: for each bit position,
 * if x=1 → select y's bit, if x=0 → select z's bit.
 * Used in rounds 0–15 (first pass over the message).
 */
__device__ __forceinline__ uint32_t F(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) | (~x & z);
}

/*
 * G(x, y, z) = (x AND z) OR (y AND NOT z)
 *
 * Same multiplexer idea but with z as the selector instead of x.
 * Used in rounds 16–31.
 */
__device__ __forceinline__ uint32_t G(uint32_t x, uint32_t y, uint32_t z) {
    return (x & z) | (y & ~z);
}

/*
 * H(x, y, z) = x XOR y XOR z
 *
 * Parity function — changes output if any single input bit flips.
 * Provides maximum diffusion (avalanche effect).
 * Used in rounds 32–47.
 */
__device__ __forceinline__ uint32_t H(uint32_t x, uint32_t y, uint32_t z) {
    return x ^ y ^ z;
}

/*
 * I(x, y, z) = y XOR (x OR NOT z)
 *
 * Asymmetric function — harder to invert than F or G.
 * Used in rounds 48–63 (final pass).
 */
__device__ __forceinline__ uint32_t I(uint32_t x, uint32_t y, uint32_t z) {
    return y ^ (x | ~z);
}

/*
 * rotl — Left circular rotation (rotate bits left by n positions)
 *
 * Bits that "fall off" the left end wrap around to the right.
 * Example: rotl(0xF0000001, 4) = 0x0000001F
 *
 * This is a core operation in MD5 — it spreads bit changes across
 * the entire 32-bit word, contributing to the avalanche effect.
 */
__device__ __forceinline__ uint32_t rotl(uint32_t x, uint32_t n) {
    return (x << n) | (x >> (32 - n));
}

/* ═══════════════════════════════════════════════════════════════════════
 * MD5 ROUND STEP MACRO
 * ═══════════════════════════════════════════════════════════════════════
 *
 * Each of the 64 MD5 rounds performs this operation:
 *
 *   a = b + ROTL(a + func(b,c,d) + M[k] + T[i], s)
 *
 * Where:
 *   func   = F, G, H, or I (depending on the round)
 *   a,b,c,d = the four state variables (rotate roles each step)
 *   M[k]   = one of the 16 message words
 *   T[i]   = sine-derived constant for this step
 *   s      = rotation amount (varies per step: 7,12,17,22 for round 1, etc.)
 *
 * We use a macro rather than a function to guarantee zero overhead —
 * the compiler substitutes it inline at each of the 64 call sites.
 */
#define MD5_STEP(func, a, b, c, d, k, s, i) \
    (a) += func((b), (c), (d)) + (k) + T[(i)]; \
    (a) = rotl((a), (s)); \
    (a) += (b);

/* ═══════════════════════════════════════════════════════════════════════
 * md5_hash — Compute the MD5 digest of a short message
 * ═══════════════════════════════════════════════════════════════════════
 *
 * Parameters:
 *   msg  : pointer to the message bytes (password candidate)
 *   len  : message length in bytes (must be ≤ 55)
 *   out  : output array of 4 × uint32_t (128-bit MD5 digest)
 *
 * Limitations:
 *   - Only handles single-block messages (≤ 55 bytes after padding).
 *     This is fine for password cracking where passwords rarely exceed
 *     15 characters.
 *
 * Performance:
 *   - The entire function runs in GPU registers.
 *   - M[16] (message block): 16 × 32-bit registers
 *   - a, b, c, d (state): 4 × 32-bit registers
 *   - Total: ~20 registers per thread. With 65,536 registers per SM
 *     (Pascal), this allows high occupancy.
 */
__device__ void md5_hash(const uint8_t *msg, uint32_t len, uint32_t out[4]) {

    /* ── Step 1: Build the padded message block ─────────────────────
     *
     * MD5 padding rules (RFC 1321 §3.1–3.2):
     *
     *   [message bytes] [0x80] [00 00 ... 00] [length in bits as 64-bit LE]
     *   |← msg_len →|  |←1→|  |← padding →|  |←────── 8 bytes ──────→|
     *   |←──────────────── total: 64 bytes (512 bits) ────────────────→|
     *
     * The 0x80 byte marks the end of the message (it's a "1" bit followed
     * by 7 zero bits). Then we pad with zeros until we reach byte 56,
     * leaving room for the 8-byte length field at the end.
     *
     * Example for "abc" (3 bytes):
     *   Bytes 0–2:  'a' 'b' 'c'
     *   Byte  3:    0x80 (end marker)
     *   Bytes 4–55: 0x00 (zero padding)
     *   Bytes 56–63: 0x18 0x00 ... (24 bits = 3 bytes × 8, little-endian)
     */
    uint32_t M[16];

    /* Zero the entire 64-byte block (16 × 32-bit words) */
    #pragma unroll
    for (int i = 0; i < 16; i++) M[i] = 0;

    /*
     * Pack message bytes into 32-bit words in little-endian order.
     *
     * MD5 uses little-endian byte order within each 32-bit word:
     *   - Byte 0 goes to bits 0–7 of M[0]
     *   - Byte 1 goes to bits 8–15 of M[0]
     *   - Byte 2 goes to bits 16–23 of M[0]
     *   - Byte 3 goes to bits 24–31 of M[0]
     *   - Byte 4 goes to bits 0–7 of M[1]
     *   - ...
     *
     * The expression (i >> 2) selects which word, and
     * ((i & 3) << 3) selects which byte position within the word.
     */
    for (uint32_t i = 0; i < len; i++) {
        M[i >> 2] |= (uint32_t)msg[i] << ((i & 3) << 3);
    }

    /* Append the 0x80 end-of-message marker */
    M[len >> 2] |= (uint32_t)0x80 << ((len & 3) << 3);

    /*
     * Append the original message length in BITS as a 64-bit little-endian
     * integer at the end of the block (words 14 and 15).
     *
     * For a 6-byte message: 6 × 8 = 48 bits → M[14] = 48, M[15] = 0.
     * M[15] stays zero (already cleared) since our messages are short.
     */
    M[14] = len << 3;

    /* ── Step 2: Initialize MD5 state variables ─────────────────────
     *
     * These are the "magic" initial values defined in RFC 1321 §3.3.
     * They are the same for every MD5 computation and were chosen
     * (somewhat arbitrarily) by the algorithm's designer.
     *
     * In little-endian hex:
     *   a = 01 23 45 67    stored as 0x67452301
     *   b = 89 ab cd ef    stored as 0xefcdab89
     *   c = fe dc ba 98    stored as 0x98badcfe
     *   d = 76 54 32 10    stored as 0x10325476
     */
    uint32_t a = 0x67452301;
    uint32_t b = 0xefcdab89;
    uint32_t c = 0x98badcfe;
    uint32_t d = 0x10325476;

    /* ── Step 3: 64 rounds of mixing ────────────────────────────────
     *
     * The 64 rounds are divided into 4 groups of 16, each using a
     * different auxiliary function (F, G, H, I). Within each group,
     * the message words (M[k]) are accessed in a different permuted
     * order, and each step uses a different rotation amount (s).
     *
     * After each step, the state variables rotate roles:
     *   Step n:   a  b  c  d
     *   Step n+1: d  a  b  c   (a becomes the "active" variable)
     *
     * This is unrolled (no loop) for maximum GPU performance.
     */

    /* ── Round 1: F(b,c,d) — linear selection ──────────────────── */
    /* Message words accessed in order: 0,1,2,...,15 (sequential) */
    /* Rotation amounts cycle: 7, 12, 17, 22 */
    MD5_STEP(F, a, b, c, d, M[ 0],  7,  0)
    MD5_STEP(F, d, a, b, c, M[ 1], 12,  1)
    MD5_STEP(F, c, d, a, b, M[ 2], 17,  2)
    MD5_STEP(F, b, c, d, a, M[ 3], 22,  3)
    MD5_STEP(F, a, b, c, d, M[ 4],  7,  4)
    MD5_STEP(F, d, a, b, c, M[ 5], 12,  5)
    MD5_STEP(F, c, d, a, b, M[ 6], 17,  6)
    MD5_STEP(F, b, c, d, a, M[ 7], 22,  7)
    MD5_STEP(F, a, b, c, d, M[ 8],  7,  8)
    MD5_STEP(F, d, a, b, c, M[ 9], 12,  9)
    MD5_STEP(F, c, d, a, b, M[10], 17, 10)
    MD5_STEP(F, b, c, d, a, M[11], 22, 11)
    MD5_STEP(F, a, b, c, d, M[12],  7, 12)
    MD5_STEP(F, d, a, b, c, M[13], 12, 13)
    MD5_STEP(F, c, d, a, b, M[14], 17, 14)
    MD5_STEP(F, b, c, d, a, M[15], 22, 15)

    /* ── Round 2: G(b,c,d) — permuted selection ────────────────── */
    /* Message words: k = (1 + 5*i) mod 16 → 1,6,11,0,5,10,15,4,... */
    /* Rotation amounts cycle: 5, 9, 14, 20 */
    MD5_STEP(G, a, b, c, d, M[ 1],  5, 16)
    MD5_STEP(G, d, a, b, c, M[ 6],  9, 17)
    MD5_STEP(G, c, d, a, b, M[11], 14, 18)
    MD5_STEP(G, b, c, d, a, M[ 0], 20, 19)
    MD5_STEP(G, a, b, c, d, M[ 5],  5, 20)
    MD5_STEP(G, d, a, b, c, M[10],  9, 21)
    MD5_STEP(G, c, d, a, b, M[15], 14, 22)
    MD5_STEP(G, b, c, d, a, M[ 4], 20, 23)
    MD5_STEP(G, a, b, c, d, M[ 9],  5, 24)
    MD5_STEP(G, d, a, b, c, M[14],  9, 25)
    MD5_STEP(G, c, d, a, b, M[ 3], 14, 26)
    MD5_STEP(G, b, c, d, a, M[ 8], 20, 27)
    MD5_STEP(G, a, b, c, d, M[13],  5, 28)
    MD5_STEP(G, d, a, b, c, M[ 2],  9, 29)
    MD5_STEP(G, c, d, a, b, M[ 7], 14, 30)
    MD5_STEP(G, b, c, d, a, M[12], 20, 31)

    /* ── Round 3: H(b,c,d) — parity (XOR) ─────────────────────── */
    /* Message words: k = (5 + 3*i) mod 16 → 5,8,11,14,1,4,7,10,... */
    /* Rotation amounts cycle: 4, 11, 16, 23 */
    MD5_STEP(H, a, b, c, d, M[ 5],  4, 32)
    MD5_STEP(H, d, a, b, c, M[ 8], 11, 33)
    MD5_STEP(H, c, d, a, b, M[11], 16, 34)
    MD5_STEP(H, b, c, d, a, M[14], 23, 35)
    MD5_STEP(H, a, b, c, d, M[ 1],  4, 36)
    MD5_STEP(H, d, a, b, c, M[ 4], 11, 37)
    MD5_STEP(H, c, d, a, b, M[ 7], 16, 38)
    MD5_STEP(H, b, c, d, a, M[10], 23, 39)
    MD5_STEP(H, a, b, c, d, M[13],  4, 40)
    MD5_STEP(H, d, a, b, c, M[ 0], 11, 41)
    MD5_STEP(H, c, d, a, b, M[ 3], 16, 42)
    MD5_STEP(H, b, c, d, a, M[ 6], 23, 43)
    MD5_STEP(H, a, b, c, d, M[ 9],  4, 44)
    MD5_STEP(H, d, a, b, c, M[12], 11, 45)
    MD5_STEP(H, c, d, a, b, M[15], 16, 46)
    MD5_STEP(H, b, c, d, a, M[ 2], 23, 47)

    /* ── Round 4: I(b,c,d) — asymmetric inversion ─────────────── */
    /* Message words: k = (7*i) mod 16 → 0,7,14,5,12,3,10,1,... */
    /* Rotation amounts cycle: 6, 10, 15, 21 */
    MD5_STEP(I, a, b, c, d, M[ 0],  6, 48)
    MD5_STEP(I, d, a, b, c, M[ 7], 10, 49)
    MD5_STEP(I, c, d, a, b, M[14], 15, 50)
    MD5_STEP(I, b, c, d, a, M[ 5], 21, 51)
    MD5_STEP(I, a, b, c, d, M[12],  6, 52)
    MD5_STEP(I, d, a, b, c, M[ 3], 10, 53)
    MD5_STEP(I, c, d, a, b, M[10], 15, 54)
    MD5_STEP(I, b, c, d, a, M[ 1], 21, 55)
    MD5_STEP(I, a, b, c, d, M[ 8],  6, 56)
    MD5_STEP(I, d, a, b, c, M[15], 10, 57)
    MD5_STEP(I, c, d, a, b, M[ 6], 15, 58)
    MD5_STEP(I, b, c, d, a, M[13], 21, 59)
    MD5_STEP(I, a, b, c, d, M[ 4],  6, 60)
    MD5_STEP(I, d, a, b, c, M[11], 10, 61)
    MD5_STEP(I, c, d, a, b, M[ 2], 15, 62)
    MD5_STEP(I, b, c, d, a, M[ 9], 21, 63)

    /* ── Step 4: Produce the final 128-bit hash ─────────────────────
     *
     * Add the initial values back to the state. This is the
     * Merkle–Damgård construction: the hash of each block feeds
     * into the next. Since we only have one block, this is the
     * final output.
     *
     * The result is 4 × 32-bit words in little-endian order:
     *   out[0] = low 32 bits  → first 8 hex characters of the hash
     *   out[1] = next 32 bits → characters 8–15
     *   out[2] = next 32 bits → characters 16–23
     *   out[3] = high 32 bits → characters 24–31
     */
    out[0] = a + 0x67452301;
    out[1] = b + 0xefcdab89;
    out[2] = c + 0x98badcfe;
    out[3] = d + 0x10325476;
}

#endif /* MD5_GPU_CUH */
