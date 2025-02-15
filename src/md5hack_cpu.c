/*
 * md5hack_cpu.c — CPU-only MD5 bruteforce cracker (single-threaded)
 *
 * ═══════════════════════════════════════════════════════════════════════
 * PURPOSE
 * ═══════════════════════════════════════════════════════════════════════
 *
 * This is the pure C, CPU-only version of md5hack. It implements the
 * exact same bruteforce algorithm as the CUDA version but runs on a
 * single CPU core. The goal is to compare performance and demonstrate
 * why GPU acceleration matters for hash cracking:
 *
 *   CPU (single core): ~30–60 million MD5/s
 *   GPU (GTX 1080):    ~1,500 million MD5/s  →  ~30x faster
 *
 * The algorithm is identical:
 *   1. Generate all possible passwords of a given length
 *   2. Compute MD5 for each candidate
 *   3. Compare with the target hash
 *   4. Stop when found
 *
 * The only difference is execution: sequential on CPU vs. massively
 * parallel on GPU (2 million threads simultaneously).
 *
 * Usage: ./md5hack_cpu <md5_hash> <password_length>
 *
 * Example:
 *   ./md5hack_cpu 202cb962ac59075b964b07152d234b70 3    → finds "123"
 *   ./md5hack_cpu a906449d5769fa7361d7ecc6aa3f6d28 6    → finds "123abc"
 */

/* Enable POSIX extensions for clock_gettime / CLOCK_MONOTONIC */
#define _POSIX_C_SOURCE 199309L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

/* ═══════════════════════════════════════════════════════════════════════
 * MD5 IMPLEMENTATION (RFC 1321)
 * ═══════════════════════════════════════════════════════════════════════
 *
 * Same algorithm as the CUDA version (md5_gpu.cuh), but using plain C.
 * On CPU there's no __device__, __constant__, or register pressure —
 * the compiler handles register allocation automatically.
 */

/* Left rotate (circular shift) — same as GPU version */
static inline uint32_t rotl(uint32_t x, uint32_t n) {
    return (x << n) | (x >> (32 - n));
}

/* MD5 auxiliary functions — pure bitwise, no branches */
#define F(x, y, z) (((x) & (y)) | (~(x) & (z)))
#define G(x, y, z) (((x) & (z)) | ((y) & ~(z)))
#define H(x, y, z) ((x) ^ (y) ^ (z))
#define I(x, y, z) ((y) ^ ((x) | ~(z)))

/* MD5 round step macro */
#define MD5_STEP(func, a, b, c, d, k, s, t) \
    (a) += func((b), (c), (d)) + (k) + (t); \
    (a) = rotl((a), (s)); \
    (a) += (b);

/* Sine-derived constants T[i] = floor(2^32 * |sin(i+1)|) */
static const uint32_t T[64] = {
    0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee,
    0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501,
    0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be,
    0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821,
    0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa,
    0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8,
    0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed,
    0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a,
    0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c,
    0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70,
    0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05,
    0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665,
    0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039,
    0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
    0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1,
    0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391
};

/*
 * md5_hash — Compute MD5 of a short message (≤ 55 bytes)
 *
 * Identical algorithm to the GPU version:
 *   1. Pad message into a 512-bit block
 *   2. Initialize state (a, b, c, d)
 *   3. Run 64 rounds of mixing
 *   4. Output final state
 */
static void md5_hash(const uint8_t *msg, uint32_t len, uint32_t out[4]) {
    uint32_t M[16];
    int i;

    /* Zero the message block and pack input bytes (little-endian) */
    for (i = 0; i < 16; i++) M[i] = 0;
    for (i = 0; i < (int)len; i++) {
        M[i >> 2] |= (uint32_t)msg[i] << ((i & 3) << 3);
    }

    /* MD5 padding: 0x80 marker + length in bits at position 14 */
    M[len >> 2] |= (uint32_t)0x80 << ((len & 3) << 3);
    M[14] = len << 3;

    /* Initial hash values (RFC 1321) */
    uint32_t a = 0x67452301;
    uint32_t b = 0xefcdab89;
    uint32_t c = 0x98badcfe;
    uint32_t d = 0x10325476;

    /* Round 1: F function */
    MD5_STEP(F, a, b, c, d, M[ 0],  7, T[ 0])
    MD5_STEP(F, d, a, b, c, M[ 1], 12, T[ 1])
    MD5_STEP(F, c, d, a, b, M[ 2], 17, T[ 2])
    MD5_STEP(F, b, c, d, a, M[ 3], 22, T[ 3])
    MD5_STEP(F, a, b, c, d, M[ 4],  7, T[ 4])
    MD5_STEP(F, d, a, b, c, M[ 5], 12, T[ 5])
    MD5_STEP(F, c, d, a, b, M[ 6], 17, T[ 6])
    MD5_STEP(F, b, c, d, a, M[ 7], 22, T[ 7])
    MD5_STEP(F, a, b, c, d, M[ 8],  7, T[ 8])
    MD5_STEP(F, d, a, b, c, M[ 9], 12, T[ 9])
    MD5_STEP(F, c, d, a, b, M[10], 17, T[10])
    MD5_STEP(F, b, c, d, a, M[11], 22, T[11])
    MD5_STEP(F, a, b, c, d, M[12],  7, T[12])
    MD5_STEP(F, d, a, b, c, M[13], 12, T[13])
    MD5_STEP(F, c, d, a, b, M[14], 17, T[14])
    MD5_STEP(F, b, c, d, a, M[15], 22, T[15])

    /* Round 2: G function */
    MD5_STEP(G, a, b, c, d, M[ 1],  5, T[16])
    MD5_STEP(G, d, a, b, c, M[ 6],  9, T[17])
    MD5_STEP(G, c, d, a, b, M[11], 14, T[18])
    MD5_STEP(G, b, c, d, a, M[ 0], 20, T[19])
    MD5_STEP(G, a, b, c, d, M[ 5],  5, T[20])
    MD5_STEP(G, d, a, b, c, M[10],  9, T[21])
    MD5_STEP(G, c, d, a, b, M[15], 14, T[22])
    MD5_STEP(G, b, c, d, a, M[ 4], 20, T[23])
    MD5_STEP(G, a, b, c, d, M[ 9],  5, T[24])
    MD5_STEP(G, d, a, b, c, M[14],  9, T[25])
    MD5_STEP(G, c, d, a, b, M[ 3], 14, T[26])
    MD5_STEP(G, b, c, d, a, M[ 8], 20, T[27])
    MD5_STEP(G, a, b, c, d, M[13],  5, T[28])
    MD5_STEP(G, d, a, b, c, M[ 2],  9, T[29])
    MD5_STEP(G, c, d, a, b, M[ 7], 14, T[30])
    MD5_STEP(G, b, c, d, a, M[12], 20, T[31])

    /* Round 3: H function */
    MD5_STEP(H, a, b, c, d, M[ 5],  4, T[32])
    MD5_STEP(H, d, a, b, c, M[ 8], 11, T[33])
    MD5_STEP(H, c, d, a, b, M[11], 16, T[34])
    MD5_STEP(H, b, c, d, a, M[14], 23, T[35])
    MD5_STEP(H, a, b, c, d, M[ 1],  4, T[36])
    MD5_STEP(H, d, a, b, c, M[ 4], 11, T[37])
    MD5_STEP(H, c, d, a, b, M[ 7], 16, T[38])
    MD5_STEP(H, b, c, d, a, M[10], 23, T[39])
    MD5_STEP(H, a, b, c, d, M[13],  4, T[40])
    MD5_STEP(H, d, a, b, c, M[ 0], 11, T[41])
    MD5_STEP(H, c, d, a, b, M[ 3], 16, T[42])
    MD5_STEP(H, b, c, d, a, M[ 6], 23, T[43])
    MD5_STEP(H, a, b, c, d, M[ 9],  4, T[44])
    MD5_STEP(H, d, a, b, c, M[12], 11, T[45])
    MD5_STEP(H, c, d, a, b, M[15], 16, T[46])
    MD5_STEP(H, b, c, d, a, M[ 2], 23, T[47])

    /* Round 4: I function */
    MD5_STEP(I, a, b, c, d, M[ 0],  6, T[48])
    MD5_STEP(I, d, a, b, c, M[ 7], 10, T[49])
    MD5_STEP(I, c, d, a, b, M[14], 15, T[50])
    MD5_STEP(I, b, c, d, a, M[ 5], 21, T[51])
    MD5_STEP(I, a, b, c, d, M[12],  6, T[52])
    MD5_STEP(I, d, a, b, c, M[ 3], 10, T[53])
    MD5_STEP(I, c, d, a, b, M[10], 15, T[54])
    MD5_STEP(I, b, c, d, a, M[ 1], 21, T[55])
    MD5_STEP(I, a, b, c, d, M[ 8],  6, T[56])
    MD5_STEP(I, d, a, b, c, M[15], 10, T[57])
    MD5_STEP(I, c, d, a, b, M[ 6], 15, T[58])
    MD5_STEP(I, b, c, d, a, M[13], 21, T[59])
    MD5_STEP(I, a, b, c, d, M[ 4],  6, T[60])
    MD5_STEP(I, d, a, b, c, M[11], 10, T[61])
    MD5_STEP(I, c, d, a, b, M[ 2], 15, T[62])
    MD5_STEP(I, b, c, d, a, M[ 9], 21, T[63])

    /* Final addition */
    out[0] = a + 0x67452301;
    out[1] = b + 0xefcdab89;
    out[2] = c + 0x98badcfe;
    out[3] = d + 0x10325476;
}

/* ═══════════════════════════════════════════════════════════════════════
 * CHARACTER SET AND CANDIDATE GENERATION
 * ═══════════════════════════════════════════════════════════════════════
 *
 * Same charset as the CUDA version: a-z A-Z 0-9 (62 characters).
 * Same base-62 conversion from index to password string.
 */

#define CHARSET_SIZE 62

static const char charset[CHARSET_SIZE] = {
    'a','b','c','d','e','f','g','h','i','j','k','l','m',
    'n','o','p','q','r','s','t','u','v','w','x','y','z',
    'A','B','C','D','E','F','G','H','I','J','K','L','M',
    'N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
    '0','1','2','3','4','5','6','7','8','9'
};

/* Convert an index to a password candidate (base-62 encoding) */
static void index_to_password(uint64_t idx, int length, uint8_t *out) {
    for (int i = 0; i < length; i++) {
        out[i] = (uint8_t)charset[idx % CHARSET_SIZE];
        idx /= CHARSET_SIZE;
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * HOST UTILITIES
 * ═══════════════════════════════════════════════════════════════════════ */

/* Parse a 32-character hex string into 4 x uint32_t (little-endian) */
static int hex_to_hash(const char *hex, uint32_t out[4]) {
    if (strlen(hex) != 32) return -1;

    for (int i = 0; i < 4; i++) {
        out[i] = 0;
        for (int j = 0; j < 4; j++) {
            unsigned int byte_val;
            char tmp[3] = { hex[i * 8 + j * 2], hex[i * 8 + j * 2 + 1], '\0' };
            if (sscanf(tmp, "%02x", &byte_val) != 1) return -1;
            out[i] |= byte_val << (j * 8);
        }
    }
    return 0;
}

/* Compute 62^length */
static uint64_t compute_keyspace(int length) {
    uint64_t total = 1;
    for (int i = 0; i < length; i++) {
        total *= CHARSET_SIZE;
    }
    return total;
}

/* ═══════════════════════════════════════════════════════════════════════
 * MAIN — SEQUENTIAL BRUTEFORCE ON CPU
 * ═══════════════════════════════════════════════════════════════════════
 *
 * Unlike the GPU version which launches millions of threads, this
 * version tests candidates one at a time in a simple for loop.
 * The algorithm is identical — only the parallelism is missing.
 *
 * This makes it easy to see the GPU speedup: same code, same results,
 * vastly different execution time.
 */

int main(int argc, char *argv[]) {

    if (argc != 3) {
        fprintf(stderr, "Usage: %s <md5_hash> <length>\n", argv[0]);
        fprintf(stderr, "  md5_hash : 32-character hex MD5 hash to crack\n");
        fprintf(stderr, "  length   : expected password length\n");
        fprintf(stderr, "\nExample:\n");
        fprintf(stderr, "  %s 202cb962ac59075b964b07152d234b70 3\n", argv[0]);
        return 1;
    }

    const char *hex_hash = argv[1];
    int length = atoi(argv[2]);

    if (length < 1 || length > 15) {
        fprintf(stderr, "Error: length must be between 1 and 15\n");
        return 1;
    }

    uint32_t target[4];
    if (hex_to_hash(hex_hash, target) != 0) {
        fprintf(stderr, "Error: invalid MD5 hash (expected 32 hex characters)\n");
        return 1;
    }

    uint64_t keyspace = compute_keyspace(length);

    printf("\n");
    printf("  md5hack_cpu - CPU MD5 cracker (single-threaded)\n");
    printf("  ==============================================\n");
    printf("  Target  : %s\n", hex_hash);
    printf("  Length  : %d\n", length);
    printf("  Charset : a-z A-Z 0-9 (%d chars)\n", CHARSET_SIZE);
    printf("  Keyspace: %llu candidates\n",
           (unsigned long long)keyspace);
    printf("  ==============================================\n\n");

    struct timespec t_start, t_now;
    clock_gettime(CLOCK_MONOTONIC, &t_start);

    /*
     * The main bruteforce loop: try every index from 0 to keyspace-1.
     *
     * On a modern CPU (single core, ~3-4 GHz), this achieves roughly
     * 30-60 million MD5 hashes per second. Compare that to ~1.5 billion
     * on a GTX 1080 — about 30x slower.
     *
     * For a 6-character password (56.8 billion candidates):
     *   CPU: ~56.8B / 50M = ~1136 seconds (~19 minutes)
     *   GPU: ~56.8B / 1.5B = ~38 seconds
     *
     * Progress is displayed every 10 million candidates to avoid
     * slowing down the loop with excessive printf calls.
     */
    uint8_t password[16];
    uint32_t hash[4];
    int found = 0;

    for (uint64_t idx = 0; idx < keyspace; idx++) {

        /* Generate candidate from index (base-62 conversion) */
        index_to_password(idx, length, password);

        /* Compute MD5 */
        md5_hash(password, (uint32_t)length, hash);

        /* Compare with target */
        if (hash[0] == target[0] && hash[1] == target[1] &&
            hash[2] == target[2] && hash[3] == target[3]) {

            /* Found it! */
            clock_gettime(CLOCK_MONOTONIC, &t_now);
            double elapsed = (t_now.tv_sec - t_start.tv_sec)
                           + (t_now.tv_nsec - t_start.tv_nsec) * 1e-9;

            char result[16];
            for (int i = 0; i < length; i++) result[i] = (char)password[i];
            result[length] = '\0';

            printf("\r  [100.0%%] %llu / %llu  |  %.2f MH/s  |  %.1fs\n\n",
                   (unsigned long long)(idx + 1),
                   (unsigned long long)keyspace,
                   (idx + 1) / elapsed / 1e6,
                   elapsed);
            printf("  FOUND: \"%s\"\n", result);
            printf("  Time : %.3f seconds\n", elapsed);
            found = 1;
            break;
        }

        /* Progress display every 10 million candidates */
        if ((idx & 0xFFFFFF) == 0 && idx > 0) {
            clock_gettime(CLOCK_MONOTONIC, &t_now);
            double elapsed = (t_now.tv_sec - t_start.tv_sec)
                           + (t_now.tv_nsec - t_start.tv_nsec) * 1e-9;
            double speed = idx / elapsed;
            double pct = 100.0 * idx / keyspace;

            printf("\r  [%5.1f%%] %llu / %llu  |  %.2f MH/s  |  %.1fs",
                   pct,
                   (unsigned long long)idx,
                   (unsigned long long)keyspace,
                   speed / 1e6,
                   elapsed);
            fflush(stdout);
        }
    }

    if (!found) {
        printf("\n\n  Password not found in keyspace.\n");
    }

    return found ? 0 : 1;
}
