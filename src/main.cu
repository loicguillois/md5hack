/*
 * main.cu — MD5 bruteforce cracker using CUDA
 *
 * ═══════════════════════════════════════════════════════════════════════
 * OVERVIEW
 * ═══════════════════════════════════════════════════════════════════════
 *
 * This program cracks MD5 hashes by trying every possible password of a
 * given length using GPU parallelism. It's a classic "bruteforce attack":
 * generate all candidates, hash each one, and compare with the target.
 *
 * What makes this fast is CUDA: instead of testing candidates one by one
 * on a CPU, we launch ~2 million GPU threads simultaneously. Each thread
 * independently generates and tests one candidate — no synchronization,
 * no shared data, no communication between threads (embarrassingly
 * parallel).
 *
 * ═══════════════════════════════════════════════════════════════════════
 * PROGRAM FLOW
 * ═══════════════════════════════════════════════════════════════════════
 *
 *   1. Parse command line: extract target hash and password length
 *   2. Convert hex hash string to 4 × 32-bit integers
 *   3. Copy target hash to GPU constant memory
 *   4. Allocate GPU memory for result buffer and "found" flag
 *   5. Loop: launch kernel batches of ~2M threads
 *      - Each thread: generate candidate → hash → compare
 *      - After each batch: check if any thread found the password
 *      - Display progress (speed, percentage)
 *   6. If found: copy result back from GPU, print it
 *
 * ═══════════════════════════════════════════════════════════════════════
 * CUDA EXECUTION MODEL — HOW THREADS ARE ORGANIZED
 * ═══════════════════════════════════════════════════════════════════════
 *
 * CUDA organizes threads in a hierarchy:
 *
 *   Grid (entire launch)
 *     └─ Block 0 ─── Thread 0, Thread 1, ..., Thread 255
 *     └─ Block 1 ─── Thread 0, Thread 1, ..., Thread 255
 *     └─ Block 2 ─── Thread 0, Thread 1, ..., Thread 255
 *     └─ ...
 *     └─ Block 8191 ─ Thread 0, Thread 1, ..., Thread 255
 *
 * Total threads per launch = 8192 blocks × 256 threads = 2,097,152
 *
 * Each thread gets a unique global index:
 *   idx = blockIdx.x × blockDim.x + threadIdx.x + batch_offset
 *
 * This index is then converted to a password candidate (see charset.cuh).
 *
 * Usage: ./md5hack <md5_hash> <password_length>
 *
 * Example:
 *   ./md5hack 202cb962ac59075b964b07152d234b70 3    -> finds "123"
 *   ./md5hack a906449d5769fa7361d7ecc6aa3f6d28 6    -> finds "123abc"
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <ctime>

#include "md5_gpu.cuh"
#include "charset.cuh"

/* ═══════════════════════════════════════════════════════════════════════
 * KERNEL CONFIGURATION — TUNING FOR GTX 1080 (PASCAL, SM 6.1)
 * ═══════════════════════════════════════════════════════════════════════
 *
 * THREADS_PER_BLOCK = 256
 *   Each Streaming Multiprocessor (SM) can handle up to 2048 concurrent
 *   threads. With 256 threads per block, we get 8 blocks per SM, which
 *   is within the hardware limit (32 blocks/SM on Pascal). This keeps
 *   the register file well-utilized without spilling to slow local mem.
 *
 * BLOCKS_PER_GRID = 8192
 *   The GTX 1080 has 20 SMs. With 8192 blocks, we have 409 blocks per
 *   SM in the queue — far more than can run concurrently, but the GPU
 *   scheduler will pipeline them efficiently. More blocks = better
 *   latency hiding (while one block waits for memory, another runs).
 *
 * BATCH_SIZE = 256 × 8192 = 2,097,152 candidates per kernel launch
 *   After each batch, we check if a thread found the password. Larger
 *   batches mean fewer host-device synchronizations but slower response
 *   time. 2M is a good balance: checking every ~1ms at 1.5 GH/s.
 */
#define THREADS_PER_BLOCK 256
#define BLOCKS_PER_GRID   8192
#define BATCH_SIZE        ((uint64_t)THREADS_PER_BLOCK * BLOCKS_PER_GRID)

/*
 * Target hash in GPU constant memory.
 *
 * __constant__ memory is a special 64 KB read-only cache on each SM.
 * When all 32 threads in a warp read the same address, the value is
 * broadcast to all of them in a single cycle — as fast as a register.
 *
 * This is perfect for the target hash: every thread reads the same
 * 4 words, so constant memory gives us register-speed access for free.
 */
__constant__ uint32_t d_target[4];

/* ═══════════════════════════════════════════════════════════════════════
 * BRUTEFORCE KERNEL — THE GPU FUNCTION THAT RUNS ON EVERY THREAD
 * ═══════════════════════════════════════════════════════════════════════
 *
 * __global__ marks this as a "kernel" — a function launched from the CPU
 * (host) but executed on the GPU (device) by thousands of threads.
 *
 * Parameters (all in GPU memory):
 *   offset : base index for this batch (each batch covers BATCH_SIZE
 *            candidates; the first batch has offset=0, the second
 *            offset=BATCH_SIZE, etc.)
 *   length : password length to generate
 *   result : GPU buffer where the found password will be written
 *   found  : atomic flag — set to 1 when a thread finds the password
 *
 * Thread lifecycle:
 *   1. Check if already found (early exit to save GPU cycles)
 *   2. Compute unique index from CUDA thread/block IDs + batch offset
 *   3. Convert index to password string (base-62 encoding)
 *   4. Compute MD5 hash (64 rounds, all in registers)
 *   5. Compare 4 words of hash with target (constant memory)
 *   6. On match: atomically claim the "found" flag and write result
 */
__global__ void bruteforce_kernel(uint64_t offset,
                                  int length,
                                  char *result,
                                  int *found) {

    /*
     * Early exit optimization: if another thread in a previous warp
     * already found the password, skip all computation.
     *
     * This read goes to global memory, but it's cached in L1/L2 and
     * only costs ~100 cycles vs ~1000 cycles for the full MD5. Worth it
     * since once found, all remaining threads can bail out immediately.
     */
    if (*found) return;

    /*
     * Compute this thread's unique global index.
     *
     * CUDA provides built-in variables:
     *   blockIdx.x  : which block this thread belongs to (0–8191)
     *   blockDim.x  : threads per block (256)
     *   threadIdx.x : thread's position within its block (0–255)
     *
     * Combined: idx = block_number × block_size + thread_position + offset
     * This gives each thread a unique candidate to test.
     */
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x + offset;

    /*
     * Generate the password candidate.
     * The index is treated as a number in base 62 and each "digit"
     * maps to a character in the charset (see charset.cuh).
     */
    uint8_t password[16];
    index_to_password(idx, length, password);

    /*
     * Compute MD5 hash of the candidate password.
     * This is the hot inner loop — 64 rounds of mixing, all in registers.
     * No memory access, no branches, no divergence between threads.
     * See md5_gpu.cuh for the full implementation.
     */
    uint32_t hash[4];
    md5_hash(password, (uint32_t)length, hash);

    /*
     * Compare with target hash (4 × 32-bit word comparison).
     * d_target lives in constant memory — broadcast to all threads
     * in the warp in a single cycle.
     */
    if (hash[0] == d_target[0] && hash[1] == d_target[1] &&
        hash[2] == d_target[2] && hash[3] == d_target[3]) {

        /*
         * Match found! Use atomicExch to claim the "found" flag.
         *
         * atomicExch atomically writes 1 and returns the old value.
         * If old value was 0, we're the first finder → write the result.
         * If old value was 1, another thread got there first → skip.
         *
         * This prevents race conditions when multiple threads find the
         * same password (unlikely but possible with overlapping batches).
         */
        if (atomicExch(found, 1) == 0) {
            for (int i = 0; i < length; i++) {
                result[i] = (char)password[i];
            }
            result[length] = '\0';
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * HOST UTILITIES — FUNCTIONS THAT RUN ON THE CPU
 * ═══════════════════════════════════════════════════════════════════════ */

/*
 * hex_to_hash — Convert a 32-character hex string to 4 × uint32_t
 *
 * MD5 digests are conventionally printed as 32 hex characters (128 bits),
 * but internally stored as 4 little-endian 32-bit words.
 *
 * "Little-endian" means the least significant byte comes first:
 *
 *   Hex string:  "48 2c 81 1d  a5 d5 b4 bc  6d 49 7f fa  98 49 1e 38"
 *   Word 0:      bytes 0–3 → 0x1d812c48 (reversed: 48,2c,81,1d → 1d,81,2c,48)
 *   Word 1:      bytes 4–7 → 0xbcb4d5a5
 *   Word 2:      bytes 8–11 → 0xfa7f496d
 *   Word 3:      bytes 12–15 → 0x381e4998
 *
 * Returns 0 on success, -1 on invalid input.
 */
static int hex_to_hash(const char *hex, uint32_t out[4]) {
    if (strlen(hex) != 32) return -1;

    for (int i = 0; i < 4; i++) {
        out[i] = 0;
        for (int j = 0; j < 4; j++) {
            unsigned int byte_val;
            /* Extract 2 hex characters → 1 byte */
            char tmp[3] = { hex[i * 8 + j * 2], hex[i * 8 + j * 2 + 1], '\0' };
            if (sscanf(tmp, "%02x", &byte_val) != 1) return -1;
            /* Pack into the word at the correct byte position (LE order) */
            out[i] |= byte_val << (j * 8);
        }
    }
    return 0;
}

/*
 * compute_keyspace — Total number of candidates for a given length
 *
 * With charset size N = 62 (a-z, A-Z, 0-9) and password length L:
 *   keyspace = 62^L
 *
 *   Length 3:  238,328 candidates            (instant)
 *   Length 4:  14,776,336                     (instant)
 *   Length 5:  916,132,832                    (< 1 second)
 *   Length 6:  56,800,235,584                 (~38 seconds)
 *   Length 7:  3,521,614,606,208              (~39 minutes)
 *   Length 8:  218,340,105,584,896            (~40 hours)
 *
 * uint64_t can hold up to 62^10 ≈ 8.4 × 10^17 before overflow.
 */
static uint64_t compute_keyspace(int length) {
    uint64_t total = 1;
    for (int i = 0; i < length; i++) {
        total *= CHARSET_SIZE;
    }
    return total;
}

/* ═══════════════════════════════════════════════════════════════════════
 * MAIN — ENTRY POINT
 * ═══════════════════════════════════════════════════════════════════════ */

int main(int argc, char *argv[]) {

    /* ── Argument parsing ────────────────────────────────────────────
     *
     * We expect exactly 2 arguments:
     *   argv[1] = MD5 hash as 32 hex characters
     *   argv[2] = expected password length (1–15)
     */
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

    /* Convert the hex hash string to four 32-bit words for GPU comparison */
    uint32_t target[4];
    if (hex_to_hash(hex_hash, target) != 0) {
        fprintf(stderr, "Error: invalid MD5 hash (expected 32 hex characters)\n");
        return 1;
    }

    /* ── GPU memory setup ────────────────────────────────────────────
     *
     * Three allocations on the GPU:
     *
     * 1. d_target (constant memory) — the hash we're looking for.
     *    Already declared as __constant__ above; we copy to it with
     *    cudaMemcpyToSymbol. This is NOT a normal cudaMemcpy — it
     *    targets the special constant memory cache.
     *
     * 2. d_result (global memory, 16 bytes) — where the winning thread
     *    writes the found password. Only one thread ever writes here.
     *
     * 3. d_found (global memory, 4 bytes) — atomic flag. Initialized
     *    to 0, set to 1 by the first thread that finds a match.
     *    Read by all threads for early exit, and by the host between
     *    batches to know when to stop.
     */
    cudaMemcpyToSymbol(d_target, target, sizeof(target));

    char *d_result;
    int *d_found;
    cudaMalloc(&d_result, 16);
    cudaMalloc(&d_found, sizeof(int));
    cudaMemset(d_found, 0, sizeof(int));

    /* Compute the total number of candidates to test */
    uint64_t keyspace = compute_keyspace(length);

    /* ── Display banner ──────────────────────────────────────────────
     *
     * Use plain ASCII to avoid UTF-8 multi-byte alignment issues
     * in terminal output (box-drawing characters are 3 bytes each
     * in UTF-8 but display as 1 column, which breaks printf padding).
     */
    printf("\n");
    printf("  md5hack - GPU MD5 cracker\n");
    printf("  ==============================================\n");
    printf("  Target  : %s\n", hex_hash);
    printf("  Length  : %d\n", length);
    printf("  Charset : a-z A-Z 0-9 (%d chars)\n", CHARSET_SIZE);
    printf("  Keyspace: %llu candidates\n",
           (unsigned long long)keyspace);
    printf("  ==============================================\n\n");

    /* ── Batch loop ──────────────────────────────────────────────────
     *
     * We can't launch 56 billion threads at once — the GPU processes
     * them in batches. Each batch launches BATCH_SIZE (~2M) threads,
     * then we synchronize and check for results.
     *
     * Timeline:
     *   Host:  launch batch 0 → wait → check → launch batch 1 → wait → ...
     *   GPU:   [====batch 0====]         [====batch 1====]
     *
     * cudaDeviceSynchronize() blocks the CPU until all GPU threads in
     * the current batch have finished. Then we copy back the "found"
     * flag to check if we can stop.
     *
     * We use CLOCK_MONOTONIC for timing — it measures real elapsed time
     * (wall clock) and is not affected by NTP adjustments or system
     * clock changes.
     */
    struct timespec t_start, t_now;
    clock_gettime(CLOCK_MONOTONIC, &t_start);

    int h_found = 0;
    uint64_t tested = 0;

    for (uint64_t offset = 0; offset < keyspace && !h_found; offset += BATCH_SIZE) {

        /*
         * Launch the kernel: <<<blocks, threads_per_block>>>
         *
         * This is CUDA syntax (not standard C++). The triple chevrons
         * tell nvcc to launch a grid of 8192 blocks, each containing
         * 256 threads. All threads execute bruteforce_kernel() with
         * the same arguments but different blockIdx.x and threadIdx.x.
         */
        bruteforce_kernel<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(
            offset, length, d_result, d_found
        );

        /*
         * Wait for all GPU threads to finish this batch.
         * Without this, the host would race ahead and check d_found
         * before the GPU has written it.
         */
        cudaDeviceSynchronize();

        tested = offset + BATCH_SIZE;
        if (tested > keyspace) tested = keyspace;

        /* Copy the "found" flag from GPU to CPU to check if we're done */
        cudaMemcpy(&h_found, d_found, sizeof(int), cudaMemcpyDeviceToHost);

        /*
         * Progress display: overwrite the same terminal line with \r
         * (carriage return). fflush is needed because stdout is
         * line-buffered and \r doesn't trigger a flush.
         */
        clock_gettime(CLOCK_MONOTONIC, &t_now);
        double elapsed = (t_now.tv_sec - t_start.tv_sec)
                       + (t_now.tv_nsec - t_start.tv_nsec) * 1e-9;
        double speed = tested / elapsed;
        double pct = 100.0 * tested / keyspace;

        printf("\r  [%5.1f%%] %llu / %llu  |  %.2f MH/s  |  %.1fs",
               pct,
               (unsigned long long)tested,
               (unsigned long long)keyspace,
               speed / 1e6,
               elapsed);
        fflush(stdout);
    }

    printf("\n\n");

    /* ── Result ──────────────────────────────────────────────────────
     *
     * If a GPU thread found the password, it wrote it to d_result
     * (GPU memory). We copy it back to host memory with cudaMemcpy.
     */
    if (h_found) {
        char h_result[16] = {0};
        cudaMemcpy(h_result, d_result, length, cudaMemcpyDeviceToHost);
        h_result[length] = '\0';

        clock_gettime(CLOCK_MONOTONIC, &t_now);
        double elapsed = (t_now.tv_sec - t_start.tv_sec)
                       + (t_now.tv_nsec - t_start.tv_nsec) * 1e-9;

        printf("  FOUND: \"%s\"\n", h_result);
        printf("  Time : %.3f seconds\n", elapsed);
    } else {
        printf("  Password not found in keyspace.\n");
    }

    /* ── GPU cleanup ─────────────────────────────────────────────────
     *
     * Free GPU memory. Not strictly necessary (the OS reclaims it on
     * exit) but good practice and helps tools like cuda-memcheck
     * detect real leaks.
     */
    cudaFree(d_result);
    cudaFree(d_found);

    return h_found ? 0 : 1;
}
