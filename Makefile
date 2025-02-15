# ══════════════════════════════════════════════════════════════════════
# md5hack — GPU-accelerated MD5 bruteforce cracker
# ══════════════════════════════════════════════════════════════════════
#
# Builds two executables:
#   md5hack      — GPU version (CUDA, ~1.5 GH/s on GTX 1080)
#   md5hack_cpu  — CPU version (single-threaded C, ~50 MH/s)
#
# The CPU version uses the exact same algorithm and can be used to
# compare performance and demonstrate the GPU speedup (~30x).
#
# Build:  make           (both GPU and CPU versions)
#         make gpu       (GPU version only)
#         make cpu       (CPU version only)
# Run:    ./md5hack <md5_hash> <length>
#         ./md5hack_cpu <md5_hash> <length>
# Clean:  make clean
# ══════════════════════════════════════════════════════════════════════

# ── GPU compiler (NVIDIA CUDA) ──────────────────────────────────────
NVCC      = /usr/local/cuda/bin/nvcc

# -O3             : maximum host-side optimization
# -arch=sm_61     : target Pascal (GTX 1080, compute capability 6.1)
# --use_fast_math : faster math intrinsics
NVCCFLAGS = -O3 -arch=sm_61 --use_fast_math

# ── CPU compiler (GCC) ──────────────────────────────────────────────
CC        = gcc

# -O3          : maximum optimization (auto-vectorization, inlining)
# -std=c99     : C99 standard (for loop variable declarations, stdint)
# -march=native: optimize for the current CPU (uses AVX2, etc.)
CFLAGS    = -O3 -std=c99 -march=native -Wall

# ── Targets ──────────────────────────────────────────────────────────

GPU_TARGET  = md5hack
GPU_SRC     = src/main.cu

CPU_TARGET  = md5hack_cpu
CPU_SRC     = src/md5hack_cpu.c

# ── Build rules ──────────────────────────────────────────────────────

# Default: build both GPU and CPU versions
all: $(GPU_TARGET) $(CPU_TARGET)

# GPU version only
gpu: $(GPU_TARGET)

# CPU version only
cpu: $(CPU_TARGET)

$(GPU_TARGET): $(GPU_SRC) src/md5_gpu.cuh src/charset.cuh
	$(NVCC) $(NVCCFLAGS) -o $(GPU_TARGET) $(GPU_SRC)

$(CPU_TARGET): $(CPU_SRC)
	$(CC) $(CFLAGS) -o $(CPU_TARGET) $(CPU_SRC)

clean:
	rm -f $(GPU_TARGET) $(CPU_TARGET)

# Quick test: crack the MD5 of "123" with both versions
run: $(GPU_TARGET)
	./$(GPU_TARGET) 202cb962ac59075b964b07152d234b70 3

run-cpu: $(CPU_TARGET)
	./$(CPU_TARGET) 202cb962ac59075b964b07152d234b70 3

.PHONY: all gpu cpu clean run run-cpu
