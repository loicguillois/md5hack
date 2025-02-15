# md5hack

An educational project to learn CUDA GPU programming and demonstrate why MD5 is dangerously weak for password storage.

## About

This project was built as a hands-on exercise in CUDA development: writing GPU kernels, managing device memory, tuning thread/block configurations, and understanding the massive parallelism of modern GPUs.

It also serves as a practical demonstration of why MD5 should never be used to hash passwords in databases. By bruteforcing MD5 hashes on a consumer GPU, md5hack makes the weakness of MD5 for password storage tangible — a 6-character password cracked in 38 seconds, not in theory, but in practice.

Each of the thousands of GPU threads generates a unique password candidate, computes its MD5 hash entirely in registers (no memory traffic), and compares it against the target. A pure C single-threaded CPU version (`md5hack_cpu`) is included to compare performance and show the GPU speedup.

## CPU vs GPU performance

Both versions implement the exact same algorithm. The only difference is parallelism.

Benchmarked on an Intel Core i7-8700K @ 3.70 GHz (CPU) and NVIDIA GeForce GTX 1080 (GPU).

| | CPU (i7-8700K, 1 core) | GPU (GTX 1080) | Speedup |
|---|---|---|---|
| Speed | ~10 MH/s | ~1,500 MH/s | **~150x** |
| "test5" (5 chars) | 87.2s | 0.6s | 145x |
| "AZE128" (6 chars) | ~95 min (est.) | 37.8s | ~150x |

## Usage

```bash
# GPU version (requires NVIDIA GPU + CUDA)
./md5hack <md5_hash> <password_length>

# CPU version (runs anywhere)
./md5hack_cpu <md5_hash> <password_length>
```

### Examples

```bash
# Crack a 3-digit password
./md5hack 202cb962ac59075b964b07152d234b70 3
# → FOUND: "123" (instant)

# Crack a 6-character alphanumeric password
./md5hack a906449d5769fa7361d7ecc6aa3f6d28 6
# → FOUND: "123abc" (~1.2s on GTX 1080)

# Crack a mixed-case password with digits
./md5hack 45c6666f0cfa1d37778f7a9ad7f428ca 6
# → FOUND: "AZE128" (~38s on GTX 1080, 1.5 GH/s)
```

### Character set

The bruteforce covers `a-z A-Z 0-9` (62 characters), giving 62^L total candidates for password length L.

| Length | Keyspace | GPU (GTX 1080) | CPU (single core) |
|--------|----------|----------------|-------------------|
| 3 | 238,328 | instant | instant |
| 4 | 14,776,336 | instant | 1.5s |
| 5 | 916,132,832 | < 1s | ~1.5 min |
| 6 | 56.8 billion | ~38s | ~1.6 hours |
| 7 | 3.5 trillion | ~40 min | ~4 days |
| 8 | 218 trillion | ~40 hours | ~253 days |

## Building

### Requirements

- GCC (for CPU version)
- NVIDIA GPU + CUDA toolkit (for GPU version, tested with CUDA 12.8)

```bash
# Build both versions
make

# Build GPU version only
make gpu

# Build CPU version only (no CUDA required)
make cpu
```

## Project structure

```
md5hack/
├── Makefile
├── README.md
└── src/
    ├── main.cu           GPU version: CUDA kernel launch, progress display
    ├── md5_gpu.cuh       MD5 algorithm on GPU (register-only, __device__)
    ├── charset.cuh       Candidate generation from thread index (__device__)
    └── md5hack_cpu.c     CPU version: single-threaded C, same algorithm
```

## How it works

1. The target MD5 hash is parsed and stored in GPU **constant memory** (fast broadcast to all threads)
2. The keyspace (62^length candidates) is divided into batches of ~2 million
3. Each batch launches a CUDA kernel with 8192 blocks × 256 threads
4. Every thread converts its global index into a password candidate (base-62 encoding)
5. The MD5 hash is computed entirely in GPU **registers** — no shared or global memory access during hashing
6. If a thread finds a match, it writes the result via an atomic operation
7. The host checks for results between batches and displays progress (speed, percentage)

The CPU version follows the same steps but sequentially: one candidate at a time in a simple for loop.

## Why MD5 should never be used for passwords

MD5 was widely used in the 2000s to store hashed passwords in databases. Instead of saving the password in plain text, applications would store `MD5(password)` — if the database leaked, attackers would only see hashes, not passwords.

**This is no longer safe.** As this tool demonstrates, a single consumer GPU can test 1.5 billion MD5 candidates per second. A high-end RTX 4090 reaches 164 billion MD5/s ([source](https://gist.github.com/Chick3nman/32e662a5bb63bc4f51b847bb422222fd)). An 8-character alphanumeric password hashed with MD5 can be cracked in under 48 minutes on such hardware ([source](https://www.bitdefender.com/en-us/blog/hotforsecurity/rtx-4090-8-card-rig-cracks-random-and-powerful-eight-character-passwords-in-48-minutes)).

The fundamental problem: **MD5 is too fast.** It was designed for checksums and data integrity, not for password storage. A good password hash must be intentionally slow to make bruteforce impractical.

### What to use instead

Modern password hashing algorithms are designed to be slow, memory-hard, and resistant to GPU acceleration:

| Algorithm | RTX 4090 speed | vs MD5 | Status |
|-----------|---------------|--------|--------|
| MD5 | 164 GH/s | baseline | **Broken** — never use |
| SHA-256 | ~22 GH/s | ~7x slower | **Broken** for passwords |
| bcrypt (cost 10) | 184 kH/s | **~900,000x slower** | Recommended |
| Argon2id | ~10 kH/s | **~16,000,000x slower** | Best practice |

At 184 kH/s for bcrypt, cracking an 8-character password would take **~37,000 years** instead of 48 minutes with MD5.

**Current recommendations** ([OWASP Password Storage Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Password_Storage_Cheat_Sheet.html)):

1. **Argon2id** (winner of the 2015 Password Hashing Competition) — the gold standard. Memory-hard, resistant to GPU/ASIC attacks. Recommended config: 19 MiB memory, 2 iterations, 1 thread.
2. **bcrypt** — battle-tested since 1999, widely supported. Use cost factor 10+. Limited to 72-byte passwords.
3. **scrypt** — memory-hard alternative, good when Argon2 is unavailable.

Never use MD5, SHA-1, SHA-256, or any fast hash for password storage — even with salting.

### References

- [OWASP Password Storage Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Password_Storage_Cheat_Sheet.html)
- [Password Hashing Guide 2025: Argon2 vs Bcrypt vs Scrypt vs PBKDF2](https://guptadeepak.com/the-complete-guide-to-password-hashing-argon2-vs-bcrypt-vs-scrypt-vs-pbkdf2-2026/)
- [Hashcat RTX 4090 benchmark](https://gist.github.com/Chick3nman/32e662a5bb63bc4f51b847bb422222fd)
- [RTX 4090 cracks 8-char passwords in 48 minutes](https://www.bitdefender.com/en-us/blog/hotforsecurity/rtx-4090-8-card-rig-cracks-random-and-powerful-eight-character-passwords-in-48-minutes)

## Author

**Loïc Guillois** — [github.com/loicguillois](https://github.com/loicguillois)

## License

MIT
