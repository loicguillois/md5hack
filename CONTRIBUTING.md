# Contributing to md5hack

Thanks for your interest in contributing!

## How to contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-change`)
3. Make your changes
4. Test that the project builds: `make clean && make`
5. Test with a known hash: `./md5hack 202cb962ac59075b964b07152d234b70 3`
6. Commit your changes with a clear message
7. Push to your fork and open a Pull Request

## Ideas for contributions

- Support for additional charsets (special characters, unicode)
- Multi-GPU support (split keyspace across devices)
- Dictionary attack mode (wordlist input)
- Support for other hash algorithms (SHA-1, SHA-256)
- Resume/checkpoint support for long-running cracks
- Automatic password length detection (try lengths 1, 2, 3, ... sequentially)

## Code style

- CUDA C++ with clear separation between host and device code
- English comments — didactic style, explain the GPU/crypto concepts
- Keep it focused: one tool, one purpose
