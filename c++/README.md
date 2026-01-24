# SIMD Distance Computation Benchmark

High-performance Euclidean distance computation using AVX2 and AVX512 SIMD instructions.

# Results for 100K dataset with 10K queries
Ran on a R7iz.4xlarge machine https://aws.amazon.com/ec2/instance-types/r7iz/

| Dimension | AVX2 Time | AVX512 Time | Speedup |
|-----------|-----------|-------------|----------|
| 768       | 17.72s    | 15.66s      | 1.13x    |
| 1536      | 34.70s    | 31.95s      | 1.09x    |
| 3072      | 69.72s    | 64.15s      | 1.09x    |
| 4096      | 96.03s    | 85.91s      | 1.12x    |

## Raw results
```
bash ./build_and_run.sh 
=== Compiling with AVX2 ===

=== Running AVX2 version ===
Using AVX2
Dim 768 (1000 queries x 100000 vectors): 17.7229s
Dim 1536 (1000 queries x 100000 vectors): 34.7004s
Dim 3072 (1000 queries x 100000 vectors): 69.7185s
Dim 4096 (1000 queries x 100000 vectors): 96.027s

=== Compiling with AVX512 ===

=== Running AVX512 version ===
Using AVX512
Dim 768 (1000 queries x 100000 vectors): 15.6638s
Dim 1536 (1000 queries x 100000 vectors): 31.9533s
Dim 3072 (1000 queries x 100000 vectors): 64.1456s
Dim 4096 (1000 queries x 100000 vectors): 85.9123s

```




## Features

- **AVX512**: 512-bit SIMD processing 16 floats at a time
- **AVX2**: 256-bit SIMD processing 8 floats at a time
- **Batch-8 Processing**: Computes 8 distances simultaneously for better performance
- **Multiple Dimensions**: Tests with 768, 1536, 3072, and 4096 dimensions

## Benchmark

- 1,000,000 vectors
- 10,000 query vectors
- Computes 10 billion distance calculations

## Build & Run

```bash
./build_and_run.sh
```

Or manually:

```bash
# AVX2
g++ -O3 -mavx2 -mfma -o main_avx2 main.cpp
./main_avx2

# AVX512
g++ -O3 -mavx512f -o main_avx512 main.cpp
./main_avx512
```

### What is -O3 doing?
-O3 is a g++ compiler optimization flag that stands for "Optimization level 3".

#### Optimization Levels:
-O0: No optimization (default, fastest compilation)
-O1: Basic optimizations
-O2: Moderate optimizations (recommended for most cases)
-O3: Aggressive optimizations (maximum performance)

#### What -O3 does:
* Enables all -O2 optimizations plus more aggressive ones
* Function inlining (replaces function calls with actual code)
* Loop unrolling (duplicates loop body to reduce overhead)
* Vectorization (automatically uses SIMD when possible)
* More aggressive code transformations

## Requirements

- C++11 or later
- CPU with AVX2 support (for AVX2 build)
- CPU with AVX512F support (for AVX512 build)
- AL2023 or compatible Linux distribution
