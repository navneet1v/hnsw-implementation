#!/bin/bash

echo "=== Compiling with AVX2 ==="
g++ -O3 -mavx2 -mfma -o main_avx2 main.cpp
echo ""

echo "=== Running AVX2 version ==="
./main_avx2
echo ""

echo "=== Compiling with AVX512 ==="
g++ -O3 -mavx512f -o main_avx512 main.cpp
echo ""

echo "=== Running AVX512 version ==="
./main_avx512
echo ""

rm -f main_avx2 main_avx512
