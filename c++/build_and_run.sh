#!/bin/bash

MODE="both"
if [ $# -gt 0 ]; then
    case "$1" in
        simd|--simd|-s)
            MODE="simd"
            ;;
        bulk-simd|--bulk-simd|-bs)
            MODE="bulk-simd"
            ;;
        both|--both|-b)
            MODE="both"
            ;;
        *)
            echo "Usage: $0 [simd|bulk-simd|both]"
            echo "  simd:      Run only regular SIMD benchmarks"
            echo "  bulk-simd: Run only bulk SIMD benchmarks"
            echo "  both:      Run both (default)"
            exit 1
            ;;
    esac
fi

echo "=== Compiling with AVX2 ==="
g++ -O3 -mavx2 -mfma -o main_avx2 main.cpp
echo ""

if [ "$MODE" = "simd" ] || [ "$MODE" = "both" ]; then
    echo "=== Running AVX2 with regular SIMD ==="
    ./main_avx2 --simd
    echo ""
fi

if [ "$MODE" = "bulk-simd" ] || [ "$MODE" = "both" ]; then
    echo "=== Running AVX2 with bulk SIMD ==="
    ./main_avx2 --bulk-simd
    echo ""
fi

echo "=== Compiling with AVX512 ==="
g++ -O3 -mavx512f -o main_avx512 main.cpp
echo ""

if [ "$MODE" = "simd" ] || [ "$MODE" = "both" ]; then
    echo "=== Running AVX512 with regular SIMD ==="
    ./main_avx512 --simd
    echo ""
fi

if [ "$MODE" = "bulk-simd" ] || [ "$MODE" = "both" ]; then
    echo "=== Running AVX512 with bulk SIMD ==="
    ./main_avx512 --bulk-simd
    echo ""
fi

rm -f main_avx2 main_avx512
