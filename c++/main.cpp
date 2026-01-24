#include <iostream>
#include <immintrin.h>
#include <vector>
#include <chrono>

#ifdef __AVX512F__
// AVX512: Compute squared Euclidean distance between two vectors using 512-bit SIMD (16 floats at a time)
float distance_simd(const float* a, const float* b, int dim) {
    __m512 sum = _mm512_setzero_ps();
    for (int i = 0; i < dim; i += 16) {
        __m512 va = _mm512_loadu_ps(&a[i]);        // Load 16 floats from vector a
        __m512 vb = _mm512_loadu_ps(&b[i]);        // Load 16 floats from vector b
        __m512 diff = _mm512_sub_ps(va, vb);       // Compute difference
        sum = _mm512_fmadd_ps(diff, diff, sum);    // Fused multiply-add: sum += diff * diff
    }
    return _mm512_reduce_add_ps(sum);              // Horizontal sum of all 16 elements
}

// AVX512: Compute squared distances for 8 vectors simultaneously against one query vector
void distance_simd_batch8(const float* a, const float* b0, const float* b1, const float* b2, const float* b3,
                          const float* b4, const float* b5, const float* b6, const float* b7, int dim, float* results) {
    __m512 sum0 = _mm512_setzero_ps(), sum1 = _mm512_setzero_ps(), sum2 = _mm512_setzero_ps(), sum3 = _mm512_setzero_ps();
    __m512 sum4 = _mm512_setzero_ps(), sum5 = _mm512_setzero_ps(), sum6 = _mm512_setzero_ps(), sum7 = _mm512_setzero_ps();
    for (int i = 0; i < dim; i += 16) {
        __m512 va = _mm512_loadu_ps(&a[i]);        // Load query vector once
        __m512 diff0 = _mm512_sub_ps(va, _mm512_loadu_ps(&b0[i]));
        __m512 diff1 = _mm512_sub_ps(va, _mm512_loadu_ps(&b1[i]));
        __m512 diff2 = _mm512_sub_ps(va, _mm512_loadu_ps(&b2[i]));
        __m512 diff3 = _mm512_sub_ps(va, _mm512_loadu_ps(&b3[i]));
        __m512 diff4 = _mm512_sub_ps(va, _mm512_loadu_ps(&b4[i]));
        __m512 diff5 = _mm512_sub_ps(va, _mm512_loadu_ps(&b5[i]));
        __m512 diff6 = _mm512_sub_ps(va, _mm512_loadu_ps(&b6[i]));
        __m512 diff7 = _mm512_sub_ps(va, _mm512_loadu_ps(&b7[i]));
        sum0 = _mm512_fmadd_ps(diff0, diff0, sum0); // Accumulate squared differences
        sum1 = _mm512_fmadd_ps(diff1, diff1, sum1);
        sum2 = _mm512_fmadd_ps(diff2, diff2, sum2);
        sum3 = _mm512_fmadd_ps(diff3, diff3, sum3);
        sum4 = _mm512_fmadd_ps(diff4, diff4, sum4);
        sum5 = _mm512_fmadd_ps(diff5, diff5, sum5);
        sum6 = _mm512_fmadd_ps(diff6, diff6, sum6);
        sum7 = _mm512_fmadd_ps(diff7, diff7, sum7);
    }
    results[0] = _mm512_reduce_add_ps(sum0);       // Reduce and store results
    results[1] = _mm512_reduce_add_ps(sum1);
    results[2] = _mm512_reduce_add_ps(sum2);
    results[3] = _mm512_reduce_add_ps(sum3);
    results[4] = _mm512_reduce_add_ps(sum4);
    results[5] = _mm512_reduce_add_ps(sum5);
    results[6] = _mm512_reduce_add_ps(sum6);
    results[7] = _mm512_reduce_add_ps(sum7);
}
#else
// AVX2: Compute squared Euclidean distance between two vectors using 256-bit SIMD (8 floats at a time)
float distance_simd(const float* a, const float* b, int dim) {
    __m256 sum = _mm256_setzero_ps();
    for (int i = 0; i < dim; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);        // Load 8 floats from vector a
        __m256 vb = _mm256_loadu_ps(&b[i]);        // Load 8 floats from vector b
        __m256 diff = _mm256_sub_ps(va, vb);       // Compute difference
        sum = _mm256_fmadd_ps(diff, diff, sum);    // Fused multiply-add: sum += diff * diff
    }
    float result[8];
    _mm256_storeu_ps(result, sum);                 // Store SIMD register to array
    return result[0] + result[1] + result[2] + result[3] +  // Manual horizontal sum
           result[4] + result[5] + result[6] + result[7];
}

// AVX2: Compute squared distances for 8 vectors simultaneously against one query vector
void distance_simd_batch8(const float* a, const float* b0, const float* b1, const float* b2, const float* b3,
                          const float* b4, const float* b5, const float* b6, const float* b7, int dim, float* results) {
    __m256 sum0 = _mm256_setzero_ps(), sum1 = _mm256_setzero_ps(), sum2 = _mm256_setzero_ps(), sum3 = _mm256_setzero_ps();
    __m256 sum4 = _mm256_setzero_ps(), sum5 = _mm256_setzero_ps(), sum6 = _mm256_setzero_ps(), sum7 = _mm256_setzero_ps();
    for (int i = 0; i < dim; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);        // Load query vector once
        __m256 diff0 = _mm256_sub_ps(va, _mm256_loadu_ps(&b0[i]));
        __m256 diff1 = _mm256_sub_ps(va, _mm256_loadu_ps(&b1[i]));
        __m256 diff2 = _mm256_sub_ps(va, _mm256_loadu_ps(&b2[i]));
        __m256 diff3 = _mm256_sub_ps(va, _mm256_loadu_ps(&b3[i]));
        __m256 diff4 = _mm256_sub_ps(va, _mm256_loadu_ps(&b4[i]));
        __m256 diff5 = _mm256_sub_ps(va, _mm256_loadu_ps(&b5[i]));
        __m256 diff6 = _mm256_sub_ps(va, _mm256_loadu_ps(&b6[i]));
        __m256 diff7 = _mm256_sub_ps(va, _mm256_loadu_ps(&b7[i]));
        sum0 = _mm256_fmadd_ps(diff0, diff0, sum0); // Accumulate squared differences
        sum1 = _mm256_fmadd_ps(diff1, diff1, sum1);
        sum2 = _mm256_fmadd_ps(diff2, diff2, sum2);
        sum3 = _mm256_fmadd_ps(diff3, diff3, sum3);
        sum4 = _mm256_fmadd_ps(diff4, diff4, sum4);
        sum5 = _mm256_fmadd_ps(diff5, diff5, sum5);
        sum6 = _mm256_fmadd_ps(diff6, diff6, sum6);
        sum7 = _mm256_fmadd_ps(diff7, diff7, sum7);
    }
    float temp[8];
    _mm256_storeu_ps(temp, sum0);                  // Store and manually reduce each sum
    results[0] = temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6] + temp[7];
    _mm256_storeu_ps(temp, sum1);
    results[1] = temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6] + temp[7];
    _mm256_storeu_ps(temp, sum2);
    results[2] = temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6] + temp[7];
    _mm256_storeu_ps(temp, sum3);
    results[3] = temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6] + temp[7];
    _mm256_storeu_ps(temp, sum4);
    results[4] = temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6] + temp[7];
    _mm256_storeu_ps(temp, sum5);
    results[5] = temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6] + temp[7];
    _mm256_storeu_ps(temp, sum6);
    results[6] = temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6] + temp[7];
    _mm256_storeu_ps(temp, sum7);
    results[7] = temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6] + temp[7];
}
#endif

int main() {
    const int dims[] = {768, 1536, 3072, 4096};
    const int num_vectors = 1000000;
    const int num_queries = 10000;

    if (num_vectors % 8 != 0) {
        std::cout << "Error: num_vectors must be divisible by 8" << std::endl;
        return 1;
    }

#ifdef __AVX512F__
    std::cout << "Using AVX512" << std::endl;
#else
    std::cout << "Using AVX2" << std::endl;
#endif

    for (int dim : dims) {
        {
            std::vector<float> vectors(num_vectors * dim);
            std::vector<float> queries(num_queries * dim);

            for (int i = 0; i < num_vectors * dim; ++i) vectors[i] = static_cast<float>(rand()) / RAND_MAX;
            for (int i = 0; i < num_queries * dim; ++i) queries[i] = static_cast<float>(rand()) / RAND_MAX;

            auto start = std::chrono::high_resolution_clock::now();
            float results[8];
            for (int q = 0; q < num_queries; ++q) {
                for (int v = 0; v < num_vectors; v += 8) {
                    distance_simd_batch8(&queries[q * dim], &vectors[v * dim], &vectors[(v+1) * dim],
                                        &vectors[(v+2) * dim], &vectors[(v+3) * dim], &vectors[(v+4) * dim],
                                        &vectors[(v+5) * dim], &vectors[(v+6) * dim], &vectors[(v+7) * dim], dim, results);
                }
            }
            auto end = std::chrono::high_resolution_clock::now();

            std::cout << "Dim " << dim << " (" << num_queries << " queries x " << num_vectors << " vectors): "
                      << std::chrono::duration<double>(end - start).count() << "s" << std::endl;
        }
    }
    return 0;
}