package org.navneev;

import org.junit.jupiter.api.Test;
import org.navneev.utils.VectorDistanceCalculationUtils;

import static org.junit.jupiter.api.Assertions.*;

class VectorDistanceCalculationUtilsTest {

    private static final double DELTA = 1e-6;

    @Test
    void testEuclideanDistanceIdenticalVectors() {
        float[] a = {1.0f, 2.0f, 3.0f};
        float[] b = {1.0f, 2.0f, 3.0f};

        double result = VectorDistanceCalculationUtils.euclideanDistance(a, b);
        assertEquals(0.0, result, DELTA);
    }

    @Test
    void testEuclideanDistanceSimpleCase() {
        float[] a = {0.0f, 0.0f};
        float[] b = {3.0f, 4.0f};

        double result = VectorDistanceCalculationUtils.euclideanDistance(a, b);
        assertEquals(25.0, result, DELTA); // 3² + 4² = 25
    }

    @Test
    void testEuclideanDistanceSingleElement() {
        float[] a = {5.0f};
        float[] b = {2.0f};

        double result = VectorDistanceCalculationUtils.euclideanDistance(a, b);
        assertEquals(9.0, result, DELTA); // (5-2)² = 9
    }

    @Test
    void testEuclideanDistanceLargeVector() {
        // Test with vector larger than SIMD register size
        float[] a = new float[100];
        float[] b = new float[100];

        for (int i = 0; i < 100; i++) {
            a[i] = i;
            b[i] = i + 1;
        }

        double result = VectorDistanceCalculationUtils.euclideanDistance(a, b);
        assertEquals(100.0, result, DELTA); // 100 * 1² = 100
    }

    @Test
    void testEuclideanDistanceNegativeValues() {
        float[] a = {-1.0f, -2.0f, -3.0f};
        float[] b = {1.0f, 2.0f, 3.0f};

        double result = VectorDistanceCalculationUtils.euclideanDistance(a, b);
        assertEquals(56.0, result, DELTA); // (-2)² + (-4)² + (-6)² = 4 + 16 + 36 = 56
    }

    @Test
    void testEuclideanDistanceZeroVectors() {
        float[] a = {0.0f, 0.0f, 0.0f};
        float[] b = {0.0f, 0.0f, 0.0f};

        double result = VectorDistanceCalculationUtils.euclideanDistance(a, b);
        assertEquals(0.0, result, DELTA);
    }

    @Test
    void testEuclideanDistanceFloatingPoint() {
        float[] a = {1.5f, 2.7f, 3.14f};
        float[] b = {1.1f, 2.3f, 3.0f};

        // Calculate expected using float precision to match SIMD implementation
        float diff1 = 1.5f - 1.1f;
        float diff2 = 2.7f - 2.3f;
        float diff3 = 3.14f - 3.0f;
        double expected = diff1 * diff1 + diff2 * diff2 + diff3 * diff3;

        double result = VectorDistanceCalculationUtils.euclideanDistance(a, b);

        assertEquals(expected, result, DELTA);
    }

    @Test
    void testInnerProductIdenticalVectors() {
        float[] a = {1.0f, 2.0f, 3.0f};
        float[] b = {1.0f, 2.0f, 3.0f};

        float result = VectorDistanceCalculationUtils.innerProduct(a, b);
        assertEquals(14.0f, result, DELTA); // 1*1 + 2*2 + 3*3 = 14
    }

    @Test
    void testInnerProductOrthogonalVectors() {
        float[] a = {1.0f, 0.0f};
        float[] b = {0.0f, 1.0f};

        float result = VectorDistanceCalculationUtils.innerProduct(a, b);
        assertEquals(0.0f, result, DELTA);
    }

    @Test
    void testInnerProductSimpleCase() {
        float[] a = {2.0f, 3.0f, 4.0f};
        float[] b = {1.0f, 2.0f, 3.0f};

        float result = VectorDistanceCalculationUtils.innerProduct(a, b);
        assertEquals(20.0f, result, DELTA); // 2*1 + 3*2 + 4*3 = 20
    }

    @Test
    void testInnerProductSingleElement() {
        float[] a = {5.0f};
        float[] b = {3.0f};

        float result = VectorDistanceCalculationUtils.innerProduct(a, b);
        assertEquals(15.0f, result, DELTA);
    }

    @Test
    void testInnerProductLargeVector() {
        float[] a = new float[100];
        float[] b = new float[100];

        for (int i = 0; i < 100; i++) {
            a[i] = 2.0f;
            b[i] = 3.0f;
        }

        float result = VectorDistanceCalculationUtils.innerProduct(a, b);
        assertEquals(600.0f, result, DELTA); // 100 * (2 * 3) = 600
    }

    @Test
    void testInnerProductNegativeValues() {
        float[] a = {-1.0f, 2.0f, -3.0f};
        float[] b = {4.0f, -5.0f, 6.0f};

        float result = VectorDistanceCalculationUtils.innerProduct(a, b);
        assertEquals(-32.0f, result, DELTA); // (-1*4) + (2*-5) + (-3*6) = -4 - 10 - 18 = -32
    }

    @Test
    void testInnerProductZeroVectors() {
        float[] a = {0.0f, 0.0f, 0.0f};
        float[] b = {1.0f, 2.0f, 3.0f};

        float result = VectorDistanceCalculationUtils.innerProduct(a, b);
        assertEquals(0.0f, result, DELTA);
    }

    @Test
    void testInnerProductEmptyArrays() {
        float[] a = {};
        float[] b = {};

        float result = VectorDistanceCalculationUtils.innerProduct(a, b);
        assertEquals(0.0f, result, DELTA);
    }
}