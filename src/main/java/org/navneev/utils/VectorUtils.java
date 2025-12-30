package org.navneev.utils;


import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * Utility class providing SIMD-optimized vector operations for high-performance computing.
 * 
 * <p>This class leverages the Java Vector API (JDK 23+) to perform vectorized operations
 * on float arrays, utilizing Single Instruction Multiple Data (SIMD) instructions when
 * available on the target hardware. Operations automatically fall back to scalar
 * computation for array elements that don't fit in complete SIMD vectors.
 * 
 * <p>All methods in this class are static and thread-safe. The class uses the preferred
 * vector species for the target platform to maximize performance.
 * 
 * <h3>Performance Considerations:</h3>
 * <ul>
 *   <li>SIMD operations provide significant speedup for large arrays</li>
 *   <li>Performance gains are most noticeable with arrays larger than the SIMD register size</li>
 *   <li>Optimal performance requires proper JVM flags: {@code --add-modules jdk.incubator.vector}</li>
 * </ul>
 * 
 * <h3>Usage Example:</h3>
 * <pre>{@code
 * float[] vector1 = {1.0f, 2.0f, 3.0f, 4.0f};
 * float[] vector2 = {2.0f, 3.0f, 4.0f, 5.0f};
 * 
 * float distance = VectorUtils.euclideanDistance(vector1, vector2);
 * float dotProduct = VectorUtils.innerProduct(vector1, vector2);
 * }</pre>
 * 
 * @author Navneev
 * @version 1.0
 * @since JDK 23
 */
public class VectorUtils {

    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

    /**
     * Computes the squared Euclidean distance between two float arrays using SIMD optimization.
     * 
     * <p>The squared Euclidean distance is calculated as the sum of squared differences:
     * distance² = Σ(a[i] - b[i])²
     * 
     * <p>This implementation uses the Vector API for SIMD acceleration, processing multiple
     * elements simultaneously when possible, and falls back to scalar computation for
     * remaining elements.
     * 
     * @param a the first vector as a float array
     * @param b the second vector as a float array
     * @return the squared Euclidean distance between the two vectors
     * @throws IllegalArgumentException if arrays have different lengths
     * @throws NullPointerException if either array is null
     * 
     * @since JDK 23 (requires --add-modules jdk.incubator.vector)
     */
    public static double euclideanDistance(float[] a, float[] b) {
        // 1. Get vector length (how many floats can fit in one SIMD register)
        int vectorLength = SPECIES.length();
        // 2. Process arrays in chunks of vectorLength
        int loopBound = SPECIES.loopBound(a.length);
        double sum = 0;

        for(int i = 0 ; i < loopBound; i += vectorLength) {
            FloatVector va = FloatVector.fromArray(SPECIES, a, i);
            FloatVector vb = FloatVector.fromArray(SPECIES, b, i);

            FloatVector diff = va.sub(vb);
            FloatVector squared = diff.mul(diff);

            sum += squared.reduceLanes(VectorOperators.ADD);
        }

        // 3. Handle remaining elements with scalar code
        for(int i = loopBound; i < a.length; i++) {
            double diff = a[i] - b[i];
            sum += diff * diff;
        }

        // 4. Return sum
        return sum;
    }

    /**
     * Computes the inner product (dot product) of two float arrays using SIMD optimization.
     * 
     * <p>The inner product is calculated as the sum of element-wise products:
     * result = Σ(a[i] * b[i])
     * 
     * <p>This implementation uses the Vector API for SIMD acceleration, processing multiple
     * elements simultaneously when possible, and falls back to scalar computation for
     * remaining elements.
     * 
     * @param a the first vector as a float array
     * @param b the second vector as a float array
     * @return the inner product of the two vectors
     * @throws IllegalArgumentException if arrays have different lengths
     * @throws NullPointerException if either array is null
     * 
     * @since JDK 23 (requires --add-modules jdk.incubator.vector)
     */
    public static float innerProduct(float[] a, float[] b) {
        // 1. Get vector length (how many floats can fit in one SIMD register)
        int vectorLength = SPECIES.length();

        // 2. Process arrays in chunks of SIMD length
        int loopBound = SPECIES.loopBound(a.length);
        double sum = 0;

        // 3. Rum the loop to do inner product
        for(int i = 0; i < loopBound; i += vectorLength) {
            FloatVector va = FloatVector.fromArray(SPECIES, a, i);
            FloatVector vb = FloatVector.fromArray(SPECIES, b, i);

            FloatVector mul = va.mul(vb);
            sum += mul.reduceLanes(VectorOperators.ADD);
        }

        //4. Do the remaining elements with scaler code
        for(int i = loopBound; i < a.length; i++) {
            sum += a[i] * b[i];
        }

        // 5. Now return the sum
        return (float)sum;
    }

}

