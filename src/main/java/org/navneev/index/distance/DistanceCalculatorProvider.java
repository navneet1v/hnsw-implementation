package org.navneev.index.distance;

import org.navneev.index.storage.OffHeapVectorsStorage;
import org.navneev.index.storage.OnHeapVectorStorage;
import org.navneev.index.storage.VectorStorage;

/**
 * Factory and provider for creating and managing DistanceCalculator instances based on storage type.
 *
 * <p>This class abstracts the complexity of choosing the optimal distance calculation strategy
 * based on the underlying vector storage implementation. It automatically selects between:
 * <ul>
 *   <li><b>MemorySegment-based:</b> For off-heap storage with direct memory access</li>
 *   <li><b>Array-based:</b> For on-heap storage or when MemorySegment is disabled</li>
 * </ul>
 *
 * <p><b>Configuration:</b>
 * <p>The behavior can be controlled via system property:
 * <pre>{@code
 * // Enable MemorySegment for off-heap storage (default)
 * java -Dvector.use.memory.segment=true MyApp
 *
 * // Disable MemorySegment, use array-based approach
 * java -Dvector.use.memory.segment=false MyApp
 * }</pre>
 *
 * <p><b>Performance Characteristics:</b>
 * <table border="1">
 *   <tr>
 *     <th>Storage Type</th>
 *     <th>Strategy</th>
 *     <th>Performance</th>
 *   </tr>
 *   <tr>
 *     <td>OffHeapVectorsStorage</td>
 *     <td>MemorySegment (default)</td>
 *     <td>5-15% faster, no array allocation</td>
 *   </tr>
 *   <tr>
 *     <td>OffHeapVectorsStorage</td>
 *     <td>Array (with cloning)</td>
 *     <td>Slower, requires copy to heap</td>
 *   </tr>
 *   <tr>
 *     <td>OnHeapVectorStorage</td>
 *     <td>Array (direct reference)</td>
 *     <td>Optimal for heap-based data</td>
 *   </tr>
 * </table>
 *
 * <p><b>Usage Example:</b>
 * <pre>{@code
 * VectorStorage storage = StorageFactory.createStorage(dimensions, capacity);
 * DistanceCalculatorProvider provider = new DistanceCalculatorProvider(storage);
 *
 * // Create calculator for a stored vector
 * DistanceCalculator calc = provider.createDistanceCalculatorForVectorStorage(vectorId, dimensions);
 *
 * // Calculate distance to another vector
 * double distance = provider.calculateDistanceFromId(calc, otherVectorId);
 * }</pre>
 *
 * <p><b>Thread Safety:</b> This class is thread-safe after construction. Multiple threads can
 * safely call the factory methods concurrently.
 *
 * @author Navneev
 * @version 1.0
 * @since 1.0
 */
public class DistanceCalculatorProvider {

    /**
     * System property key for enabling/disabling MemorySegment-based distance calculation
     */
    private static final String USE_MEMORY_SEGMENT = "vector.use.memory.segment";

    /**
     * Default value for MemorySegment usage (enabled for better performance)
     */
    private static final String USE_MEMORY_SEGMENT_DEFAULT = "true";

    /**
     * Flag indicating whether to use MemorySegment-based distance calculation
     */
    private final boolean useMemorySegment;

    /**
     * Flag indicating whether vectors need to be cloned before use
     */
    private final boolean cloneVector;

    /**
     * The underlying vector storage implementation
     */
    private final VectorStorage vectorStorage;

    /**
     * Constructs a DistanceCalculatorProvider for the given vector storage.
     *
     * <p>Automatically determines the optimal distance calculation strategy based on:
     * <ul>
     *   <li>Storage type (on-heap vs off-heap)</li>
     *   <li>System property configuration</li>
     *   <li>Performance characteristics</li>
     * </ul>
     *
     * <p><b>Decision Logic:</b>
     * <pre>
     * OffHeapVectorsStorage + vector.use.memory.segment=true  → MemorySegment (fastest)
     * OffHeapVectorsStorage + vector.use.memory.segment=false → Array with cloning
     * OnHeapVectorStorage                                     → Array (direct reference)
     * </pre>
     *
     * @param inputVectorStorage the vector storage implementation to use
     * @throws IllegalArgumentException if storage type is not supported
     */
    public DistanceCalculatorProvider(final VectorStorage inputVectorStorage) {
        this.vectorStorage = inputVectorStorage;
        if (inputVectorStorage instanceof OffHeapVectorsStorage) {
            if (Boolean.parseBoolean(System.getProperty(USE_MEMORY_SEGMENT, USE_MEMORY_SEGMENT_DEFAULT))) {
                System.out.printf("%s property set to true hence using memory segment \n", USE_MEMORY_SEGMENT);
                useMemorySegment = true;
                cloneVector = false;
            } else {
                System.out.printf("%s property set to false hence not using memory segment \n", USE_MEMORY_SEGMENT);
                useMemorySegment = false;
                cloneVector = true;
            }
        } else if (inputVectorStorage instanceof OnHeapVectorStorage) {
            System.out.println("Storage type is OnHeapVectorStorage hence not using memory segment");
            useMemorySegment = false;
            cloneVector = false;
        } else {
            throw new IllegalArgumentException("Unknown vector storage: " + vectorStorage.getClass().getName());
        }
    }

    /**
     * Creates a new DistanceCalculator for a vector stored in the vector storage.
     *
     * <p>This method creates an appropriate calculator based on the storage type and configuration:
     * <ul>
     *   <li><b>MemorySegment mode:</b> Returns calculator with direct memory access</li>
     *   <li><b>Array mode with cloning:</b> Copies vector to new array (thread-safe)</li>
     *   <li><b>Array mode without cloning:</b> Returns calculator with direct array reference</li>
     * </ul>
     *
     * <p><b>Performance Note:</b> For off-heap storage with MemorySegment enabled,
     * this is 5-15% faster than array-based approach.
     *
     * @param vectorId the ID of the vector in storage
     * @param length   the number of dimensions in the vector
     * @return a DistanceCalculator configured for the specified vector
     */
    public DistanceCalculator createDistanceCalculatorForVectorStorage(final int vectorId, int length) {
        if (useMemorySegment) {
            return new MemorySegmentBasedDistanceCalculator(vectorStorage.getMemorySegment(vectorId), length);
        } else {
            if (cloneVector) {
                final float[] vector = new float[length];
                loadVectorInArray(vectorId, vector, vectorStorage);
                return new ArrayBasedDistanceCalculator(vector);
            } else {
                return new ArrayBasedDistanceCalculator(vectorStorage.getVector(vectorId));
            }
        }
    }

    /**
     * Updates an existing DistanceCalculator to reference a different vector.
     *
     * <p>This method is more efficient than creating a new calculator when you need to
     * compute distances for multiple vectors sequentially. It reuses the calculator object
     * and updates its internal reference.
     *
     * <p><b>Use Case:</b>
     * <pre>{@code
     * DistanceCalculator calc = provider.createDistanceCalculatorForVectorStorage(id1, dims);
     * double dist1 = calc.calculate(otherVector);
     *
     * // Reuse calculator for different vector
     * provider.updateDistanceCalculator(calc, id2, dims);
     * double dist2 = calc.calculate(otherVector);
     * }</pre>
     *
     * @param distanceCalculator the calculator to update
     * @param vectorId           the new vector ID to reference
     * @param length             the number of dimensions in the vector
     */
    public void updateDistanceCalculator(final DistanceCalculator distanceCalculator, final int vectorId, int length) {
        if (useMemorySegment) {
            distanceCalculator.update(vectorStorage.getMemorySegment(vectorId));
        } else {
            if (cloneVector) {
                final float[] vector = new float[length];
                loadVectorInArray(vectorId, vector, vectorStorage);
                distanceCalculator.update(vector);
            } else {
                distanceCalculator.update(vectorStorage.getVector(vectorId));
            }
        }
    }

    /**
     * Creates a DistanceCalculator for a query vector provided as a float array.
     *
     * <p>This method is typically used for query vectors that are not stored in the vector storage,
     * such as search queries in k-NN search operations.
     *
     * <p><b>Note:</b> Always returns an ArrayBasedDistanceCalculator regardless of storage type,
     * since the input is already a heap-allocated array.
     *
     * @param vector the query vector as a float array
     * @return an ArrayBasedDistanceCalculator for the query vector
     */
    public DistanceCalculator createDistanceCalculator(float[] vector) {
        return new ArrayBasedDistanceCalculator(vector);
    }

    /**
     * Calculates the distance between a vector represented by a DistanceCalculator and
     * another vector identified by its ID in storage.
     *
     * <p>This is a convenience method that handles the complexity of retrieving the vector
     * from storage in the appropriate format (MemorySegment or array) based on configuration.
     *
     * <p><b>Performance:</b> Uses the optimal access method for the storage type:
     * <ul>
     *   <li>MemorySegment mode: Direct memory access, no allocation</li>
     *   <li>Array mode: Array retrieval from storage</li>
     * </ul>
     *
     * @param distanceCalculator the calculator containing the first vector
     * @param vectorId           the ID of the second vector in storage
     * @return the Euclidean distance between the two vectors
     */
    public float calculateDistanceFromId(DistanceCalculator distanceCalculator, int vectorId) {
        if (useMemorySegment) {
            return distanceCalculator.calculate(vectorStorage.getMemorySegment(vectorId));
        } else {
            return distanceCalculator.calculate(vectorStorage.getVector(vectorId));
        }
    }

    /**
     * Loads a vector from storage into a pre-allocated array.
     *
     * <p>This is a helper method used internally when vector cloning is required
     * (off-heap storage with MemorySegment disabled).
     *
     * @param id            the vector ID in storage
     * @param vector        the pre-allocated array to load the vector into
     * @param vectorStorage the storage to load from
     */
    private void loadVectorInArray(int id, float[] vector, VectorStorage vectorStorage) {
        vectorStorage.loadVectorInArray(id, vector);
    }
}
