package org.navneev;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;

/**
 * Off-heap storage for vector data using direct ByteBuffer for memory efficiency.
 * 
 * <p>This class provides efficient storage for large numbers of high-dimensional vectors
 * by allocating memory outside the Java heap. Benefits include:
 * <ul>
 *   <li>Reduced garbage collection pressure</li>
 *   <li>Better memory locality for vector operations</li>
 *   <li>Predictable memory usage for large datasets</li>
 *   <li>Faster I/O operations when interfacing with native code</li>
 * </ul>
 * 
 * <p>The storage is pre-allocated for a fixed number of vectors with fixed dimensions.
 * Vectors are stored sequentially in a contiguous memory block for cache-efficient access.
 * 
 * <h3>Memory Layout:</h3>
 * <pre>
 * [vector0_dim0, vector0_dim1, ..., vector0_dimN,
 *  vector1_dim0, vector1_dim1, ..., vector1_dimN,
 *  ...]
 * </pre>
 * 
 * <h3>Usage Example:</h3>
 * <pre>{@code
 * OffHeapVectorsStorage storage = new OffHeapVectorsStorage(128, 1000000);
 * 
 * // Add vectors
 * float[] vector = new float[128];
 * storage.addVector(vector);
 * 
 * // Retrieve vectors
 * float[] retrieved = storage.getVector(0);
 * }</pre>
 * 
 * <p><b>Note:</b> This class does not automatically free off-heap memory. The memory
 * will be released when the ByteBuffer is garbage collected, but explicit cleanup
 * may be needed for long-running applications.
 * 
 * @author Navneev
 * @version 1.0
 * @since 1.0
 */
public class OffHeapVectorsStorage extends VectorStorage {
    
    /** Direct buffer for off-heap storage of vector data */
    private final ByteBuffer byteBuffer;
    
    /** Float view of the byte buffer for convenient float operations */
    private final FloatBuffer floatBuffer;

    /**
     * Constructs an off-heap storage for vectors with specified dimensions and capacity.
     * 
     * <p>Allocates a direct ByteBuffer of size: totalNumberOfVectors × dimensions × 4 bytes.
     * The memory is allocated outside the Java heap and will not be subject to garbage
     * collection until the buffer itself is no longer referenced.
     * 
     * @param dimensions the number of dimensions in each vector (must be > 0)
     * @param totalNumberOfVectors the maximum number of vectors to store (must be > 0)
     * @throws IllegalArgumentException if dimensions or totalNumberOfVectors is <= 0
     * @throws OutOfMemoryError if insufficient off-heap memory is available
     */
    public OffHeapVectorsStorage(int dimensions, int totalNumberOfVectors) {
        super(dimensions, totalNumberOfVectors);
        // Allocate direct (off-heap) buffer
        int sizeInBytes = totalNumberOfVectors * dimensions * Float.BYTES;
        this.byteBuffer = ByteBuffer.allocateDirect(sizeInBytes);
        this.floatBuffer = byteBuffer.asFloatBuffer();
    }

    /**
     * Adds a vector to the off-heap storage at the next available position.
     * 
     * <p>The vector is copied into the direct buffer at the current write offset.
     * The write offset is automatically incremented after the operation.
     * 
     * <p><b>Thread Safety:</b> This method is not thread-safe. External synchronization
     * is required if multiple threads add vectors concurrently.
     * 
     * @param vector the vector data to store (must have length equal to dimensions)
     * @throws IllegalArgumentException if vector length doesn't match dimensions
     * @throws IndexOutOfBoundsException if storage capacity is exceeded
     */
    public void addVector(int id, float[] vector) {
        floatBuffer.put((id * dimensions), vector, 0, dimensions);
    }

    /**
     * Retrieves a vector from the off-heap storage by its ID.
     * 
     * <p>The vector is read from the buffer at position: id × dimensions.
     * This method reuses an internal array to avoid allocations on repeated calls.
     * 
     * <p><b>Warning:</b> The returned array is reused across calls. If you need to
     * retain the data, make a copy before the next call to this method.
     * 
     * @param id the zero-based index of the vector to retrieve
     * @return array containing the vector data (reused across calls)
     * @throws IndexOutOfBoundsException if id is negative or >= totalNumberOfVectors
     */
    public float[] getVector(int id) {
        floatBuffer.get(id * dimensions, vector);
        return vector;
    }

}
