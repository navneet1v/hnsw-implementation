package org.navneev;

import lombok.Getter;

/**
 * Abstract base class for vector storage implementations.
 * 
 * <p>This class defines the contract for storing and retrieving high-dimensional vectors
 * in the HNSW index. Implementations can choose different storage strategies based on
 * performance requirements and memory constraints:
 * <ul>
 *   <li><b>On-heap storage</b> - Uses Java heap memory (HashMap-based)</li>
 *   <li><b>Off-heap storage</b> - Uses direct ByteBuffers for reduced GC pressure</li>
 *   <li><b>Memory-mapped storage</b> - Uses file-backed memory for very large datasets</li>
 * </ul>
 * 
 * <p>All implementations must support:
 * <ul>
 *   <li>Adding vectors with unique integer IDs</li>
 *   <li>Retrieving vectors by ID</li>
 *   <li>Fixed dimensionality for all vectors</li>
 *   <li>Pre-defined capacity (total number of vectors)</li>
 * </ul>
 * 
 * <h3>Design Considerations:</h3>
 * <p>Subclasses should consider:
 * <ul>
 *   <li><b>Thread safety</b> - Whether concurrent access is supported</li>
 *   <li><b>Memory efficiency</b> - Heap vs off-heap trade-offs</li>
 *   <li><b>Access patterns</b> - Sequential vs random access optimization</li>
 *   <li><b>Cleanup</b> - Whether explicit resource cleanup is needed</li>
 * </ul>
 * 
 * <h3>Usage Example:</h3>
 * <pre>{@code
 * VectorStorage storage = new OnHeapVectorStorage(128, 100000);
 * 
 * // Add vectors
 * float[] vector = new float[128];
 * storage.addVector(0, vector);
 * 
 * // Retrieve vectors
 * float[] retrieved = storage.getVector(0);
 * }</pre>
 * 
 * @author Navneev
 * @version 1.0
 * @since 1.0
 * @see OnHeapVectorStorage
 * @see OffHeapVectorsStorage
 */
public abstract class VectorStorage {

    /** Number of dimensions in each vector */
    protected final int dimensions;

    /** Total capacity of vectors that can be stored */
    @Getter
    protected final int totalNumberOfVectors;

    /** Reusable array for vector retrieval to avoid allocations */
    protected final float[] vector;

    /**
     * Constructs a vector storage with specified dimensions and capacity.
     * 
     * <p>Initializes common fields used by all storage implementations:
     * <ul>
     *   <li>Dimensions - fixed size for all vectors</li>
     *   <li>Total capacity - maximum number of vectors</li>
     *   <li>Reusable array - for efficient vector retrieval</li>
     * </ul>
     * 
     * @param dimensions the number of dimensions in each vector (must be > 0)
     * @param totalNumberOfVectors the maximum number of vectors to store (must be > 0)
     * @throws IllegalArgumentException if dimensions or totalNumberOfVectors is <= 0
     */
    public VectorStorage(int dimensions, int totalNumberOfVectors) {
        this.dimensions = dimensions;
        this.totalNumberOfVectors = totalNumberOfVectors;
        this.vector = new float[dimensions];
    }

    /**
     * Adds a vector to the storage with the specified ID.
     * 
     * <p>Implementations must handle:
     * <ul>
     *   <li>Validating vector dimensions match the storage dimensions</li>
     *   <li>Storing the vector data (copy or reference as appropriate)</li>
     *   <li>Handling duplicate IDs (replace or error)</li>
     * </ul>
     * 
     * <p><b>Thread Safety:</b> Implementations should document their thread safety guarantees.
     * 
     * @param id the unique identifier for this vector (typically 0-based sequential)
     * @param vector the vector data to store (length must equal dimensions)
     * @throws IllegalArgumentException if vector length doesn't match dimensions
     * @throws IndexOutOfBoundsException if storage capacity is exceeded
     * @throws NullPointerException if vector is null
     */
    public abstract void addVector(int id, float[] vector);

    /**
     * Retrieves a vector from the storage by its ID.
     * 
     * <p>Implementations may:
     * <ul>
     *   <li>Return a direct reference to stored data (fast but mutable)</li>
     *   <li>Return a copy of the data (safe but slower)</li>
     *   <li>Reuse a shared array (fastest but not thread-safe)</li>
     * </ul>
     * 
     * <p>Callers should check implementation documentation for:
     * <ul>
     *   <li>Whether the returned array can be modified</li>
     *   <li>Whether the array is reused across calls</li>
     *   <li>Thread safety guarantees</li>
     * </ul>
     * 
     * @param id the unique identifier of the vector to retrieve
     * @return the vector data, or null if no vector exists with the given ID
     * @throws IndexOutOfBoundsException if id is negative or exceeds capacity
     */
    public abstract float[] getVector(int id);

}
