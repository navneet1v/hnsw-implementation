# Developer Guide

Comprehensive technical documentation for HNSW implementation developers.

## Table of Contents

1. [Project Structure](#project-structure)
2. [Core Classes](#core-classes)
3. [Algorithm Implementation](#algorithm-implementation)
4. [SIMD Optimization](#simd-optimization)
5. [Testing Strategy](#testing-strategy)
6. [HDF5 Integration](#hdf5-integration)
7. [Performance Optimization](#performance-optimization)
8. [Build Configuration](#build-configuration)
9. [Contributing Guidelines](#contributing-guidelines)

## Project Structure

```
src/
├── main/java/org/navneev/
│   ├── HNSWIndex.java          # Main HNSW implementation
│   ├── HNSWNode.java           # Graph node representation
│   ├── VectorUtils.java        # SIMD-optimized vector operations
│   ├── HDF5Reader.java         # HDF5 dataset reader
│   ├── Vector.java             # Vector data structure
│   └── Main.java               # Benchmark and testing
└── test/java/org/navneev/
    ├── HNSWIndexTest.java      # Unit tests for HNSW
    └── VectorUtilsTest.java    # Unit tests for vector operations
```

## Core Classes

### HNSWIndex.java

Main implementation of the HNSW algorithm:

- `addNode(float[] vector)` - Inserts new vectors into the graph
- `search(float[] query, int k, int efSearch)` - Performs k-NN search
- `searchLayer()` - Layer-specific search algorithm
- `selectNeighborsHeuristic()` - Connection pruning

**Key Parameters:**
```java
private int M = 16;              // Max connections per node
private int efConstruction = 100; // Construction search width
```

### VectorStorage (Abstract)

Base class for vector storage strategies:

- `addVector(int id, float[] vector)` - Store vector with ID
- `getVector(int id)` - Retrieve vector by ID

**Implementations:**

#### OnHeapVectorStorage
- Uses HashMap for storage
- Automatic memory management
- Best for datasets < 1M vectors
- Simple implementation

```java
VectorStorage storage = new OnHeapVectorStorage(dimensions, capacity);
```

#### OffHeapVectorsStorage
- Uses direct ByteBuffer
- Reduced GC pressure
- Better cache locality
- Requires explicit cleanup
- Best for datasets > 1M vectors

```java
OffHeapVectorsStorage storage = new OffHeapVectorsStorage(dimensions, capacity);
// ... use storage ...
storage.cleanup(); // Important: free memory
```

### HNSWNode.java

Represents a node in the HNSW graph:

- Stores node ID and level
- Maintains neighbor connections per layer
- Provides methods for adding/retrieving neighbors

### VectorUtils.java

SIMD-optimized vector operations:

- `euclideanDistance()` - SIMD-accelerated distance calculation
- `innerProduct()` - Dot product computation
- Uses JDK 23 Vector API for performance

### HDF5Reader.java

Utility for reading vector datasets:

- `readVectors()` - Reads training/test vectors
- `readGroundTruths()` - Reads ground truth neighbors
- `printDatasetInfo()` - Dataset exploration

## Algorithm Implementation

### Level Assignment

```java
private int getRandomLevel() {
    double prob = 1.0 / Math.log(M);
    int level = 0;
    while (random.nextDouble() < prob) {
        level++;
    }
    return level;
}
```

### Search Process

1. **Upper layers** - Greedy search to find entry point for layer 0
2. **Layer 0** - Beam search with ef candidates
3. **Result selection** - Return k closest candidates

### Connection Heuristic

The `selectNeighborsHeuristic` method implements diversity-based pruning:

- Selects diverse neighbors to avoid clustering
- Prevents creation of "hub" nodes
- Maintains graph connectivity

## SIMD Optimization

### Vector API Usage

```java
static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

public static float euclideanDistance(float[] a, float[] b) {
    int vectorLength = SPECIES.length();
    int loopBound = SPECIES.loopBound(a.length);
    double sum = 0;

    // SIMD loop
    for(int i = 0; i < loopBound; i += vectorLength) {
        FloatVector va = FloatVector.fromArray(SPECIES, a, i);
        FloatVector vb = FloatVector.fromArray(SPECIES, b, i);
        FloatVector diff = va.sub(vb);
        FloatVector squared = diff.mul(diff);
        sum += squared.reduceLanes(VectorOperators.ADD);
    }

    // Scalar remainder
    for(int i = loopBound; i < a.length; i++) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }

    return (float)sum;
}
```

### Performance Considerations

- **SIMD width** - Automatically detected by JVM
- **Memory alignment** - Handled by Vector API
- **Fallback** - Scalar operations for remainder elements

## Testing Strategy

### Unit Tests

- **VectorUtilsTest** - Tests SIMD operations
- **HNSWIndexTest** - Tests core HNSW functionality

### Integration Tests

- **Recall evaluation** - Compares against ground truth
- **Performance benchmarking** - Measures build/search times
- **Scalability testing** - Tests with different dataset sizes

### Test Data Generation

```java
private float[][] generateUniqueVectors(int numVectors, int dimensions, int seed) {
    float[][] vectors = new float[numVectors][dimensions];
    Random random = new Random(seed);
    
    for (int i = 0; i < numVectors; i++) {
        for (int j = 0; j < dimensions; j++) {
            vectors[i][j] = (float) (random.nextGaussian() * 0.5f);
        }
        normalizeVector(vectors[i]);
    }
    return vectors;
}
```

## HDF5 Integration

### Dataset Format

```
dataset.h5
├── train     # Training vectors [N x D]
├── test      # Query vectors [Q x D]  
└── neighbors # Ground truth [Q x K]
```

### Reading Data

```java
// Read training vectors
float[][] vectors = HDF5Reader.readVectors("dataset.h5", "train");

// Read ground truth
int[][] groundTruth = HDF5Reader.readGroundTruths("dataset.h5", "neighbors");
```

## Performance Optimization

### Vector Storage Selection

**Choose On-Heap when:**
- Dataset < 1M vectors
- Simplicity is priority
- GC pauses are acceptable
- Development/testing

**Choose Off-Heap when:**
- Dataset > 1M vectors
- Low GC pressure required
- Memory efficiency critical
- Production deployments

### Memory Management

**On-Heap:**
```java
VectorStorage storage = new OnHeapVectorStorage(128, 1000000);
// Automatic cleanup via GC
```

**Off-Heap:**
```java
OffHeapVectorsStorage storage = new OffHeapVectorsStorage(128, 1000000);
try {
    // Use storage
} finally {
    storage.cleanup(); // Explicit cleanup required
}
```

### Memory Management

- Use primitive arrays instead of wrapper classes
- Minimize object allocation in hot paths
- Consider memory pooling for frequent operations

### Distance Calculation Optimization

- Cache frequently computed distances
- Use SIMD operations for bulk calculations
- Consider approximate distance for pruning

### Graph Construction Optimization

- Batch node insertions when possible
- Use efficient data structures (BitSet for visited)
- Minimize priority queue operations
- Choose appropriate vector storage for dataset size

## Build Configuration

### Gradle Setup

```gradle
java {
    toolchain {
        languageVersion = JavaLanguageVersion.of(23)
    }
}

compileJava {
    options.compilerArgs += ['--add-modules', 'jdk.incubator.vector']
}

run {
    jvmArgs '--add-modules', 'jdk.incubator.vector'
}
```

### Dependencies

- **Lombok** - Code generation
- **jHDF** - HDF5 file support
- **JUnit 5** - Testing framework

## Contributing Guidelines

### Code Style

- Follow Java naming conventions
- Use meaningful variable names
- Add JavaDoc for public methods
- Keep methods focused and small

### Testing Requirements

- Write unit tests for new functionality
- Ensure all tests pass before submitting
- Add integration tests for major features
- Include performance regression tests

### Performance Considerations

- Profile before optimizing
- Measure impact of changes
- Consider algorithmic improvements over micro-optimizations
- Document performance characteristics

## Debugging and Profiling

### Common Issues

1. **Poor recall** - Check algorithm correctness, parameter tuning
2. **Slow performance** - Profile distance calculations, memory allocation
3. **Memory issues** - Monitor heap usage, optimize data structures

### Profiling Tools

- **JProfiler** - Memory and CPU profiling
- **JVM flags** - `-XX:+PrintGC`, `-XX:+PrintGCDetails`
- **Benchmarking** - Use JMH for micro-benchmarks

## Future Improvements

### Algorithmic Enhancements

- Dynamic parameter adjustment
- Multi-threaded construction
- Alternative distance metrics
- Graph compression techniques

### Implementation Optimizations

- Native memory management
- GPU acceleration
- Distributed index construction
- Incremental updates

## References

- [HNSW Paper](https://arxiv.org/abs/1603.09320)
- [JDK Vector API](https://openjdk.org/jeps/426)
- [HDF5 Documentation](https://www.hdfgroup.org/solutions/hdf5/)
- [Java Performance Tuning](https://docs.oracle.com/en/java/javase/21/gctuning/)