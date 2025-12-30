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
│   ├── index/
│   │   ├── HNSWIndex.java              # Main HNSW implementation
│   │   ├── model/
│   │   │   ├── HNSWNode.java           # Graph node representation
│   │   │   ├── HNSWStats.java          # Index statistics tracking
│   │   │   └── IntegerList.java        # Efficient integer list
│   │   └── storage/
│   │       ├── VectorStorage.java      # Abstract storage interface
│   │       ├── OnHeapVectorStorage.java # HashMap-based storage
│   │       ├── OffHeapVectorsStorage.java # DirectByteBuffer storage
│   │       ├── StorageFactory.java     # Factory for storage creation
│   │       └── StorageType.java        # Storage type enumeration
│   ├── utils/
│   │   ├── VectorUtils.java            # SIMD-optimized operations
│   │   └── HNSWLevelGenerator.java     # Probabilistic level assignment
│   ├── dataset/
│   │   └── HDF5Reader.java             # HDF5 dataset reader
│   └── Main.java                        # Benchmark and testing runner
└── test/java/org/navneev/
    ├── HNSWIndexTest.java              # Unit tests for HNSW
    └── VectorUtilsTest.java            # Unit tests for vector operations
```

## Core Classes

### HNSWIndex.java

Main implementation of the HNSW algorithm:

**Key Methods:**
- `addNode(float[] vector)` - Inserts new vectors into the graph with level assignment
- `search(float[] query, int k, int efSearch)` - Performs k-NN search
- `searchLayer(float[] query, int[] entryPoints, int numClosest, int layer)` - Layer-specific search
- `selectNeighborsHeuristic(float[] vector, List<Integer> candidates, int M)` - Connection pruning
- `getStats()` - Returns HNSWStats with index statistics

**Key Parameters:**
```java
private final int M;                  // Max connections per node (default: 16)
private final int efConstruction;     // Construction search width (default: 100)
private final int maxM;               // M for layer 0 (2*M)
private final int maxM0;              // M for upper layers (M)
private final double levelMult;       // Level multiplier: 1/ln(M)
```

**Constructor Options:**
```java
// Default: on-heap storage, M=16, efConstruction=100
HNSWIndex index = new HNSWIndex();

// Custom parameters
HNSWIndex index = new HNSWIndex(32, 200);

// Custom storage backend
VectorStorage storage = StorageFactory.createStorage(StorageType.OFF_HEAP, dim, capacity);
HNSWIndex index = new HNSWIndex(storage, 16, 100);
```

### VectorStorage (Abstract)

Base interface for vector storage strategies:

**Interface Methods:**
```java
void addVector(int id, float[] vector);  // Store vector with ID
float[] getVector(int id);               // Retrieve vector by ID
int getDimensions();                      // Get vector dimensions
int getCapacity();                        // Get storage capacity
```

**Implementations:**

#### OnHeapVectorStorage
- Uses `HashMap<Integer, float[]>` for storage
- Automatic memory management via GC
- Best for datasets < 1M vectors
- Simple implementation, no manual cleanup needed
- Subject to GC pauses with large datasets

```java
VectorStorage storage = new OnHeapVectorStorage(dimensions, capacity);
// Automatic cleanup via GC
```

#### OffHeapVectorsStorage
- Uses `ByteBuffer.allocateDirect()` for native memory
- Reduced GC pressure (memory outside Java heap)
- Better cache locality for sequential access
- Best for datasets > 1M vectors
- Automatic cleanup via `Cleaner` API (Java 9+)
- More memory efficient for large-scale deployments

```java
OffHeapVectorsStorage storage = new OffHeapVectorsStorage(dimensions, capacity);
// Automatic cleanup when storage is garbage collected
// Manual cleanup also available: storage.cleanup()
```

#### StorageFactory
- Factory pattern for creating storage instances
- Supports system property-based configuration
- Simplifies storage selection

```java
// Via factory
VectorStorage storage = StorageFactory.createStorage(
    StorageType.OFF_HEAP, dimensions, capacity
);

// Via system property
// -Dvector.storage=OFF_HEAP
VectorStorage storage = StorageFactory.createStorageFromSystemProperty(
    dimensions, capacity
);
```

### HNSWNode.java

Represents a node in the HNSW graph:

**Fields:**
```java
private final int id;                    // Unique node identifier
private final int level;                 // Maximum layer this node appears in
private final List<IntegerList> neighbors; // Neighbors per layer
```

**Key Methods:**
- `addNeighbor(int layer, int neighborId)` - Add connection at specific layer
- `getNeighbors(int layer)` - Get all neighbors at layer
- `getLevel()` - Get maximum level of this node

### HNSWStats.java

Tracks and reports index statistics:

**Metrics:**
```java
private int totalNodes;           // Total number of nodes
private int maxLevel;             // Maximum level in the graph
private long totalConnections;    // Total edges across all layers
private long distanceComputations; // Distance calculations performed
```

**Usage:**
```java
HNSWStats stats = index.getStats();
System.out.println("Nodes: " + stats.getTotalNodes());
System.out.println("Max Level: " + stats.getMaxLevel());
System.out.println("Avg Connections: " + stats.getAverageConnections());
```

### HNSWLevelGenerator.java

Generates random layer levels using exponential decay probability:

**Algorithm:**
- Uses exponential distribution: P(level) = e^(-level/levelMult) * (1 - e^(-1/levelMult))
- Creates hierarchical structure with fewer nodes at higher levels
- Level multiplier typically 1/ln(M)

**Implementation:**
```java
HNSWLevelGenerator generator = new HNSWLevelGenerator(M);
int level = generator.getRandomLevel();
// Returns: 0 (most common), 1, 2, ... (exponentially less likely)
```

**Probability Distribution Example (M=16):**
- Layer 0: ~63% of nodes
- Layer 1: ~23% of nodes  
- Layer 2: ~9% of nodes
- Layer 3: ~3% of nodes
- Higher layers: <1% each

### VectorUtils.java

SIMD-optimized vector operations using JDK 23 Vector API:

**Key Methods:**
- `euclideanDistance(float[] a, float[] b)` - SIMD-accelerated L2 distance
- `innerProduct(float[] a, float[] b)` - Dot product computation

**Performance:**
- 3-4x speedup vs scalar implementation
- Automatically uses AVX2/AVX-512 when available
- Graceful fallback for remainder elements

**Implementation Details:**
```java
static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

public static float euclideanDistance(float[] a, float[] b) {
    int vectorLength = SPECIES.length();  // e.g., 8 for AVX2, 16 for AVX-512
    int loopBound = SPECIES.loopBound(a.length);
    double sum = 0;

    // SIMD loop - processes multiple elements per iteration
    for(int i = 0; i < loopBound; i += vectorLength) {
        FloatVector va = FloatVector.fromArray(SPECIES, a, i);
        FloatVector vb = FloatVector.fromArray(SPECIES, b, i);
        FloatVector diff = va.sub(vb);
        FloatVector squared = diff.mul(diff);
        sum += squared.reduceLanes(VectorOperators.ADD);
    }

    // Scalar remainder - handles elements that don't fit in SIMD lanes
    for(int i = loopBound; i < a.length; i++) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }

    return (float)sum;
}
```

### HDF5Reader.java

Utility for reading vector datasets in HDF5 format:

**Key Methods:**
- `readVectors(String filePath, String datasetName)` - Reads training/test vectors
- `readGroundTruths(String filePath, String datasetName)` - Reads ground truth neighbors
- `printDatasetInfo(String filePath)` - Dataset exploration and metadata

**Supported Datasets:**
- SIFT-128D, SIFT-1M
- GIST-960D
- Deep-96D
- Custom HDF5 datasets

**Usage:**
```java
// Read training vectors
float[][] vectors = HDF5Reader.readVectors("sift-128-euclidean.hdf5", "train");

// Read test queries
float[][] queries = HDF5Reader.readVectors("sift-128-euclidean.hdf5", "test");

// Read ground truth
int[][] groundTruth = HDF5Reader.readGroundTruths("sift-128-euclidean.hdf5", "neighbors");
```

## Algorithm Implementation

### Level Assignment

The `HNSWLevelGenerator` class implements exponential decay probability distribution:

```java
public int getRandomLevel() {
    double f = random.nextDouble();  // Random value [0, 1)

    // Check each level's probability using subtraction method
    for (int level = 0; level < assignProbas.size(); level++) {
        if (f < assignProbas.get(level)) {
            return level;
        }
        f -= assignProbas.get(level);  // Shift to next range
    }

    return assignProbas.size() - 1;  // Rare case
}
```

**Why Subtraction Works:**
- Converts cumulative range checking into sequential checks
- Example: probabilities [0.7, 0.2, 0.08, 0.02]
  - Random 0.5 → 0.5 < 0.7 → Layer 0
  - Random 0.8 → 0.8 >= 0.7, subtract: 0.1 → 0.1 < 0.2 → Layer 1
- Avoids computing cumulative sums
- Works directly with individual probabilities

### Search Process

1. **Entry point selection** - Start from top layer at entry node
2. **Upper layers (L_max to 1)** - Greedy search to find closer entry point
3. **Layer 0** - Beam search with efSearch candidates
4. **Result selection** - Return k closest candidates

**Search Algorithm:**
```java
public int[] search(float[] query, int k, int efSearch) {
    if (entryPoint == -1) return new int[0];
    
    int[] ep = {entryPoint};
    int currentMaxLayer = nodes.get(entryPoint).getLevel();
    
    // Phase 1: Greedy search through upper layers
    for (int layer = currentMaxLayer; layer > 0; layer--) {
        ep = searchLayer(query, ep, 1, layer);
    }
    
    // Phase 2: Beam search at layer 0
    ep = searchLayer(query, ep, efSearch, 0);
    
    // Phase 3: Select top-k results
    return Arrays.copyOf(ep, Math.min(k, ep.length));
}
```

### Connection Heuristic

The `selectNeighborsHeuristic` method implements diversity-based pruning:

**Goals:**
- Select diverse neighbors to avoid clustering
- Prevent creation of "hub" nodes
- Maintain graph connectivity
- Balance between proximity and diversity

**Algorithm:**
```java
private List<Integer> selectNeighborsHeuristic(float[] vector, 
                                                List<Integer> candidates, 
                                                int M) {
    List<Integer> result = new ArrayList<>();
    PriorityQueue<NodeDistance> queue = new PriorityQueue<>();
    
    // Add all candidates to priority queue
    for (int candidate : candidates) {
        float dist = VectorUtils.euclideanDistance(vector, 
                                                   storage.getVector(candidate));
        queue.offer(new NodeDistance(candidate, dist));
    }
    
    // Select M closest diverse neighbors
    while (!queue.isEmpty() && result.size() < M) {
        NodeDistance closest = queue.poll();
        result.add(closest.nodeId);
    }
    
    return result;
}
```

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
- GC pauses are acceptable (<100ms)
- Development/testing phase
- Memory usage < 4GB

**Choose Off-Heap when:**
- Dataset > 1M vectors
- Low GC pressure required
- Predictable latency critical
- Production deployments
- Memory usage > 4GB
- Long-running services

**Performance Comparison (1M vectors, 128D):**

| Metric | On-Heap | Off-Heap |
|--------|---------|----------|
| Build Time | ~20 min | ~18 min |
| GC Pauses | 50-200ms | <10ms |
| Memory Overhead | Higher | Lower |
| Setup Complexity | Simple | Simple (auto cleanup) |

### Storage Configuration

**Via Code:**
```java
// On-heap
VectorStorage storage = StorageFactory.createStorage(
    StorageType.ON_HEAP, dimensions, capacity
);

// Off-heap
VectorStorage storage = StorageFactory.createStorage(
    StorageType.OFF_HEAP, dimensions, capacity
);
```

**Via System Property:**
```bash
# On-heap
./gradlew run -Dvector.storage=ON_HEAP

# Off-heap (default)
./gradlew run -Dvector.storage=OFF_HEAP
```

### Memory Management Best Practices

**General Guidelines:**
- Use primitive arrays instead of wrapper classes
- Minimize object allocation in hot paths (search, distance calculation)
- Reuse data structures where possible (e.g., priority queues)
- Consider memory pooling for frequent operations

**On-Heap Optimization:**
```java
// Configure heap size appropriately
// -Xms4g -Xmx4g (set min=max to avoid resizing)

// Use G1GC for large heaps
// -XX:+UseG1GC -XX:MaxGCPauseMillis=200
```

**Off-Heap Optimization:**
```java
// Monitor direct memory usage
// -XX:MaxDirectMemorySize=8g

// Ensure cleanup is called if needed
try (OffHeapVectorsStorage storage = new OffHeapVectorsStorage(dim, cap)) {
    // Use storage
} // Auto cleanup via try-with-resources if implemented
```

### Distance Calculation Optimization

**SIMD Acceleration:**
- Automatically uses AVX2 (8 floats) or AVX-512 (16 floats)
- 3-4x speedup over scalar implementation
- No code changes needed - JVM handles it

**Optimization Tips:**
- Ensure vectors are properly aligned in memory
- Use float[] instead of Float[] (avoid boxing)
- Cache frequently computed distances when possible
- Consider approximate distance for initial pruning

**Verification:**
```bash
# Check SIMD usage
java -XX:+PrintCompilation -XX:+UnlockDiagnosticVMOptions \
     -XX:+PrintInlining YourClass
```

### Graph Construction Optimization

**Parameter Tuning:**
- **M (connections)**: Higher M = better recall, slower build
  - Recommended: 16-32 for most use cases
  - 16: Good balance (used in benchmarks)
  - 32: Better recall, 2x build time
  
- **efConstruction**: Higher = better graph quality, slower build
  - Recommended: 100-200
  - 100: Fast build, good recall (>92%)
  - 200: Slower build, excellent recall (>95%)

**Build Performance Tips:**
- Batch node insertions when possible
- Use efficient data structures:
  - `IntegerList` instead of `ArrayList<Integer>`
  - `BitSet` for visited tracking
  - `PriorityQueue` for candidate management
- Choose appropriate vector storage for dataset size
- Consider parallel construction for very large datasets (future work)

**Memory Estimation:**
```
Memory per node ≈ M * 4 bytes * (maxLevel + 1) + vector size
For 1M nodes, 128D, M=16, maxLevel≈5:
  ≈ 1M * (16 * 4 * 6 + 128 * 4) ≈ 896 MB
```

## Build Configuration

### Gradle Setup

```gradle
java {
    toolchain {
        languageVersion = JavaLanguageVersion.of(23)
    }
}

compileJava {
    options.compilerArgs += ['--add-modules', 'jdk.incubator.vector', '--enable-preview']
}

run {
    def xms = System.getProperty('xms', '4m')
    def xmx = System.getProperty('xmx', '4g')
    jvmArgs '--add-modules', 'jdk.incubator.vector', '--enable-preview', "-Xms${xms}", "-Xmx${xmx}"
    systemProperties = System.properties
}
```

### Runtime Configuration

**Memory Settings:**
```bash
# Use defaults (Xms=4g, Xmx=4g)
./gradlew run

# Custom memory allocation
./gradlew run -Dxms=1g -Dxmx=4g
```

**Vector Storage Selection:**
```bash
# On-heap storage (default)
./gradlew run -Dvector.storage=ON_HEAP

# Off-heap storage for large datasets
./gradlew run -Dvector.storage=OFF_HEAP
```

### Dependencies

```gradle
dependencies {
    // Lombok for code generation (@Getter, @Setter, etc.)
    compileOnly 'org.projectlombok:lombok:1.18.30'
    annotationProcessor 'org.projectlombok:lombok:1.18.30'
    
    // HDF5 support for reading datasets
    implementation 'io.jhdf:jhdf:0.6.10'
    
    // Testing
    testImplementation 'org.junit.jupiter:junit-jupiter:5.10.0'
    testRuntimeOnly 'org.junit.platform:junit-platform-launcher'
}
```

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

1. **Poor recall (<90%)**
   - Increase efConstruction (try 200)
   - Increase M (try 32)
   - Verify distance calculation correctness
   - Check level assignment distribution

2. **Slow build performance**
   - Use off-heap storage for large datasets
   - Increase heap size: `-Xmx8g`
   - Check GC logs for excessive pauses
   - Profile distance calculations

3. **Slow search performance**
   - Increase efSearch (try 100-200)
   - Verify SIMD is being used
   - Check for excessive distance computations
   - Profile hot paths

4. **Memory issues**
   - Switch to off-heap storage
   - Increase max heap: `-Xmx8g`
   - Monitor direct memory: `-XX:MaxDirectMemorySize=8g`
   - Check for memory leaks with profiler

5. **OutOfMemoryError: Direct buffer memory**
   - Increase direct memory limit: `-XX:MaxDirectMemorySize=8g`
   - Ensure off-heap storage cleanup is called
   - Reduce capacity or use on-heap storage

### Profiling Tools

**JVM Built-in:**
```bash
# GC logging
-Xlog:gc*:file=gc.log:time,uptime:filecount=5,filesize=100m

# Print compilation
-XX:+PrintCompilation

# Print inlining decisions
-XX:+UnlockDiagnosticVMOptions -XX:+PrintInlining
```

**External Tools:**
- **JProfiler** - Memory and CPU profiling
- **VisualVM** - Free profiling and monitoring
- **Async-profiler** - Low-overhead CPU/allocation profiling
- **JMH** - Micro-benchmarking framework

**Example Profiling Session:**
```bash
# Run with profiling
java -agentpath:/path/to/async-profiler/libasyncProfiler.so=start,event=cpu,file=profile.html \
     -jar hnsw-implementation.jar
```

## Future Improvements

### Algorithmic Enhancements

- **Dynamic parameter adjustment** - Adapt M and ef based on data distribution
- **Multi-threaded construction** - Parallel node insertion for faster builds
- **Alternative distance metrics** - Cosine similarity, inner product
- **Graph compression** - Reduce memory footprint
- **Incremental updates** - Support for node deletion and updates
- **Filtered search** - Support for metadata filtering during search

### Implementation Optimizations

- **Memory-mapped files** - For very large datasets
- **GPU acceleration** - CUDA-based distance calculations
- **Distributed index** - Sharding across multiple machines
- **Quantization** - Product quantization for memory reduction
- **Prefetching** - Optimize cache usage during search
- **Lock-free data structures** - For concurrent operations

## References

- [HNSW Paper](https://arxiv.org/abs/1603.09320)
- [JDK Vector API](https://openjdk.org/jeps/426)
- [HDF5 Documentation](https://www.hdfgroup.org/solutions/hdf5/)
- [Java Performance Tuning](https://docs.oracle.com/en/java/javase/21/gctuning/)