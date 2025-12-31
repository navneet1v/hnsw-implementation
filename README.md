# HNSW Implementation in Java

A high-performance implementation of Hierarchical Navigable Small World (HNSW) algorithm for approximate nearest neighbor search, featuring SIMD optimization and HDF5 dataset support.

## Overview

This project implements the HNSW algorithm as described in the paper "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs" by Malkov and Yashunin (arXiv:1603.09320). The implementation includes:

- **SIMD-optimized vector operations** using JDK 23 Vector API for fast distance calculations
- **HDF5 dataset support** for large-scale vector datasets (SIFT, GIST, etc.)
- **Pluggable storage backends** with on-heap and off-heap implementations
- **Comprehensive testing suite** with recall evaluation against ground truth
- **Performance benchmarking** with detailed statistics and percentile analysis
- **Production-ready code** with extensive JavaDoc documentation

## Features

- ✅ Multi-layer graph structure with exponential decay level assignment
- ✅ SIMD-accelerated Euclidean distance calculations (JDK 23 Vector API)
- ✅ Pluggable vector storage backends:
  - On-heap storage (HashMap-based, automatic GC)
  - Off-heap storage (DirectByteBuffer, reduced GC pressure)
- ✅ HDF5 file reading for standard datasets (SIFT, GIST, etc.)
- ✅ Configurable HNSW parameters (M, efConstruction, efSearch)
- ✅ Comprehensive statistics tracking (HNSWStats)
- ✅ Recall evaluation against ground truth
- ✅ Performance metrics (build time, search time, percentiles)
- ✅ Factory pattern for storage selection
- ✅ Extensive JavaDoc documentation

## Requirements

- **Java 23+** (required for Vector API)
- **Gradle 8.0+**
- **HDF5 dataset** (optional, for benchmarking)

## Quick Start

### 1. Clone and Build
```bash
git clone <repository-url>
cd hnsw-implementation
./gradlew build
```

### 2. Run with Synthetic Data
```bash
./gradlew run
```

### 3. Run with Custom Configuration
```bash
# Configure memory (defaults: Xms=512m, Xmx=2g)
./gradlew run -Dxms=1g -Dxmx=4g

# Configure vector storage (ON_HEAP or OFF_HEAP)
./gradlew run -Dvector.storage=OFF_HEAP

# Pass custom system properties
./gradlew run -Dfile.path=/path/to/data.h5 -Ddataset.name=train

# Combine multiple parameters
./gradlew run -Dxms=2g -Dxmx=8g -Dvector.storage=OFF_HEAP
```

### 4. Run with HDF5 Dataset
```bash
# Download SIFT dataset (example)
wget http://corpus-texmex.irisa.fr/sift.tar.gz
tar -xzf sift.tar.gz

# Run with dataset path
./gradlew run -Dfile.path=sift/sift_base.hdf5
```

### 5. Run Tests
```bash
# Run all tests
./gradlew test

# Run specific tests
./gradlew test --tests HNSWIndexTest
```

## Configuration Parameters

### Gradle Runtime Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `-Dxms` | 4g      | Initial heap size |
| `-Dxmx` | 4g      | Maximum heap size |
| `-Dvector.storage` | OFF_HEAP | Vector storage type (ON_HEAP or OFF_HEAP) |

**Example:**
```bash
./gradlew run -Dxms=1g -Dxmx=4g -Dvector.storage=OFF_HEAP
```

## Basic Usage

```java
import org.navneev.index.HNSWIndex;
import org.navneev.index.storage.*;

// Create index with default parameters (M=16, efConstruction=100)
HNSWIndex index = new HNSWIndex();

// Or create with custom parameters
HNSWIndex customIndex = new HNSWIndex(32, 200); // M=32, efConstruction=200

// Or use off-heap storage for large datasets
VectorStorage storage = StorageFactory.createStorage(
    StorageType.OFF_HEAP, dimensions, capacity
);
HNSWIndex offHeapIndex = new HNSWIndex(storage, 16, 100);

// Add vectors
float[] vector1 = {1.0f, 2.0f, 3.0f};
float[] vector2 = {4.0f, 5.0f, 6.0f};
index.addNode(vector1);
index.addNode(vector2);

// Search for k nearest neighbors
float[] query = {1.1f, 2.1f, 3.1f};
int[] results = index.search(query, 5, 50); // k=5, efSearch=50

// Get index statistics
HNSWStats stats = index.getStats();
System.out.println("Total nodes: " + stats.getTotalNodes());
System.out.println("Max level: " + stats.getMaxLevel());
```

## Vector Storage Options

### On-Heap Storage
- Uses HashMap for vector storage
- Automatic memory management via GC
- Best for small to medium datasets (<1M vectors)
- Simple and straightforward

```java
VectorStorage storage = StorageFactory.createStorage(
    StorageType.ON_HEAP, dimensions, capacity
);
HNSWIndex index = new HNSWIndex(storage, M, efConstruction);
```

### Off-Heap Storage
- Uses direct ByteBuffer (native memory)
- Reduced GC pressure and pauses
- Better cache locality
- Best for large datasets (>1M vectors)
- Automatic cleanup via Cleaner API

```java
VectorStorage storage = StorageFactory.createStorage(
    StorageType.OFF_HEAP, dimensions, capacity
);
HNSWIndex index = new HNSWIndex(storage, M, efConstruction);
// Automatic cleanup when storage is garbage collected
```

### Storage Selection via System Property

```bash
# Use on-heap storage
./gradlew run -Dvector.storage=ON_HEAP

# Use off-heap storage (default)
./gradlew run -Dvector.storage=OFF_HEAP
```

## Performance

All performance tests conducted on SIFT-128D dataset:

### Build Performance
- **Dataset**: 1M vectors, 128 dimensions
- **Build time**: ~6.8 minutes
- **Parameters**: M=16, efConstruction=100
- **Storage**: Off-heap (reduced GC overhead)
- **Shrink Algorithm**: Greedy Neighbors Shrink (Faster build times)

### Build Times with Different Approaches with 1M dataset
- 18 mins with no neighbor pruning, using Integer and pretty bad code
- 9 mins with neighbor pruning
- 6.8 mins with native byte ordering 
- 6.2 mins with Memory segment + FromArray for some vectors
- 5.8 mins changing everything to use Memory segment
- 6.08 mins with Memory Segment but adding few classes on top

### Search Performance
- **Latency (P50)**: ~0.5ms per query
- **Latency (P99)**: ~1ms per query
- **Parameters**: efSearch=100

### Accuracy
- **Recall@100**: >87% (efSearch=100, M=16)

### SIMD Acceleration
- **Distance calculation speedup**: 3-4x vs scalar implementation
- **Vector API**: Automatically uses AVX2/AVX-512 when available

## Project Structure

```
src/main/java/org/navneev/
├── index/
│   ├── HNSWIndex.java              # Main HNSW implementation
│   ├── model/
│   │   ├── HNSWNode.java           # Graph node representation
│   │   ├── HNSWStats.java          # Index statistics
│   │   └── IntegerList.java        # Efficient int list
│   └── storage/
│       ├── VectorStorage.java      # Storage interface
│       ├── OnHeapVectorStorage.java
│       ├── OffHeapVectorsStorage.java
│       ├── StorageFactory.java     # Factory for storage creation
│       └── StorageType.java        # Storage type enum
├── utils/
│   ├── VectorUtils.java            # SIMD vector operations
│   └── HNSWLevelGenerator.java     # Level assignment logic
├── dataset/
│   └── HDF5Reader.java             # HDF5 file reader
└── Main.java                        # Benchmark runner
```

## Documentation

- [Developer Guide](DEVELOPER_GUIDE.md) - Detailed implementation guide
- [JavaDoc](src/main/java/) - Comprehensive inline documentation
- [HNSW Paper](https://arxiv.org/abs/1603.09320) - Original algorithm paper

## License

Apache License 2.0 - see LICENSE file for details.

## References

- Malkov, Yu A., and D. A. Yashunin. "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs." arXiv:1603.09320 (2016).
- [JDK Vector API Documentation](https://openjdk.org/jeps/426)
- [HDF5 Java Library](https://github.com/HDFGroup/hdf5)