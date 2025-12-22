# HNSW Implementation in Java

A high-performance implementation of Hierarchical Navigable Small World (HNSW) algorithm for approximate nearest neighbor search, featuring SIMD optimization and HDF5 dataset support.

## Overview

This project implements the HNSW algorithm as described in the paper "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs" by Malkov and Yashunin (arXiv:1603.09320). The implementation includes:

- **SIMD-optimized vector operations** using JDK 23 Vector API
- **HDF5 dataset support** for large-scale vector datasets
- **Comprehensive testing suite** with recall evaluation
- **Performance benchmarking** with percentile analysis

## Features

- ✅ Multi-layer graph structure with probabilistic level assignment
- ✅ SIMD-accelerated Euclidean distance calculations
- ✅ HDF5 file reading for datasets (SIFT, GIST, etc.)
- ✅ Configurable parameters (M, efConstruction, efSearch)
- ✅ Recall evaluation against ground truth
- ✅ Performance metrics (build time, search time percentiles)

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

### 3. Run with HDF5 Dataset
```bash
# Download SIFT dataset (example)
wget http://corpus-texmex.irisa.fr/sift.tar.gz
tar -xzf sift.tar.gz

# Update file path in Main.java
# Then run:
./gradlew run
```

### 4. Run Tests
```bash
# Run all tests
./gradlew test

# Run specific tests
./gradlew test --tests HNSWIndexTest
```

## Basic Usage

```java
HNSWIndex index = new HNSWIndex();

// Add vectors
float[] vector1 = {1.0f, 2.0f, 3.0f};
float[] vector2 = {4.0f, 5.0f, 6.0f};
index.addNode(vector1);
index.addNode(vector2);

// Search
float[] query = {1.1f, 2.1f, 3.1f};
int[] results = index.search(query, k=5, efSearch=50);
```

## Performance

- **Build time**: ~1-2 seconds per 100K vectors
- **Search time**: <1ms per query (P50)
- **Recall@10**: >80% with proper parameters

## Documentation

- [Developer Guide](DEVELOPER_GUIDE.md) - Detailed implementation guide
- [API Documentation](docs/) - JavaDoc documentation

## License

Apache License 2.0 - see LICENSE file for details.

## References

- Malkov, Yu A., and D. A. Yashunin. "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs." arXiv:1603.09320 (2016).
- [JDK Vector API Documentation](https://openjdk.org/jeps/426)
- [HDF5 Java Library](https://github.com/HDFGroup/hdf5)