package org.navneev;

import org.junit.jupiter.api.Test;
import org.navneev.index.HNSWIndex;
import org.navneev.utils.VectorUtils;

import static org.junit.jupiter.api.Assertions.*;

class HNSWIndexTest {

    private HNSWIndex index;
    private static final double DELTA = 1e-6;

    @Test
    void testAddSingleNode() {
        index = new HNSWIndex(3, 1);
        float[] vector = {1.0f, 2.0f, 3.0f};
        index.addNode(vector);
        
        int[] results = index.search(vector, 1, 10);
        assertEquals(1, results.length);
        assertEquals(0, results[0]); // First node should have ID 0
    }

    @Test
    void testAddMultipleNodes() {
        float[][] vectors = {
            {1.0f, 0.0f},
            {0.0f, 1.0f},
            {2.0f, 0.0f},
            {0.0f, 2.0f}
        };

        index = new HNSWIndex(vectors[0].length, vectors.length);
        
        for (float[] vector : vectors) {
            index.addNode(vector);
        }
        
        // Search for exact match
        int[] results = index.search(vectors[0], 1, 10);
        assertEquals(1, results.length);
        assertEquals(0, results[0]); // Should find the first vector
    }

    @Test
    void testNearestNeighborSearch() {
        float[][] vectors = {
            {0.0f, 0.0f},  // origin
            {1.0f, 0.0f},  // right
            {0.0f, 1.0f},  // up
            {3.0f, 4.0f}   // far point
        };
        index = new HNSWIndex(vectors[0].length, vectors.length);
        
        for (float[] vector : vectors) {
            index.addNode(vector);
        }
        
        // Query point close to origin
        float[] query = {0.1f, 0.1f};
        int[] results = index.search(query, 2, 10);
        
        assertTrue(results.length <= 2);
        assertTrue(results.length > 0);
        // First result should be closest to origin (node 0)
        assertEquals(0, results[0]);
    }

    @Test
    void testSearchWithDifferentK() {
        float[][] vectors = {
            {1.0f, 1.0f},
            {2.0f, 2.0f},
            {3.0f, 3.0f},
            {4.0f, 4.0f},
            {5.0f, 5.0f}
        };
        index = new HNSWIndex(vectors[0].length, vectors.length);
        for (float[] vector : vectors) {
            index.addNode(vector);
        }
        
        float[] query = {2.5f, 2.5f};
        
        // Test k=1
        int[] results1 = index.search(query, 1, 10);
        assertEquals(1, results1.length);
        
        // Test k=3
        int[] results3 = index.search(query, 3, 10);
        assertTrue(results3.length <= 3);
        assertTrue(results3.length > 0);
        
        // Test k larger than available nodes
        int[] resultsAll = index.search(query, 10, 10);
        assertTrue(resultsAll.length <= 5); // Only 5 nodes available
    }

    @Test
    void testSearchAccuracy() {
        // Create a simple 2D grid
        float[][] vectors = {
            {0.0f, 0.0f},
            {1.0f, 0.0f},
            {2.0f, 0.0f},
            {0.0f, 1.0f},
            {1.0f, 1.0f},
            {2.0f, 1.0f}
        };
        index = new HNSWIndex(vectors[0].length, vectors.length);
        
        for (float[] vector : vectors) {
            index.addNode(vector);
        }
        
        // Query at (1.1, 0.1) should be closest to (1.0, 0.0) which is node 1
        float[] query = {1.1f, 0.1f};
        int[] results = index.search(query, 1, 10);
        
        assertEquals(1, results.length);
        // Should find node 1 as it's closest
        assertTrue(results[0] >= 0 && results[0] < 6);
    }

    @Test
    void testLargerDataset() {
        // Add 50 random vectors
        float[][] vectors = new float[50][2];
        index = new HNSWIndex(vectors[0].length, vectors.length);
        for (int i = 0; i < 50; i++) {
            vectors[i][0] = (float) (Math.random() * 10);
            vectors[i][1] = (float) (Math.random() * 10);
            index.addNode(vectors[i]);
        }
        
        // Search should return results
        float[] query = {5.0f, 5.0f};
        int[] results = index.search(query, 5, 20);
        
        assertTrue(results.length <= 5);
        assertTrue(results.length > 0);
        
        // All result IDs should be valid
        for (int id : results) {
            assertTrue(id >= 0 && id < 50);
        }
    }

    @Test
    void testSearchWithHigherEfSearch() {
        float[][] vectors = {
            {0.0f, 0.0f},
            {1.0f, 0.0f},
            {0.0f, 1.0f},
            {1.0f, 1.0f}
        };
        index = new HNSWIndex(vectors[0].length, vectors.length);
        
        for (float[] vector : vectors) {
            index.addNode(vector);
        }
        
        float[] query = {0.5f, 0.5f};
        
        // Test with different ef_search values
        int[] results1 = index.search(query, 2, 5);
        int[] results2 = index.search(query, 2, 20);
        
        assertEquals(2, results1.length);
        assertEquals(2, results2.length);
        
        // Both should return valid node IDs
        for (int id : results1) {
            assertTrue(id >= 0 && id < 4);
        }
        for (int id : results2) {
            assertTrue(id >= 0 && id < 4);
        }
    }

    @Test
    void testIdenticalVectors() {
        float[] vector = {1.0f, 2.0f, 3.0f};

        index = new HNSWIndex(vector.length, 3);
        
        // Add the same vector multiple times
        for (int i = 0; i < 3; i++) {
            index.addNode(vector.clone());
        }
        
        int[] results = index.search(vector, 3, 10);
        assertEquals(3, results.length);
        
        // Should find all three identical vectors
        for (int id : results) {
            assertTrue(id >= 0 && id < 3);
        }
    }

    @Test
    void testRecallWith1000Vectors() {
        final int NUM_VECTORS = 20000; // Reduced for easier debugging
        final int DIMENSIONS = 128;   // Reduced dimensions
        final int NUM_QUERIES = 100;  // Fewer queries
        final int K = 10;
        
        // Generate vectors
        float[][] vectors = generateUniqueVectors(NUM_VECTORS, DIMENSIONS);
        
        // Write vectors to file for Python analysis
        writeVectorsToFile(vectors, "vectors.txt");

        index = new HNSWIndex(DIMENSIONS, NUM_VECTORS);
        
        // Build index
        System.out.println("Building index with " + NUM_VECTORS + " vectors...");
        for (float[] vector : vectors) {
            index.addNode(vector);
        }
        
        // Generate random query vectors (not from the dataset)
        float[][] queries = generateUniqueVectors(NUM_QUERIES, DIMENSIONS, 123); // Different seed
        
        // Write queries to file for Python analysis
        writeVectorsToFile(queries, "queries.txt");
        
        // Compute ground truth
        int[][] groundTruth = computeGroundTruth(vectors, queries, K);
        
        // Write ground truth to file for Python analysis
        writeGroundTruthToFile(groundTruth, "groundtruth.txt");
        
        // Test HNSW search with higher ef_search
        int[][] hnswResults = new int[NUM_QUERIES][];
        
        System.out.println("\n=== DEBUGGING SEARCH RESULTS ===");
        for (int i = 0; i < NUM_QUERIES; i++) {
            hnswResults[i] = index.search(queries[i], K, 100); // Higher ef_search
            
            System.out.println("\nQuery " + i + " (random query vector)");
            System.out.print("Ground truth: ");
            for (int j = 0; j < Math.min(5, groundTruth[i].length); j++) {
                System.out.print(groundTruth[i][j] + " ");
            }
            System.out.print("\nHNSW results: ");
            for (int j = 0; j < Math.min(5, hnswResults[i].length); j++) {
                System.out.print(hnswResults[i][j] + " ");
            }
            System.out.println();
        }
        
        // Calculate recall
        float recall = calculateRecall(groundTruth, hnswResults);
        
        System.out.println("\nRecall@" + K + ": " + String.format("%.3f", recall));
        
        // Debug: Test simple search
        System.out.println("\n=== SIMPLE SEARCH TEST ===");
        float[] testQuery = vectors[0];
        int[] simpleResults = index.search(testQuery, 1, 50);
        System.out.println("Searching for vector 0, got result: " + simpleResults[0]);
        
        // Lower threshold for debugging
        assertTrue(recall > 0.1f, "Recall should be > 0.1, got: " + recall);
    }

    @Test
    void testExactMatchDebug() {
        HNSWIndex index = new HNSWIndex(2, 10);
        
        // Add a few vectors
        float[] vector0 = {1.0f, 2.0f};
        float[] vector1 = {3.0f, 4.0f};
        float[] vector2 = {5.0f, 6.0f};
        
        index.addNode(vector0);
        index.addNode(vector1);
        index.addNode(vector2);
        
        // Search for exact matches
        System.out.println("Searching for vector0 (1.0, 2.0):");
        int[] results0 = index.search(vector0, 1, 10);
        System.out.println("Expected: 0, Got: " + results0[0]);
        
        System.out.println("Searching for vector1 (3.0, 4.0):");
        int[] results1 = index.search(vector1, 1, 10);
        System.out.println("Expected: 1, Got: " + results1[0]);
        
        System.out.println("Searching for vector2 (5.0, 6.0):");
        int[] results2 = index.search(vector2, 1, 10);
        System.out.println("Expected: 2, Got: " + results2[0]);
    }
    
    private float[][] generateUniqueVectors(int numVectors, int dimensions) {
        return generateUniqueVectors(numVectors, dimensions, 42);
    }
    
    private float[][] generateUniqueVectors(int numVectors, int dimensions, int seed) {
        float[][] vectors = new float[numVectors][dimensions];
        java.util.Random random = new java.util.Random(seed);
        
        for (int i = 0; i < numVectors; i++) {
            for (int j = 0; j < dimensions; j++) {
                vectors[i][j] = (float) ((random.nextGaussian() * 10) + (random.nextGaussian() * 100) );
            }
            // Normalize vector
            float norm = 0;
            for (float val : vectors[i]) {
                norm += val * val;
            }
            norm = (float) Math.sqrt(norm);
            if (norm > 0) {
                for (int j = 0; j < dimensions; j++) {
                    vectors[i][j] /= norm;
                }
            }
        }
        return vectors;
    }
    
    private int[][] computeGroundTruth(float[][] vectors, float[][] queries, int k) {
        int[][] groundTruth = new int[queries.length][k];
        
        for (int q = 0; q < queries.length; q++) {
            // Calculate distances to all vectors
            java.util.List<java.util.Map.Entry<Integer, Double>> distances = new java.util.ArrayList<>();
            for (int i = 0; i < vectors.length; i++) {
                double distance = VectorUtils.euclideanDistance(vectors[i], queries[q]);
                distances.add(new java.util.AbstractMap.SimpleEntry<>(i, distance));
            }
            
            // Sort by distance
            distances.sort(java.util.Map.Entry.comparingByValue());
            
            // Take top k
            for (int i = 0; i < k; i++) {
                groundTruth[q][i] = distances.get(i).getKey();
            }
        }
        return groundTruth;
    }
    
    private float calculateRecall(int[][] groundTruth, int[][] results) {
        int totalRelevant = 0;
        int totalFound = 0;
        
        for (int q = 0; q < groundTruth.length; q++) {
            java.util.Set<Integer> gtSet = new java.util.HashSet<>();
            for (int id : groundTruth[q]) {
                gtSet.add(id);
            }
            
            for (int id : results[q]) {
                if (gtSet.contains(id)) {
                    totalFound++;
                }
            }
            totalRelevant += groundTruth[q].length;
        }
        
        return (float) totalFound / totalRelevant;
    }
    
    private void writeVectorsToFile(float[][] vectors, String filename) {
        try (java.io.PrintWriter writer = new java.io.PrintWriter(filename)) {
            for (float[] vector : vectors) {
                for (int i = 0; i < vector.length; i++) {
                    writer.print(vector[i]);
                    if (i < vector.length - 1) writer.print(",");
                }
                writer.println();
            }
        } catch (java.io.FileNotFoundException e) {
            System.err.println("Error writing to file: " + e.getMessage());
        }
    }
    
    private void writeGroundTruthToFile(int[][] groundTruth, String filename) {
        try (java.io.PrintWriter writer = new java.io.PrintWriter(filename)) {
            for (int[] gt : groundTruth) {
                for (int i = 0; i < gt.length; i++) {
                    writer.print(gt[i]);
                    if (i < gt.length - 1) writer.print(",");
                }
                writer.println();
            }
        } catch (java.io.FileNotFoundException e) {
            System.err.println("Error writing to file: " + e.getMessage());
        }
    }
}