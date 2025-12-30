package org.navneev;

import org.navneev.dataset.HDF5Reader;
import org.navneev.index.HNSWIndex;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * Main entry point for HNSW index benchmarking and evaluation.
 * 
 * <p>This class provides comprehensive testing and performance evaluation of the HNSW
 * implementation using HDF5 datasets. It measures:
 * <ul>
 *   <li>Index construction time</li>
 *   <li>Search performance (percentile analysis)</li>
 *   <li>Recall accuracy against ground truth</li>
 * </ul>
 * 
 * <h3>Usage:</h3>
 * <pre>
 * # Run with default dataset
 * ./gradlew run
 * 
 * # Run with custom dataset
 * ./gradlew run --args="path/to/dataset.h5"
 * </pre>
 * 
 * <h3>Expected HDF5 Dataset Structure:</h3>
 * <pre>
 * dataset.h5
 * ├── train     # Training vectors [N x D]
 * ├── test      # Query vectors [Q x D]
 * └── neighbors # Ground truth neighbors [Q x K]
 * </pre>
 * 
 * @author Navneev
 * @version 1.0
 * @since 1.0
 */
public class Main {

    /** Default path to HDF5 dataset file */
    private static String HDF5_FILE_PATH = "sift-128-euclidean.hdf5";

    /**
     * Main entry point for the application.
     * 
     * @param args command line arguments; args[0] can specify HDF5 file path
     */
    public static void main(String[] args) {
        if (args.length > 0) {
            HDF5_FILE_PATH = args[0];
        }
        testHNSWWithHDF5(HDF5_FILE_PATH);
    }
    
    /**
     * Tests HNSW index with HDF5 dataset including construction, search, and evaluation.
     * 
     * <p>This method performs a complete benchmark:
     * <ol>
     *   <li>Prints dataset information</li>
     *   <li>Builds HNSW index from training vectors</li>
     *   <li>Executes search queries from test set</li>
     *   <li>Computes recall against ground truth</li>
     *   <li>Reports search time percentiles</li>
     * </ol>
     * 
     * @param hdf5FilePath path to the HDF5 dataset file
     */
    private static void testHNSWWithHDF5(String hdf5FilePath) {
        System.out.println("Testing HNSW with HDF5 file: " + hdf5FilePath);
        
        HDF5Reader.printDatasetInfo(hdf5FilePath);
        
        try {
            HNSWIndex index = buildIndex(hdf5FilePath);
            System.gc();

            System.out.println("HNSW Stats: " + index.getHNSWIndexStats());
            
            System.out.println("\nTesting search...");
            int k = 100;

            float[][] query_vectors = HDF5Reader.readVectors(hdf5FilePath, "test");

            int[][] actual_results = new int[query_vectors.length][];
            List<Long> searchTimes = new ArrayList<>();
            int counter = 0;
            long startTime;
            for (float[] query : query_vectors) {
                startTime = System.currentTimeMillis();
                int[] topK = index.search(query, k, 100);
                long searchTime = System.currentTimeMillis() - startTime;
                actual_results[counter] = topK;
                searchTimes.add(searchTime);
                counter++;
            }

            float recall = computeRecall(actual_results);
            System.out.println("Recall is : " + recall + "\n");
            
            printSearchTimePercentiles(searchTimes);
        } catch (Exception e) {
            System.out.println("Error processing HDF5 file: " + e.getMessage());
            e.printStackTrace();
        }
    }
    

    /**
     * Prints search time percentiles for performance analysis.
     * 
     * <p>Calculates and displays:
     * <ul>
     *   <li>P50 (median) - typical search time</li>
     *   <li>P90 - 90th percentile search time</li>
     *   <li>P99 - 99th percentile search time</li>
     *   <li>P100 (max) - worst-case search time</li>
     * </ul>
     * 
     * @param searchTimes list of search times in milliseconds
     */
    private static void printSearchTimePercentiles(List<Long> searchTimes) {
        searchTimes.sort(Long::compareTo);
        int size = searchTimes.size();
        
        long p50 = searchTimes.get((int) (size * 0.50));
        long p90 = searchTimes.get((int) (size * 0.90));
        long p99 = searchTimes.get((int) (size * 0.99));
        long p100 = searchTimes.get(size - 1);

        System.out.println("Search Time Percentiles:");
        System.out.println("P50: " + p50 + " ms");
        System.out.println("P90: " + p90 + " ms");
        System.out.println("P99: " + p99 + " ms");
        System.out.println("P100: " + p100 + " ms");
    }
    
    /**
     * Computes recall@K metric by comparing search results against ground truth.
     * 
     * <p>Recall is calculated as:
     * <pre>
     * recall = (number of correct neighbors found) / (total ground truth neighbors)
     * </pre>
     * 
     * <p>A higher recall indicates better search quality. Perfect recall (1.0) means
     * all ground truth neighbors were found in the search results.
     * 
     * @param actualResults 2D array of search results [queries][k neighbors]
     * @return recall value between 0.0 and 1.0
     */
    private static float computeRecall(int [][] actualResults) {
        int[][] gt_results = HDF5Reader.readGroundTruths(HDF5_FILE_PATH, "neighbors");
        float neighbors_found = 0.0f;
        for(int i = 0; i < actualResults.length; i++) {
            Set<Integer> gt = Arrays.stream(gt_results[i]).boxed().collect(Collectors.toSet());
            for(int j = 0; j < actualResults[i].length; j++) {
                if(gt.contains(actualResults[i][j])) {
                    neighbors_found++;
                }
            }
        }
        return neighbors_found/(gt_results.length * gt_results[0].length);
    }

    /**
     * Builds HNSW index from training vectors in the HDF5 dataset.
     * 
     * <p>This method:
     * <ol>
     *   <li>Loads training vectors from HDF5 file</li>
     *   <li>Creates new HNSW index</li>
     *   <li>Adds all vectors to the index</li>
     *   <li>Reports construction time and progress</li>
     * </ol>
     * 
     * <p>Progress is printed every 100,000 vectors to monitor long-running builds.
     * 
     * @param hdf5FilePath path to the HDF5 dataset file
     * @return constructed HNSW index ready for search
     * @throws RuntimeException if HDF5 file cannot be read
     */
    private static HNSWIndex buildIndex(final String hdf5FilePath) {
        float[][] vectors = HDF5Reader.readVectors(hdf5FilePath, "train");
        System.out.println("\nLoaded " + vectors.length + " vectors with dimension " + vectors[0].length);

        final HNSWIndex index = new HNSWIndex(vectors[0].length, vectors.length);

        int numVectors = vectors.length;
        System.out.println("Adding " + numVectors + " vectors to HNSW index...");

        long startTime = System.currentTimeMillis();
        for (int i = 0; i < numVectors; i++) {
            index.addNode(vectors[i]);
            if ((i + 1) % 100000 == 0) {
                System.out.println("Added " + (i + 1) + " vectors");
            }
        }
        long buildTime = System.currentTimeMillis() - startTime;

        System.out.println("Index built in " + buildTime + " ms");
        return index;
    }
}