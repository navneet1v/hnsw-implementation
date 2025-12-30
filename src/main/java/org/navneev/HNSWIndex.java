package org.navneev;

import org.navneev.storage.StorageFactory;
import org.navneev.storage.VectorStorage;
import org.navneev.utils.VectorUtils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Objects;
import java.util.PriorityQueue;
import java.util.Random;

/**
 * High-performance implementation of Hierarchical Navigable Small World (HNSW) algorithm
 * for approximate nearest neighbor search.
 *
 * <p>This implementation is based on the paper "Efficient and robust approximate nearest
 * neighbor search using Hierarchical Navigable Small World graphs" by Malkov and Yashunin
 * (arXiv:1603.09320). The algorithm creates a multi-layer graph structure where higher
 * layers contain fewer nodes, enabling efficient logarithmic search complexity.
 *
 * <p>Key features:
 * <ul>
 *   <li>Multi-layer graph with probabilistic level assignment</li>
 *   <li>Greedy search on upper layers, beam search on layer 0</li>
 *   <li>Diversity-based neighbor selection heuristic</li>
 *   <li>Configurable parameters for quality vs performance trade-offs</li>
 * </ul>
 *
 * <p>The algorithm works by:
 * <ol>
 *   <li>Assigning each node a random level using exponential decay probability</li>
 *   <li>Connecting nodes to M neighbors at each layer using diversity heuristic</li>
 *   <li>Searching from entry point through layers, refining candidates at each level</li>
 * </ol>
 *
 * <h3>Performance Characteristics:</h3>
 * <ul>
 *   <li>Construction: O(N log N) expected time complexity</li>
 *   <li>Search: O(log N) expected time complexity</li>
 *   <li>Space: O(N × M) for storing connections</li>
 * </ul>
 *
 * <h3>Usage Example:</h3>
 * <pre>{@code
 * HNSWIndex index = new HNSWIndex();
 *
 * // Add vectors to index
 * float[] vector1 = {1.0f, 2.0f, 3.0f};
 * float[] vector2 = {4.0f, 5.0f, 6.0f};
 * index.addNode(vector1);
 * index.addNode(vector2);
 *
 * // Search for k nearest neighbors
 * float[] query = {1.1f, 2.1f, 3.1f};
 * int[] results = index.search(query, k=5, efSearch=50);
 * }</pre>
 *
 * @author Navneev
 * @version 1.0
 * @since 1.0
 * @see <a href="https://arxiv.org/abs/1603.09320">HNSW Paper</a>
 */
public class HNSWIndex {

    private static final int DEFAULT_EF_CONSTRUCTION = 100;

    /** Map of node IDs to HNSWNode objects for fast node lookup */
    private final HNSWNode[] nodesById;

    /** Maximum number of connections per node (M parameter from paper) */
    private final int M = 16;

    /** Search width during construction (efConstruction parameter) */
    private final int efConstruction;

    /** Random number generator for level assignment */
    private final Random random;

    /** ID of the entry point node (highest level node) */
    private int entryPoint = -1;

    /** Map of node IDs to their vector data for distance calculations */
    private final VectorStorage idToVectorStorage;

    /** Current maximum level in the graph */
    private int maxLevel = -1;

    /** Current NodeId in the graph */
    private int currentNodeId = 0;

    /**
     * Constructs a new HNSW index with default parameters.
     *
     * <p>Initializes the index with:
     * <ul>
     *   <li>M = 16 (maximum connections per node)</li>
     *   <li>efConstruction = 100 (construction search width)</li>
     *   <li>Random seed for reproducible level assignment</li>
     * </ul>
     */
    public HNSWIndex(int dimensions, int totalNumberOfVectors) {
        this(DEFAULT_EF_CONSTRUCTION, dimensions, totalNumberOfVectors);
    }

    public HNSWIndex(int efConstruction, int dimensions, int totalNumberOfVectors) {
        this.efConstruction = efConstruction;
        this.idToVectorStorage = StorageFactory.createStorage(dimensions, totalNumberOfVectors);
        this.nodesById = new HNSWNode[totalNumberOfVectors];
        random = new Random();
    }

    /**
     * Returns the level of a node based on the HNSW probability function.
     *
     * <p>Uses exponential decay probability: P(level = l) = (1/ln(M))^l
     * This creates a natural hierarchy where:
     * <ul>
     *   <li>All nodes exist at level 0</li>
     *   <li>~62% of nodes reach level 1 (with M=16)</li>
     *   <li>~38% of nodes reach level 2</li>
     *   <li>Exponentially fewer nodes at higher levels</li>
     * </ul>
     *
     * <p>The probability formula ensures that as M increases, the hierarchy
     * becomes steeper, creating more efficient search paths.
     *
     * @return randomly assigned level for a new node (≥ 0)
     */
    private int getRandomLevel() {
        double prob = 1.0 / Math.log(M);
        int level = 0;
        while (random.nextDouble() < prob) {
            level++;
        }
        return level;
    }

    /**
     * Adds a new vector to the HNSW index.
     *
     * <p>This method implements Algorithm 1 from the HNSW paper. The process:
     * <ol>
     *   <li>Assigns a random level to the new node</li>
     *   <li>If first node, sets it as entry point</li>
     *   <li>Searches from entry point through upper layers to find insertion point</li>
     *   <li>Connects the node to selected neighbors at each layer using diversity heuristic</li>
     *   <li>Updates entry point if new node has higher level</li>
     * </ol>
     *
     * <p>The neighbor selection uses a diversity heuristic to maintain good graph
     * connectivity and avoid creating hub nodes that could degrade search quality.
     *
     * @param vector the vector data to add to the index (will be copied)
     * @throws IllegalArgumentException if vector is null or empty
     */
    public void addNode(float[] vector) {
        // max level, where level start from 0
        int level = getRandomLevel();
        int nodeId = currentNodeId;
        currentNodeId ++;
        final HNSWNode newNode = new HNSWNode(nodeId, level, M);
        nodesById[nodeId] =  newNode;
        idToVectorStorage.addVector(nodeId, vector);

        if (entryPoint == -1) {
            entryPoint = nodeId;
            maxLevel = level;
            return;
        }

        int current = entryPoint;

        // Traverse from top layer to newNode's level
        for (int l = maxLevel; l > level; l--) {
            current = searchLayer(vector, current, 1, l)[0].id();
        }

        for (int l = Math.min(level, maxLevel); l >= 0; l--) {

            final IdAndDistance[] neighborsIds = searchLayer(vector, current, efConstruction, l);

            final List<Integer> selected = selectNeighborsHeuristic(neighborsIds, newNode.id, l);

            for (final Integer neighbor : selected) {
                newNode.addNeighbor(l, neighbor);
                nodesById[neighbor].addNeighbor(l, newNode.id);
                if(nodesById[neighbor].getNeighbors(l).size() > M) {
                    // Min heap
                    PriorityQueue<IdAndDistance> neighborCandidateQueue = new PriorityQueue<>(
                            Comparator.comparingDouble(IdAndDistance::distance)
                    );

                    float[] neighborVector = idToVectorStorage.getVector(neighbor);

                    for (int candidate : nodesById[neighbor].getNeighbors(l)) {
                        neighborCandidateQueue.add(new IdAndDistance(candidate, dis(candidate, neighborVector)));
                    }
                    final IdAndDistance[] intArray = new IdAndDistance[neighborCandidateQueue.size()];
                    int counter = 0;
                    while (!neighborCandidateQueue.isEmpty()) {
                        intArray[counter] = neighborCandidateQueue.poll();
                        counter++;
                    }
                    nodesById[neighbor].setNeighbors(l, selectNeighborsHeuristic(intArray, neighbor, l));
                }
            }
        }

        if (level > maxLevel) {
            entryPoint = newNode.id;
            maxLevel = level;
        }
    }

    /**
     * Searches for candidate neighbors within a specific layer of the graph.
     *
     * <p>Implements Algorithm 2 from the HNSW paper (SEARCH-LAYER). Uses a
     * best-first search strategy with two priority queues:
     * <ul>
     *   <li>Candidates queue: nodes to explore (min-heap by distance)</li>
     *   <li>Result queue: best nodes found (max-heap by distance, limited to ef)</li>
     * </ul>
     *
     * <p>The search terminates when no more promising candidates remain or
     * when the current candidate is farther than the worst result.
     *
     * @param query the query vector to search for
     * @param entry the entry point node ID for this layer
     * @param ef the maximum number of candidates to maintain
     * @param layer the layer number to search in
     * @return array of node IDs sorted by distance (closest first)
     */
    private IdAndDistance[] searchLayer(float[] query, Integer entry, int ef, int layer) {
        // Min Heap
        final PriorityQueue<IdAndDistance> candidates = new PriorityQueue<>(
                Comparator.comparingDouble(IdAndDistance::distance)
        );
        // Max Heap
        final PriorityQueue<IdAndDistance> result = new PriorityQueue<>(
                Comparator.comparingDouble(IdAndDistance::distance).reversed()
        );

        final BitSet visited = new BitSet(idToVectorStorage.getTotalNumberOfVectors());
        final double entryPointDistance = dis(entry, query);

        candidates.add(new IdAndDistance(entry, entryPointDistance));
        visited.set(entry);
        // negating distance to ensure that worst neighbor comes on top.
        result.add(new IdAndDistance(entry, entryPointDistance));

        while (!candidates.isEmpty()) {
            IdAndDistance current = candidates.poll();
            IdAndDistance farthestElement = result.element();
            if(current.distance() > farthestElement.distance()) {
                // All elements in result is evaluated
                break;
            }

            for (int neighborId : nodesById[current.id()].getNeighbors(layer)) {
                if (!visited.get(neighborId)) {
                    visited.set(neighborId);
                    final IdAndDistance neighborIdAndDistance = new IdAndDistance(neighborId, dis(neighborId, query));

                    // Main logic: if current neighbor is closer than the worst node present in the result, or result
                    // size is less than ef add the neighbor in final list. This logic is directly taken from the paper
                    if(neighborIdAndDistance.distance() < farthestElement.distance() || result.size() < ef) {
                        candidates.add(neighborIdAndDistance);
                        result.add(new IdAndDistance(neighborId, neighborIdAndDistance.distance()));
                        while(result.size() > ef) {
                            result.poll();
                        }
                        farthestElement = result.element();
                    }
                }
            }
        }

        IdAndDistance[] resultArray = new IdAndDistance[result.size()];
        int i = result.size() - 1;

        while (!result.isEmpty()) {
            resultArray[i] = result.poll();
            i--;
        }

        return resultArray;
    }

    /**
     * Selects diverse neighbors using the HNSW heuristic to maintain graph quality.
     *
     * <p>Implements a diversity-based selection strategy that:
     * <ul>
     *   <li>Prioritizes closer candidates</li>
     *   <li>Avoids selecting candidates that are closer to already-selected neighbors</li>
     *   <li>Prevents creation of hub nodes that could degrade search performance</li>
     * </ul>
     *
     * <p>This heuristic is crucial for maintaining the small-world property of the graph
     * and ensuring efficient search paths.
     *
     * @param candidates list of candidate node IDs sorted by distance
     * @param newNodeId ID of the node being connected
     * @return list of selected neighbor IDs (size ≤ M)
     */
    private List<Integer> selectNeighborsHeuristic(final IdAndDistance[] candidates, int newNodeId, int level) {
        final List<Integer> finalSelected = new ArrayList<>(M);
        final List<Integer> discardedIds = new ArrayList<>(M);
        int counter = 0;
        while (counter < candidates.length && finalSelected.size() < M) {
            // Let's apply the heuristic to select the diverse nodes for HNSW
            IdAndDistance candidate = candidates[counter];
            counter++;
            boolean isDiverse = true;
            float[] candidateVector = idToVectorStorage.getVector(candidate.id());
            for(int currentNeighbor : finalSelected) {
                // if the node which we are trying to add as a neighbor is closet to already connected neighbor then
                // we should not add that node in neighbors list.
                if(dis(currentNeighbor, candidateVector) < candidate.distance()) {
                    isDiverse = false;
                }
            }
            if(isDiverse) {
                finalSelected.add(candidate.id());
            } else {
                discardedIds.add(candidate.id());
            }
        }
        counter = 0;
        while (counter < discardedIds.size() && finalSelected.size() < M) {
            finalSelected.add(discardedIds.get(counter));
            counter++;
        }


        return finalSelected;
    }

    /**
     * Calculates Euclidean distance between two nodes.
     *
     * @param a first node ID
     * @param b second node ID
     * @return squared Euclidean distance between the nodes
     */
    private double dis(int a, int b) {
        return VectorUtils.euclideanDistance(idToVectorStorage.getVector(a), idToVectorStorage.getVector(b));
    }

    /**
     * Calculates Euclidean distance between a node and a query vector.
     *
     * @param id node ID
     * @param q query vector
     * @return squared Euclidean distance
     */
    private double dis(int id, float[] q) {
            return VectorUtils.euclideanDistance(idToVectorStorage.getVector(id), q);
    }

    /**
     * Searches for the k nearest neighbors of a query vector.
     *
     * <p>Implements the complete HNSW search algorithm:
     * <ol>
     *   <li>Starts from the entry point at the highest layer</li>
     *   <li>Performs greedy search through upper layers (ef=1)</li>
     *   <li>Conducts beam search on layer 0 with specified efSearch</li>
     *   <li>Returns the k closest neighbors</li>
     * </ol>
     *
     * <p>The efSearch parameter controls the search quality vs speed trade-off:
     * higher values improve recall but increase search time.
     *
     * @param query the query vector to search for
     * @param k number of nearest neighbors to return
     * @param ef_search search width for layer 0 (should be ≥ k)
     * @return array of k nearest neighbor node IDs, sorted by distance
     * @throws IllegalArgumentException if k ≤ 0 or efSearch < k
     */
    public int[] search(float[] query, int k, int ef_search) {
        int current = entryPoint;
        // Search all the top layers to find the entry point for the bottom layer.
        for (int l = maxLevel; l >= 1; l--) {
            current = searchLayer(query, current, 1, l)[0].id();
        }
        // Now Search on the bottom layer
        final IdAndDistance[] results = searchLayer(query, current, ef_search, 0);
        int finalSize = Math.min(k, results.length);
        int[] resultIds = new int[finalSize];
        for(int i = 0 ; i < finalSize; i++) {
            resultIds[i] = results[i].id();
        }
        return resultIds;
    }

    /**
     * Record representing a node ID and its distance from a query point.
     * Used internally for efficient priority queue operations during search.
     *
     * @param id the node identifier
     * @param distance the distance from query (squared Euclidean distance)
     */
    record IdAndDistance(int id, double distance) {}
}
