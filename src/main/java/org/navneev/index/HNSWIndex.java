package org.navneev.index;

import org.navneev.index.model.HNSWNode;
import org.navneev.index.model.IntegerList;
import org.navneev.index.storage.StorageFactory;
import org.navneev.index.storage.VectorStorage;
import org.navneev.utils.VectorUtils;

import java.util.BitSet;
import java.util.Comparator;
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
    private final Random random = new Random();

    /** ID of the entry point node (highest level node) */
    private int entryPoint = -1;

    /** Map of node IDs to their vector data for distance calculations */
    private final VectorStorage idToVectorStorage;

    /** Current maximum level in the graph */
    private int maxLevel = -1;

    /** Current NodeId in the graph */
    private int currentNodeId = 0;

    /** Distance vector used for distance calculations */
    private final float[] distanceVector;

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
        this.distanceVector = new float[dimensions];
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

        int neighbor;
        for (int l = Math.min(level, maxLevel); l >= 0; l--) {
            // find the neighbors to be added
            final IdAndDistance[] neighborsIdAndDistance = searchLayer(vector, current, efConstruction, l);
            // select the final list of neighbors to be added.
            final IntegerList selected = selectNeighborsHeuristic(neighborsIdAndDistance, newNode.id, l);
            // Add the new node per level and also attach the correct neighbors
            for (int i = 0; i < selected.size(); i++) {
                neighbor = selected.get(i);
                // Create all the bidirectional links
                newNode.addNeighbor(l, neighbor);
                nodesById[neighbor].addNeighbor(l, newNode.id);
            }

            for (int i = 0; i < selected.size(); i++) {
                neighbor = selected.get(i);
                final IntegerList neighborsConnections = nodesById[neighbor].getNeighbors(l);
                if (neighborsConnections.size() > M) {
                    // Min Heap
                    final PriorityQueue<IdAndDistance> neighborCandidatesPQ = new PriorityQueue<>(
                            Comparator.comparingDouble(IdAndDistance::distance)
                    );
                    float[] neighborsVector = cloneAndGetVector(neighbor);
                    for(int j = 0 ; j < neighborsConnections.size(); j++) {
                        neighborCandidatesPQ.add(new IdAndDistance(neighborsConnections.get(j), dis(neighborsConnections.get(j), neighborsVector)));
                    }

                    IdAndDistance[] neighborsIdAndDistanceNew = new IdAndDistance[neighborsConnections.size()];
                    for(int j = 0 ; j < neighborsConnections.size(); j++) {
                        neighborsIdAndDistanceNew[j] = neighborCandidatesPQ.poll();
                    }

                    // shrink the neighbors
                    IntegerList shrinkedNeighborhood = selectNeighborsHeuristic(neighborsIdAndDistanceNew, neighbor, l);
                    nodesById[neighbor].updateNeighborhood(l, shrinkedNeighborhood);
                }

            }
        }

        // New level got created and the current node is getting added here, lets make this new node as the entry point.
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
    private IdAndDistance[] searchLayer(float[] query, int entry, int ef, int layer) {
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
        result.add(new IdAndDistance(entry, entryPointDistance));

        while (!candidates.isEmpty()) {
            IdAndDistance candidate = candidates.poll();
            IdAndDistance farthestElementInResult = result.element();
            if(candidate.distance() > farthestElementInResult.distance() && candidate.id() != farthestElementInResult.id()) {
                // All elements in result is evaluated
                break;
            }

            final IntegerList neighborsList = nodesById[candidate.id()].getNeighbors(layer);
            int neighborId;
            for (int i = 0; i < neighborsList.size(); i++) {
                neighborId = neighborsList.get(i);
                if (!visited.get(neighborId)) {
                    visited.set(neighborId);
                    final IdAndDistance neighborIdAndDistance = new IdAndDistance(neighborId, dis(neighborId, query));

                    // Main logic: if current neighbor is closer than the worst node present in the result, or result
                    // size is less than ef add the neighbor in final list.
                    farthestElementInResult = result.element();
                    if(neighborIdAndDistance.distance() < farthestElementInResult.distance() || result.size() < ef) {
                        candidates.add(neighborIdAndDistance);
                        result.add(new IdAndDistance(neighborId, neighborIdAndDistance.distance()));
                        if(result.size() > ef) {
                            result.poll();
                        }
                    }
                }
            }
        }

        // reversing the result to ensure that closest neighbors are returned
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
     * @param nodeToLinkNeighborsTo ID of the node being connected
     * @return list of selected neighbor IDs (size ≤ M)
     */
    private IntegerList selectNeighborsHeuristic(final IdAndDistance[] candidates, int nodeToLinkNeighborsTo,
                                                 int level) {
        final IntegerList finalSelected = new IntegerList(M);
        final IntegerList discardedList = new IntegerList(M);
        int counter = 0;
        while (counter < candidates.length && finalSelected.size() < M) {
            // Let's apply the heuristic to select the diverse nodes for HNSW
            final IdAndDistance candidate = candidates[counter];
            counter++;
            boolean isDiverse = true;
            int currentNeighbor;
            float[] candidateVector = cloneAndGetVector(candidate.id());
            for(int i = 0; i < finalSelected.size(); i++) {
                currentNeighbor = finalSelected.get(i);
                // if the candidate is closer to an already selected neighbor than to the node we're linking to,
                // then it's not diverse enough and should be rejected
                if(dis(currentNeighbor, candidateVector) < candidate.distance() ) {
                    isDiverse = false;
                    break;
                }
            }
            if(isDiverse) {
                finalSelected.add(candidate.id());
            } else {
                discardedList.add(candidate.id());
            }
        }

        counter = 0;
        while(finalSelected.size() < M && level == 0 && counter < discardedList.size()) {
            finalSelected.add(discardedList.get(counter));
            counter++;
        }

        return finalSelected;
    }

    /**
     * Calculates Euclidean distance between a node and a query vector.
     *
     * @param id node ID
     * @param q query vector
     * @return squared Euclidean distance
     */
    private double dis(int id, float[] q) {
        float[] tempVector = idToVectorStorage.getVector(id);
        System.arraycopy(tempVector, 0, distanceVector, 0, tempVector.length);
        return VectorUtils.euclideanDistance(distanceVector, q);
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
        for(int i = 0; i < finalSize; i++) {
            resultIds[i] = results[i].id();
        }
        return resultIds;
    }

    private float[] cloneAndGetVector(int id) {
        float[] tempVector = idToVectorStorage.getVector(id);
        float[] newVector = new float[tempVector.length];
        System.arraycopy(tempVector, 0, newVector, 0, tempVector.length);
        return  newVector;
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
