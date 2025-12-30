package org.navneev.index;

import org.navneev.index.model.HNSWNode;
import org.navneev.index.model.HNSWStats;
import org.navneev.index.model.IntegerList;
import org.navneev.index.storage.OffHeapVectorsStorage;
import org.navneev.index.storage.StorageFactory;
import org.navneev.index.storage.VectorStorage;
import org.navneev.utils.HNSWIndexUtils;
import org.navneev.utils.VectorUtils;

import java.util.BitSet;
import java.util.Comparator;
import java.util.PriorityQueue;

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
 * @see <a href="https://arxiv.org/abs/1603.09320">HNSW Paper</a>
 * @since 1.0
 */
public class HNSWIndex {

    private static final int DEFAULT_EF_CONSTRUCTION = 100;

    /**
     * Map of node IDs to HNSWNode objects for fast node lookup
     */
    private final HNSWNode[] nodesById;

    /**
     * Maximum number of connections per node (M parameter from paper)
     */
    private final int M = 16;

    /**
     * Search width during construction (efConstruction parameter)
     */
    private final int efConstruction;

    /**
     * Random number generator for level assignment
     */
    private final HNSWLevelGenerator levelGenerator;

    /**
     * ID of the entry point node (highest level node)
     */
    private int entryPoint = -1;

    /**
     * Map of node IDs to their vector data for distance calculations
     */
    private final VectorStorage idToVectorStorage;

    /**
     * Current maximum level in the graph
     */
    private int maxLevel = -1;

    /**
     * Current NodeId in the graph
     */
    private int currentNodeId = 0;

    private final int dimensions;

    private long totalBuildTimeInMillis;

    private final PriorityQueue<IdAndDistance> candidatesQueue;

    private final PriorityQueue<IdAndDistance> resultQueue;

    private final BitSet visited;

    private final float[] temporaryClonedVector;

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
        this.dimensions = dimensions;
        this.levelGenerator = new HNSWLevelGenerator(M);
        candidatesQueue = new PriorityQueue<>(Comparator.comparingDouble(IdAndDistance::distance));
        resultQueue = new PriorityQueue<>(Comparator.comparingDouble(IdAndDistance::distance).reversed());
        visited = new BitSet(totalNumberOfVectors);
        temporaryClonedVector = new float[dimensions];
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
        long startTime = System.currentTimeMillis();
        // max level, where level start from 0
        int level = levelGenerator.getRandomLevel();
        int nodeId = currentNodeId;
        currentNodeId++;
        final HNSWNode newNode = new HNSWNode(nodeId, level, M);
        nodesById[nodeId] = newNode;
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
            // update the entry point which will be used as an entry for the next level
            current = neighborsIdAndDistance[0].id();
            // select the final list of neighbors to be added.
            final IntegerList selected = selectNeighborsHeuristic(neighborsIdAndDistance, newNode.id, l);
            // Add the new node per level and also attach the correct neighbors
            for (int i = 0; i < selected.size(); i++) {
                neighbor = selected.get(i);
                // Create all the bidirectional links
                newNode.addNeighbor(l, neighbor);
                if (nodesById[neighbor].getNeighbors(l).size() < getM(l)) {
                    nodesById[neighbor].addNeighbor(l, newNode.id);
                } else {
                    nodesById[neighbor].updateNeighborhood(l, shrinkNeighbors(neighbor, newNode.id, l));
                }

            }
        }

        // New level got created and the current node is getting added here, lets make this new node as the entry point.
        if (level > maxLevel) {
            entryPoint = newNode.id;
            maxLevel = level;
        }
        this.totalBuildTimeInMillis += System.currentTimeMillis() - startTime;
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
     * @param query     the query vector to search for
     * @param k         number of nearest neighbors to return
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
        for (int i = 0; i < finalSize; i++) {
            resultIds[i] = results[i].id();
        }
        return resultIds;
    }

    public HNSWStats getHNSWIndexStats() {
        return HNSWStats.builder()
                .M(M)
                .efConstruction(efConstruction)
                .totalNumberOfNodes(idToVectorStorage.getTotalNumberOfVectors())
                .dimensions(dimensions)
                .maxLevel(maxLevel)
                .entryPoint(entryPoint)
                .totalBuildTimeInMillis(totalBuildTimeInMillis)
                .build();
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
     * @param ef    the maximum number of candidates to maintain
     * @param layer the layer number to search in
     * @return array of node IDs sorted by distance (closest first)
     */
    private IdAndDistance[] searchLayer(float[] query, int entry, int ef, int layer) {
        final double entryPointDistance = dis(entry, query);

        candidatesQueue.add(new IdAndDistance(entry, entryPointDistance));
        visited.set(entry);
        resultQueue.add(new IdAndDistance(entry, entryPointDistance));

        while (!candidatesQueue.isEmpty()) {
            IdAndDistance candidate = candidatesQueue.poll();
            IdAndDistance farthestElementInResult = resultQueue.element();
            if (candidate.distance() > farthestElementInResult.distance()) {
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
                    farthestElementInResult = resultQueue.element();
                    if (neighborIdAndDistance.distance() < farthestElementInResult.distance() || resultQueue.size() < ef) {
                        candidatesQueue.add(neighborIdAndDistance);
                        resultQueue.add(new IdAndDistance(neighborId, neighborIdAndDistance.distance()));
                        if (resultQueue.size() > ef) {
                            resultQueue.poll();
                        }
                    }
                }
            }
        }

        // reversing the result to ensure that closest neighbors are returned
        IdAndDistance[] resultArray = new IdAndDistance[resultQueue.size()];
        int i = resultQueue.size() - 1;

        while (!resultQueue.isEmpty()) {
            resultArray[i] = resultQueue.poll();
            i--;
        }

        candidatesQueue.clear();
        resultQueue.clear();
        visited.clear();

        return resultArray;
    }

    /**
     * Shrinks a node's neighbor list to maintain the maximum connection limit M.
     *
     * <p>This method is called when a node's neighbor count exceeds getM(level) when
     * adding bidirectional links during index construction. It ensures that each node
     * maintains at most M connections per layer.
     *
     * <p>Two shrinking strategies are supported:
     * <ul>
     *   <li><b>Greedy strategy:</b> Keeps the M-1 closest neighbors and adds the new node.
     *       Fast but may reduce graph quality.</li>
     *   <li><b>Heuristic strategy:</b> Uses selectNeighborsHeuristic to choose diverse neighbors.
     *       Slower but maintains better graph connectivity and search quality.</li>
     * </ul>
     *
     * <p><b>Why shrinking is necessary:</b>
     * <p>
     * When node A connects to node B, we create a bidirectional link. If B already has M neighbors,
     * adding A would exceed the limit. We must prune B's neighbors to maintain exactly M connections.
     *
     * <p><b>Algorithm:</b>
     * <ol>
     *   <li>Get all current neighbors of the node (size > M)</li>
     *   <li>Compute distances from node to all its neighbors</li>
     *   <li>Add neighbors to priority queue (sorted by distance)</li>
     *   <li>If greedy: Keep M-1 closest + new node</li>
     *   <li>If heuristic: Use diversity-based selection to pick M neighbors</li>
     * </ol>
     *
     * <p><b>Example:</b>
     * <pre>
     * Node B has neighbors: [1, 2, 3, 4, 5] (M=4, but has 5)
     * New node A wants to connect to B
     * 
     * Greedy: Keep 3 closest + A → [1, 2, 3, A]
     * Heuristic: Select diverse set → [1, 3, 5, A] (maintains diversity)
     * </pre>
     *
     * @param nodeId the node whose neighbors need to be pruned
     * @param newNodeId the new node being added to the neighbor list
     * @param level the layer at which to shrink neighbors
     * @return pruned list of neighbor IDs with size ≤ getM(level)
     */
    private IntegerList shrinkNeighbors(int nodeId, int newNodeId, int level) {
        IntegerList neighborsConnections = nodesById[nodeId].getNeighbors(level);
        final PriorityQueue<IdAndDistance> neighborCandidatesPQ = new PriorityQueue<>(
                Comparator.comparingDouble(IdAndDistance::distance)
        );
        float[] neighborVector = getVectorClonedIfNeeded(nodeId);
        for (int j = 0; j < neighborsConnections.size(); j++) {
            neighborCandidatesPQ.add(new IdAndDistance(neighborsConnections.get(j), dis(neighborsConnections.get(j), neighborVector)));
        }
        if (HNSWIndexUtils.useGreedyNeighborShrinkingStrategy()) {
            for (int j = 0; j < getM(level) - 1; j++) {
                neighborsConnections.update(j, neighborCandidatesPQ.poll().id());
            }
            // replace the farthest node with the new node which we are trying to add.
            neighborsConnections.update(getM(level) - 1, newNodeId);
        } else {
            neighborCandidatesPQ.add(new IdAndDistance(newNodeId, dis(newNodeId, neighborVector)));
            IdAndDistance[] neighborsIdAndDistanceNew = new IdAndDistance[neighborsConnections.size()];
            for (int j = 0; j < neighborsConnections.size(); j++) {
                neighborsIdAndDistanceNew[j] = neighborCandidatesPQ.poll();
            }
            // Final shrunk neighbors list
            neighborsConnections = selectNeighborsHeuristic(neighborsIdAndDistanceNew, nodeId, level);
        }

        return neighborsConnections;
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
     * @param candidates            list of candidate node IDs sorted by distance
     * @param nodeToLinkNeighborsTo ID of the node being connected
     * @return list of selected neighbor IDs (size ≤ M)
     */
    private IntegerList selectNeighborsHeuristic(final IdAndDistance[] candidates, int nodeToLinkNeighborsTo,
                                                 int level) {
        final IntegerList finalSelected = new IntegerList(getM(level));
        final IntegerList discardedList = new IntegerList();
        int counter = 0;
        while (counter < candidates.length && finalSelected.size() < getM(level)) {
            // Let's apply the heuristic to select the diverse nodes for HNSW
            final IdAndDistance candidate = candidates[counter];
            counter++;
            boolean isDiverse = true;
            int selectedNeighbor;
            float[] candidateVector = getVectorClonedIfNeeded(candidate.id());
            for (int i = 0; i < finalSelected.size(); i++) {
                selectedNeighbor = finalSelected.get(i);
                // if the candidate is closer to an already selected neighbor than to the node we're linking to,
                // then it's not diverse enough and should be rejected
                if (dis(selectedNeighbor, candidateVector) < candidate.distance()) {
                    isDiverse = false;
                    break;
                }
            }
            if (isDiverse) {
                finalSelected.add(candidate.id());
            } else {
                discardedList.add(candidate.id());
            }
        }

        counter = 0;
        while (finalSelected.size() < getM(level) && counter < discardedList.size()) {
            finalSelected.add(discardedList.get(counter));
            counter++;
        }

        return finalSelected;
    }

    /**
     * Calculates Euclidean distance between a node and a query vector.
     *
     * @param id node ID
     * @param q  query vector
     * @return squared Euclidean distance
     */
    private double dis(int id, float[] q) {
        float[] tempVector = idToVectorStorage.getVector(id);
        return VectorUtils.euclideanDistance(tempVector, q);
    }

    private float[] getVectorClonedIfNeeded(int id) {
        if (idToVectorStorage instanceof OffHeapVectorsStorage) {
            //float[] tempVector = idToVectorStorage.getVector(id);
            //System.arraycopy(tempVector, 0, temporaryClonedVector, 0, tempVector.length);
            idToVectorStorage.loadVectorInArray(id, temporaryClonedVector);
            return temporaryClonedVector;
        }
        return idToVectorStorage.getVector(id);
    }

    /**
     * Returns the maximum number of connections allowed for a node at the specified layer.
     *
     * <p>The HNSW algorithm uses different connection limits for different layers:
     * <ul>
     *   <li><b>Layer 0 (base layer):</b> 2*M connections - allows denser connectivity for better recall</li>
     *   <li><b>Upper layers (1, 2, ...):</b> M connections - sparser for faster navigation</li>
     * </ul>
     *
     * <p><b>Why 2*M doesn't cause memory estimation problems:</b>
     * <p>
     * While layer 0 uses 2*M connections, memory estimation remains accurate because:
     * <ol>
     *   <li><b>All nodes exist in layer 0:</b> Every node in the graph appears in the base layer,
     *       so the 2*M factor is consistently applied to all nodes at this level.</li>
     *   <li><b>Upper layers have exponentially fewer nodes:</b> Due to the exponential decay
     *       probability distribution (e.g., ~63% at L0, ~23% at L1, ~9% at L2), upper layers
     *       contain significantly fewer nodes despite using M connections each.</li>
     *   <li><b>Memory calculation accounts for both:</b> The formula considers:
     *       <pre>
     *       Memory ≈ N × (2*M × 4 bytes)           // Layer 0: all N nodes
     *              + N × 0.37 × M × 4 bytes × avgLevels  // Upper layers: ~37% of nodes
     *       </pre>
     *       Where the layer 0 term (2*M) dominates but is predictable.</li>
     *   <li><b>Bounded by design:</b> The selectNeighborsHeuristic ensures connections never
     *       exceed getM(level), preventing unbounded growth.</li>
     * </ol>
     *
     * <p><b>Example for 1M nodes, M=16:</b>
     * <ul>
     *   <li>Layer 0: 1,000,000 nodes × 32 connections × 4 bytes = 128 MB</li>
     *   <li>Layer 1: ~370,000 nodes × 16 connections × 4 bytes = 23.7 MB</li>
     *   <li>Layer 2: ~137,000 nodes × 16 connections × 4 bytes = 8.8 MB</li>
     *   <li>Higher layers: diminishing contribution</li>
     *   <li>Total: ~160 MB (predictable and bounded)</li>
     * </ul>
     *
     * @param level the layer number (0 for base layer, 1+ for upper layers)
     * @return 2*M for layer 0, M for upper layers
     */
    private int getM(int level) {
        return level == 0 ? M * 2 : M;
    }

    /**
     * Record representing a node ID and its distance from a query point.
     * Used internally for efficient priority queue operations during search.
     *
     * @param id       the node identifier
     * @param distance the distance from query (squared Euclidean distance)
     */
    record IdAndDistance(int id, double distance) {
    }
}
