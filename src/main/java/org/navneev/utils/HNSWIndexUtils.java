package org.navneev.utils;

/**
 * Utility class for HNSW index configuration and runtime settings.
 *
 * <p>Provides centralized access to system properties and configuration options
 * that control HNSW algorithm behavior. These settings can be modified at runtime
 * via JVM system properties.
 *
 * <h3>Available Configurations:</h3>
 * <ul>
 *   <li><b>Neighbor Shrinking Strategy:</b> Controls how neighbors are pruned when
 *       a node exceeds the maximum connection limit M</li>
 * </ul>
 *
 * <h3>Usage Example:</h3>
 * <pre>{@code
 * // Set via JVM argument
 * java -Dvector.neighbor.greedy.shrink=false MyApp
 *
 * // Check in code
 * if (HNSWIndexUtils.useGreedyNeighborShrinkingStrategy()) {
 *     // Use fast greedy pruning
 * } else {
 *     // Use diversity-based heuristic pruning
 * }
 * }</pre>
 *
 * @author Navneev
 * @version 1.0
 * @since 1.0
 */
public class HNSWIndexUtils {

    /** System property key for neighbor shrinking strategy selection */
    private static final String NEIGHBOR_SHRINK_STRATEGY = "vector.neighbor.greedy.shrink";
    
    /** Default value for greedy shrinking (enabled by default for performance) */
    private static final String DEFAULT_GREEDY_SHRINK = "true";

    private static final boolean USE_GREEDY = Boolean.parseBoolean(System.getProperty(NEIGHBOR_SHRINK_STRATEGY,
                                                         DEFAULT_GREEDY_SHRINK));

    /**
     * Determines whether to use greedy neighbor shrinking strategy.
     *
     * <p>When a node's neighbor count exceeds M after adding bidirectional links,
     * the neighbors must be pruned. This method controls which pruning strategy to use:
     *
     * <p><b>Greedy Strategy (true):</b>
     * <ul>
     *   <li>Keeps the M-1 closest neighbors plus the new node</li>
     *   <li>Fast: O(N log N) where N is neighbor count</li>
     *   <li>Simple distance-based selection</li>
     *   <li>May reduce graph quality slightly</li>
     *   <li>Recommended for: Large datasets, performance-critical applications</li>
     * </ul>
     *
     * <p><b>Heuristic Strategy (false):</b>
     * <ul>
     *   <li>Uses diversity-based selection (selectNeighborsHeuristic)</li>
     *   <li>Slower: O(M²) comparisons for diversity checking</li>
     *   <li>Maintains better graph connectivity</li>
     *   <li>Prevents hub nodes and clustering</li>
     *   <li>Recommended for: High recall requirements, smaller datasets</li>
     * </ul>
     *
     * <p><b>Performance Impact:</b>
     * <pre>
     * Strategy    | Build Time | Search Time
     * ------------|------------|-------------
     * Greedy      | Baseline   | Baseline
     * Heuristic   | +15-20%    | -5% (better graph)
     * </pre>
     *
     * <p><b>Configuration:</b>
     * <pre>{@code
     * // Enable greedy (default)
     * java -Dvector.neighbor.greedy.shrink=true MyApp
     *
     * // Enable heuristic for better quality
     * java -Dvector.neighbor.greedy.shrink=false MyApp
     * }</pre>
     *
     * @return true if greedy shrinking should be used, false for heuristic-based shrinking
     */
    public static boolean useGreedyNeighborShrinkingStrategy() {
        return USE_GREEDY;
    }
}
