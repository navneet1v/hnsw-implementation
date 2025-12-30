package org.navneev.utils;


import java.security.SecureRandom;
import java.util.ArrayList;
import java.util.List;

/**
 * Generates random layer levels for HNSW graph nodes using exponential decay probability distribution.
 * <p>
 * The level assignment follows the HNSW algorithm where nodes are assigned to layers with exponentially
 * decreasing probability, ensuring a hierarchical structure with fewer nodes at higher layers.
 */
public class HNSWLevelGenerator {
    /** Random number generator for level assignment */
    private final SecureRandom random;
    
    /** Precomputed probability distribution for each layer level */
    private final List<Double> assignProbas;

    /**
     * Creates a level generator with the specified maximum connection's parameter.
     *
     * @param M maximum number of connections per node, used to calculate level multiplier
     */
    public HNSWLevelGenerator(int M) {
        this.random = new SecureRandom();
        this.assignProbas = new ArrayList<>();
        setDefaultProbas((float) (1.0 / Math.log(M)));
    }

    /**
     * Initializes the probability distribution for layer assignment using exponential decay.
     * <p>
     * This method calculates the probability of a node being assigned to each layer level.
     * The probabilities follow an exponential decay pattern, meaning:
     * <ul>
     *   <li>Layer 0 (base layer): Highest probability - most nodes go here</li>
     *   <li>Layer 1: Lower probability - fewer nodes</li>
     *   <li>Layer 2: Even lower probability - even fewer nodes</li>
     *   <li>And so on...</li>
     * </ul>
     * <p>
     * This creates a pyramid-like structure where each higher layer has exponentially fewer nodes,
     * which is essential for the HNSW algorithm's logarithmic search complexity.
     * <p>
     * The formula used is: P(level) = e^(-level/levelMult) * (1 - e^(-1/levelMult))
     * <p>
     * Probabilities below 1e-9 are considered negligible and computation stops.
     *
     * @param levelMult level multiplier controlling decay rate, typically 1/ln(M).
     *                  Smaller values = steeper decay = fewer high-level nodes
     */
    private void setDefaultProbas(float levelMult) {
        for (int level = 0; ; level++) {
            float proba = (float) (Math.exp(-level / levelMult) * (1 - Math.exp(-1 / levelMult)));
            if (proba < 1e-9f) {  // Use float precision
                break;
            }
            assignProbas.add((double) proba);
        }
    }

    /**
     * Generates a random layer level based on the exponential probability distribution.
     * <p>
     * This method works like a weighted lottery using the "subtraction method" for sampling:
     * <ol>
     *   <li>Generate a random number between 0 and 1</li>
     *   <li>Check each layer's probability in order (0, 1, 2, ...)</li>
     *   <li>Subtract each layer's probability from the random number as we go</li>
     *   <li>When the random number becomes less than a layer's probability, return that layer</li>
     * </ol>
     * <p>
     * <b>Why subtraction instead of cumulative range checking?</b>
     * <p>
     * Think of the random number line [0, 1] divided into segments:
     * <pre>
     * |----Layer 0----|--L1--|L2|L3|
     * 0              0.7    0.9 0.98 1.0
     * </pre>
     * <p>
     * <b>Alternative approach (cumulative ranges):</b> Check if random ∈ [0, 0.7), [0.7, 0.9), [0.9, 0.98), etc.
     * This requires maintaining cumulative sums: 0.7, 0.7+0.2=0.9, 0.9+0.08=0.98, ...
     * <p>
     * <b>Subtraction approach (used here):</b> Subtract probabilities and check against 0.
     * This avoids computing cumulative sums and works directly with individual probabilities.
     * <ul>
     *   <li>For Layer 1: Instead of checking if 0.8 ∈ [0.7, 0.9), we subtract 0.7 → 0.1, then check if 0.1 < 0.2</li>
     *   <li>Benefit: Simpler logic, no need to track cumulative bounds, works with the probability list as-is</li>
     * </ul>
     * <p>
     * Example: If probabilities are [0.7, 0.2, 0.08, 0.02]:
     * <ul>
     *   <li>Random 0.5 → 0.5 < 0.7 → Layer 0</li>
     *   <li>Random 0.8 → 0.8 >= 0.7, subtract: 0.8-0.7=0.1 → 0.1 < 0.2 → Layer 1</li>
     *   <li>Random 0.95 → 0.95 >= 0.7, subtract: 0.25 → 0.25 >= 0.2, subtract: 0.05 → 0.05 < 0.08 → Layer 2</li>
     * </ul>
     * <p>
     * Both approaches are mathematically equivalent, but subtraction is simpler to implement.
     *
     * @return layer level (0 for base layer, higher values for upper layers)
     */
    public int getRandomLevel() {
        double f = random.nextDouble();

        for (int level = 0; level < assignProbas.size(); level++) {
            if (f < assignProbas.get(level)) {
                return level;
            }
            f -= assignProbas.get(level);
        }

        // happens with exponentially low probability
        return assignProbas.size() - 1;
    }
}

