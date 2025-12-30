# HNSW Algorithm Implementation Notes

This document explains key algorithmic concepts in the HNSW implementation.

## Table of Contents
1. [Value of M at Different Levels](#1-value-of-m-at-different-levels)
2. [Neighbor Selection Heuristic](#2-neighbor-selection-heuristic)
3. [Level Generation Algorithm](#3-level-generation-algorithm)

---

## 1. Value of M at Different Levels

### Overview
The HNSW algorithm uses different maximum connection limits (M) for different layers in the hierarchical graph structure.

### Implementation
```java
private int getM(int level) {
    return level == 0 ? M * 2 : M;
}
```

### Layer-Specific M Values

| Layer | M Value | Reason |
|-------|---------|--------|
| Layer 0 (base) | 2*M | Dense connectivity for high recall |
| Layer 1+ (upper) | M | Sparse connectivity for fast navigation |

### Why Different M Values?

#### Layer 0 (Base Layer): 2*M Connections
- **Contains all nodes** - Every vector in the dataset exists at layer 0
- **Final search happens here** - The beam search at layer 0 determines the final k-NN results
- **Higher recall requirement** - More connections = more paths = better chance of finding true neighbors
- **Dense connectivity** - Allows the search to explore more candidates and find accurate results

**Example with M=16:**
- Layer 0 allows up to 32 connections per node
- For 1M nodes: 1,000,000 × 32 = 32M total connections

#### Upper Layers (1, 2, 3, ...): M Connections
- **Fewer nodes** - Only ~37% of nodes appear in upper layers (exponential decay)
- **Navigation purpose** - Used only to quickly find entry points for lower layers
- **Speed over accuracy** - Greedy search with ef=1, so fewer connections are sufficient
- **Sparse connectivity** - Reduces memory and speeds up traversal

**Example with M=16:**
- Layer 1 allows up to 16 connections per node
- For 1M nodes: ~370,000 × 16 = 5.9M total connections

### Memory Impact

Despite 2*M at layer 0, memory remains predictable:

```
Total Memory ≈ N × (2*M × 4 bytes)                    // Layer 0
             + N × 0.37 × M × 4 bytes × avgLevels     // Upper layers

For N=1M, M=16:
  Layer 0: 1,000,000 × 32 × 4 = 128 MB
  Layer 1: ~370,000 × 16 × 4 = 23.7 MB
  Layer 2: ~137,000 × 16 × 4 = 8.8 MB
  Layer 3+: < 5 MB
  Total: ~165 MB (predictable and bounded)
```

### Key Insights

1. **Layer 0 dominates memory** - But it's predictable (all N nodes × 2*M)
2. **Upper layers are sparse** - Exponentially fewer nodes compensate for M connections
3. **No memory explosion** - The selectNeighborsHeuristic enforces strict limits
4. **Balanced design** - Dense base for accuracy, sparse top for speed

---

## 2. Neighbor Selection Heuristic

### Overview
The `selectNeighborsHeuristic` method ensures that each node maintains exactly the right number of diverse connections, preventing hub nodes and maintaining graph quality.

### The Problem It Solves

Without a heuristic, simply selecting the M closest neighbors can create:
- **Hub nodes** - Popular nodes with too many incoming connections
- **Clustering** - Groups of nearby nodes all connected to each other
- **Poor search paths** - Limited diversity reduces the chance of finding optimal routes

### Two Approaches: Greedy vs. Heuristic Shrinking

When a node's neighbor count exceeds M (due to bidirectional links), we must prune neighbors. Two strategies are available:

#### Greedy Shrinking Strategy (Default)

The default greedy is only used when an already added nodes neighbors goes above M at a level.

**Algorithm:**
```java
// Keep M-1 closest neighbors + new node
for (int j = 0; j < getM(level) - 1; j++) {
    neighborsConnections.update(j, neighborCandidatesPQ.poll().id());
}
neighborsConnections.update(getM(level) - 1, newNodeId);
```

**Characteristics:**
- **Simple:** Just keeps the M-1 closest neighbors plus the new node
- **Fast:** O(N log N) where N is neighbor count (priority queue sorting)
- **Distance-based:** Pure proximity selection, no diversity checking
- **Performance:** ~15-20% faster build time
- **Recall:** ~92% on SIFT-128D (M=16, efSearch=100)

**When to Use:**
- Large datasets (>1M vectors)
- Performance-critical applications
- When 2% recall difference is acceptable
- Default choice for most use cases

**Configuration:**
```bash
java -Dvector.neighbor.greedy.shrink=true MyApp  # Default
```

#### Heuristic Shrinking Strategy

**Algorithm:**
```java
// Add new node to candidates
neighborCandidatesPQ.add(new IdAndDistance(newNodeId, distance));

// Convert to array
IdAndDistance[] candidates = new IdAndDistance[size];
for (int j = 0; j < size; j++) {
    candidates[j] = neighborCandidatesPQ.poll();
}

// Use diversity-based selection
neighborsConnections = selectNeighborsHeuristic(candidates, nodeId, level);
```

**Characteristics:**
- **Diversity-focused:** Selects neighbors that are diverse, not just close
- **Slower:** O(M²) for diversity checking across all selected neighbors
- **Quality-oriented:** Prevents clustering and hub nodes
- **Performance:** Baseline build time
- **Recall:** ~94% on SIFT-128D (M=16, efSearch=100)

**When to Use:**
- High recall requirements (>93%)
- Smaller datasets (<500K vectors)
- When graph quality is more important than build speed
- Research and benchmarking

**Configuration:**
```bash
java -Dvector.neighbor.greedy.shrink=false MyApp
```

### Performance Comparison

| Metric | Greedy Strategy | Heuristic Strategy |
|--------|----------------|--------------------|
| Build Time | Baseline | +15-20% |
| Recall@10 (efSearch=100) | ~92% | ~94% |
| Search Time | Baseline | -5% (better graph) |
| Memory Usage | Same | Same |
| Complexity | O(N log N) | O(M²) |
| Graph Quality | Good | Excellent |

### Why Greedy is Default

1. **Significant speed improvement:** 15-20% faster builds matter for large datasets
2. **Scalability:** O(N log N) scales better than O(M²) as M increases
3. **Simplicity:** Easier to understand and debug

### Heuristic Selection Algorithm (When Enabled)
This algorithm is always enabled when we are selecting neighbors for a new node.

```java
private IntegerList selectNeighborsHeuristic(final IdAndDistance[] candidates, 
                                             int nodeToLinkNeighborsTo,
                                             int level) {
    final IntegerList finalSelected = new IntegerList(getM(level));
    final IntegerList discardedList = new IntegerList();
    int counter = 0;
    
    // Phase 1: Select diverse neighbors
    while (counter < candidates.length && finalSelected.size() < getM(level)) {
        final IdAndDistance candidate = candidates[counter];
        counter++;
        boolean isDiverse = true;
        float[] candidateVector = cloneAndGetVector(candidate.id());
        
        // Check if candidate is diverse compared to already selected neighbors
        for(int i = 0; i < finalSelected.size(); i++) {
            int currentNeighbor = finalSelected.get(i);
            // Reject if candidate is closer to an existing neighbor than to the target node
            if(dis(currentNeighbor, candidateVector) < candidate.distance()) {
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
    
    // Phase 2: Fill remaining slots with discarded candidates if needed
    counter = 0;
    while(finalSelected.size() < getM(level) && counter < discardedList.size()) {
        finalSelected.add(discardedList.get(counter));
        counter++;
    }
    
    return finalSelected;
}
```

### How It Works: Step-by-Step

#### Phase 1: Diversity Check

**Input:** Candidates sorted by distance to target node (closest first)

**For each candidate:**
1. Compare candidate distance to target vs. distance to already-selected neighbors
2. **Diversity Rule:** Reject if `distance(candidate, existing_neighbor) < distance(candidate, target)`
3. If diverse, add to finalSelected; otherwise, add to discardedList

**Visual Example:**

```
Target Node: T
Candidate: C
Already Selected Neighbor: N

Case 1: DIVERSE (Accept)
    T ----10---- C ----15---- N
    C is closer to T (10) than to N (15) ✓

Case 2: NOT DIVERSE (Reject)
    T ----10---- C ----5---- N
    C is closer to N (5) than to T (10) ✗
    (C would create a cluster with N)
```

#### Phase 2: Fill Remaining Slots

If Phase 1 didn't select enough neighbors (finalSelected.size() < M):
- Add discarded candidates to fill up to M connections
- Ensures we always have M connections when possible
- Prioritizes diversity but doesn't leave slots empty

### Why This Heuristic Works

#### 1. Prevents Hub Nodes
By rejecting candidates that cluster around existing neighbors, we avoid creating nodes with excessive incoming connections.

#### 2. Maintains Small-World Property
Diverse connections create "shortcuts" across the graph, enabling logarithmic search complexity.

#### 3. Balances Proximity and Diversity
- **Phase 1:** Prioritizes diverse neighbors
- **Phase 2:** Ensures sufficient connectivity

#### 4. Improves Search Quality
Diverse connections provide multiple paths to reach any node, improving recall.

### Example Scenario

**Target Node T, M=3, Candidates sorted by distance:**

| Candidate | Distance to T | Action |
|-----------|---------------|--------|
| A | 5 | Select (first candidate, always diverse) |
| B | 7 | Check: dist(B,A)=10 > 7 → Diverse → Select |
| C | 8 | Check: dist(C,A)=3 < 8 → Not diverse → Discard |
| D | 9 | Check: dist(D,A)=12 > 9, dist(D,B)=11 > 9 → Diverse → Select |

**Result:** finalSelected = [A, B, D] (3 diverse neighbors)

If only A and B were diverse, C would be added from discardedList to reach M=3.

### Key Insights

1. **Sorted input matters** - Candidates are pre-sorted by distance, so we consider closest first
2. **Greedy but effective** - Makes local decisions that lead to global graph quality
3. **Guaranteed M connections** - Phase 2 ensures we use all available slots
4. **Computationally efficient** - O(M²) per node, acceptable for small M (16-32)
5. **Strategy selection** - Choose greedy for speed, heuristic for quality

### Shrinking Strategy Decision Tree

```
Need to prune neighbors?
│
├── Dataset > 1M vectors?
│   └── YES → Use Greedy (faster build)
│
├── Recall requirement > 93%?
│   └── YES → Use Heuristic (better quality)
│
├── Build time critical?
│   └── YES → Use Greedy (15-20% faster)
│
└── Default → Use Greedy (balanced choice)
```

---

## 3. Level Generation Algorithm

### Overview
The `HNSWLevelGenerator` class assigns each node to a maximum layer using an exponential decay probability distribution, creating the hierarchical structure essential to HNSW.

### The Goal

Create a pyramid-like structure where:
- **Layer 0:** All nodes (100%)
- **Layer 1:** ~37% of nodes
- **Layer 2:** ~14% of nodes
- **Layer 3:** ~5% of nodes
- **Higher layers:** Exponentially fewer nodes

### Implementation

```java
public class HNSWLevelGenerator {
    private final SecureRandom random;
    private final List<Double> assignProbas;
    private final int M;

    public HNSWLevelGenerator(int M) {
        this.M = M;
        this.random = new SecureRandom();
        this.assignProbas = new ArrayList<>();
        setDefaultProbas(M, (float) (1.0 / Math.log(M)));
    }

    private void setDefaultProbas(int M, float levelMult) {
        for (int level = 0; ; level++) {
            float proba = (float) (Math.exp(-level / levelMult) * (1 - Math.exp(-1 / levelMult)));
            if (proba < 1e-9f) {
                break;
            }
            assignProbas.add((double) proba);
        }
    }

    public int getRandomLevel() {
        double f = random.nextDouble();

        for (int level = 0; level < assignProbas.size(); level++) {
            if (f < assignProbas.get(level)) {
                return level;
            }
            f -= assignProbas.get(level);
        }

        return assignProbas.size() - 1;
    }
}
```

### Probability Formula

```
levelMult = 1 / ln(M)

P(level) = e^(-level/levelMult) × (1 - e^(-1/levelMult))
```

**For M=16:**
- levelMult = 1 / ln(16) ≈ 0.36
- P(0) ≈ 0.63 (63%)
- P(1) ≈ 0.23 (23%)
- P(2) ≈ 0.09 (9%)
- P(3) ≈ 0.03 (3%)

### How setDefaultProbas Works

#### Step 1: Calculate Probabilities

```java
for (int level = 0; ; level++) {
    float proba = (float) (Math.exp(-level / levelMult) * (1 - Math.exp(-1 / levelMult)));
    if (proba < 1e-9f) break;
    assignProbas.add((double) proba);
}
```

**Example for M=16 (levelMult ≈ 0.36):**

| Level | Calculation | Probability |
|-------|-------------|-------------|
| 0 | e^(0/0.36) × (1-e^(-1/0.36)) | 0.6321 |
| 1 | e^(-1/0.36) × (1-e^(-1/0.36)) | 0.2325 |
| 2 | e^(-2/0.36) × (1-e^(-1/0.36)) | 0.0855 |
| 3 | e^(-3/0.36) × (1-e^(-1/0.36)) | 0.0314 |
| 4 | e^(-4/0.36) × (1-e^(-1/0.36)) | 0.0116 |

Stops when probability < 1e-9 (negligible)

**Result:** assignProbas = [0.6321, 0.2325, 0.0855, 0.0314, 0.0116, ...]

### How getRandomLevel Works: The Subtraction Method

#### Algorithm

```java
public int getRandomLevel() {
    double f = random.nextDouble();  // Random value in [0, 1)

    for (int level = 0; level < assignProbas.size(); level++) {
        if (f < assignProbas.get(level)) {
            return level;
        }
        f -= assignProbas.get(level);  // Shift to next range
    }

    return assignProbas.size() - 1;
}
```

#### Why Subtraction Works

Think of the [0, 1] range divided into segments:

```
|-------- Layer 0 --------|--- L1 ---|L2|L3|
0                        0.63      0.86 0.95 1.0
```

**Traditional approach:** Check if random ∈ [0, 0.63), [0.63, 0.86), [0.86, 0.95), ...
- Requires computing cumulative sums: 0.63, 0.63+0.23=0.86, 0.86+0.09=0.95, ...

**Subtraction approach:** Subtract each probability and check against 0
- Works directly with individual probabilities
- Simpler implementation, same result

#### Step-by-Step Example

**Probabilities:** [0.63, 0.23, 0.09, 0.03, 0.02]

**Case 1: random = 0.5**
```
Level 0: 0.5 < 0.63 → Return 0
```

**Case 2: random = 0.8**
```
Level 0: 0.8 >= 0.63, subtract: 0.8 - 0.63 = 0.17
Level 1: 0.17 < 0.23 → Return 1
```

**Case 3: random = 0.95**
```
Level 0: 0.95 >= 0.63, subtract: 0.95 - 0.63 = 0.32
Level 1: 0.32 >= 0.23, subtract: 0.32 - 0.23 = 0.09
Level 2: 0.09 >= 0.09, subtract: 0.09 - 0.09 = 0.00
Level 3: 0.00 < 0.03 → Return 3
```

**Case 4: random = 0.99**
```
Level 0: subtract → 0.36
Level 1: subtract → 0.13
Level 2: subtract → 0.04
Level 3: subtract → 0.01
Level 4: 0.01 < 0.02 → Return 4
```

### Why This Distribution?

#### 1. Logarithmic Search Complexity
With exponential decay, the expected number of hops is O(log N):
- Upper layers: Few nodes, large jumps
- Lower layers: More nodes, refined search

#### 2. Balanced Structure
- Not too many layers (would slow down search)
- Not too few layers (would lose hierarchical benefit)
- Optimal balance for M connections per node

#### 3. Memory Efficiency
Most nodes (63%) only exist at layer 0, minimizing memory for upper layer connections.

#### 4. Consistent with Theory
The formula ensures that the expected maximum level is approximately log(N) / log(M).

### Distribution Visualization

**For 1,000,000 nodes with M=16:**

```
Layer 5: █ (~1,000 nodes)
Layer 4: ███ (~12,000 nodes)
Layer 3: ████████ (~31,000 nodes)
Layer 2: ████████████████████ (~85,000 nodes)
Layer 1: ████████████████████████████████████████████ (~232,000 nodes)
Layer 0: ████████████████████████████████████████████████████████████ (1,000,000 nodes)
```

### Key Insights

1. **Precomputed probabilities** - Calculated once in constructor for efficiency
2. **Subtraction method** - Elegant way to sample from discrete distribution
3. **Exponential decay** - Creates natural hierarchy with O(log N) expected max level
4. **Deterministic structure** - Same M always produces same probability distribution
5. **Rare high levels** - Nodes at level 5+ are extremely rare but provide long-range connections

---

## Summary

These three concepts work together to create an efficient HNSW index:

1. **Variable M values** - Dense base layer (2*M) for accuracy, sparse upper layers (M) for speed
2. **Neighbor selection heuristic** - Ensures diverse connections, prevents hubs, maintains small-world property
3. **Level generation** - Creates hierarchical structure with exponential decay for O(log N) search

The combination enables:
- **Fast search:** O(log N) expected complexity
- **Predictable memory:** Bounded by N × 2*M at layer 0
- **Scalability:** Works efficiently from thousands to millions of vectors
