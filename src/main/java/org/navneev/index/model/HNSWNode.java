package org.navneev.index.model;

/**
 * Represents a node in the HNSW (Hierarchical Navigable Small World) graph structure.
 * 
 * <p>Each node exists at multiple layers of the graph hierarchy, from layer 0 (base layer)
 * up to its assigned maximum level. The node maintains separate neighbor lists for each
 * layer it participates in, enabling the hierarchical search algorithm.
 * 
 * <p>The HNSW algorithm uses a probabilistic level assignment where higher levels
 * contain fewer nodes, creating a natural hierarchy for efficient approximate
 * nearest neighbor search.
 * 
 * @author Navneev
 * @version 1.0
 * @since 1.0
 */
public class HNSWNode {
    /** Unique identifier for this node in the graph */
    public int id;
    
    /** Highest level this node exists in (0-based indexing) */
    private final int highestNodeLevel;
    
    /** Map of layer number to list of neighbor node IDs for that layer */
    private IntegerList[] neighborsByLayer;

    /**
     * Constructs a new HNSW node with the specified ID and level.
     * 
     * <p>Initializes empty neighbor lists for all layers from 0 to the specified level.
     * The node will participate in layers 0 through level (inclusive).
     * 
     * @param id unique identifier for this node
     * @param highestNodeLevel highest level this node will exist in (must be >= 0)
     */
    public HNSWNode(int id, int highestNodeLevel, int maxNeighbors) {
        this.id = id;
        this.highestNodeLevel = highestNodeLevel;
        // since node level is 0-based index
        this.neighborsByLayer = new IntegerList[highestNodeLevel + 1];
        // make 16 size integer list for the bottom layer as it is expected to have those many connections
        neighborsByLayer[0] = new IntegerList(maxNeighbors + 1);
        for (int l = 1; l <= this.highestNodeLevel; l++) {
            neighborsByLayer[l] = new IntegerList();
        }
    }

    /**
     * Returns the list of neighbor node IDs for the specified layer.
     * 
     * <p>If the node doesn't exist at the specified layer, returns an empty list.
     * The returned list is mutable and can be modified to update connections.
     * 
     * @param layer the layer number to get neighbors for
     * @return list of neighbor node IDs for the specified layer, never null
     */
    public IntegerList getNeighbors(int layer) {
        layerBoundsCheck(layer);
        return neighborsByLayer[layer];
    }

    /**
     * Adds a neighbor connection to this node at the specified layer.
     * 
     * <p>This method only adds the connection in one direction (from this node
     * to the neighbor). For bidirectional connections, the neighbor node must
     * also add this node as its neighbor.
     * 
     * @param layer the layer number to add the connection at
     * @param neighbor the ID of the neighbor node to connect to
     * @throws IllegalArgumentException if the layer is invalid for this node
     */
    public void addNeighbor(int layer, int neighbor) {
        layerBoundsCheck(layer);
        neighborsByLayer[layer].add(neighbor);
    }

    public void updateNeighborhood(int layer, final IntegerList neighborhood) {
        layerBoundsCheck(layer);
        neighborsByLayer[layer] = neighborhood;
    }

    private void layerBoundsCheck(int layer) {
        if (layer < 0 || layer > highestNodeLevel) {
            throw new IllegalArgumentException("Invalid layer number : " + layer + " Max value : " + highestNodeLevel);
        }
    }
}
