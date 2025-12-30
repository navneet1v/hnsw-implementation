package org.navneev.index.model;

import lombok.Builder;
import lombok.ToString;
import lombok.Value;

@Value
@Builder
@ToString
public class HNSWStats {
    int M;
    int efConstruction;
    int dimensions;
    int totalNumberOfNodes;
    int maxLevel;
    int entryPoint;
    long totalBuildTimeInMillis;
}
