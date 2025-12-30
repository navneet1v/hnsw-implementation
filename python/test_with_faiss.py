import numpy as np
import faiss
import time

# Read vectors
vectors = np.loadtxt('vectors.txt', delimiter=',').astype(np.float32)
queries = np.loadtxt('queries.txt', delimiter=',').astype(np.float32)

print(f"Vectors shape: {vectors.shape}")
print(f"Queries shape: {queries.shape}")

# Compute ground truth using brute force
def compute_ground_truth(vectors, queries, k):
    distances = np.linalg.norm(queries[:, np.newaxis] - vectors, axis=2)
    return np.argsort(distances, axis=1)[:, :k]

k = 10
groundtruth = compute_ground_truth(vectors, queries, k)

# Create FAISS HNSW index
dimension = vectors.shape[1]
index = faiss.IndexHNSWFlat(dimension, 16)
index.hnsw.efConstruction = 100
index.hnsw.efSearch = 100

# Add vectors to index one by one
start_time = time.time()
for vector in vectors:
    index.add(np.array([vector]))
end_time = time.time()
build_time_ms = (end_time - start_time) * 1000
print(f"\nIndex build time: {build_time_ms:.2f} ms")

# Search for top 10 nearest neighbors
distances, indices = index.search(queries, k)


print(f"\nFAISS HNSW Results:")
for i in range(min(5, len(queries))):
    print(f"Query {i} - Ground truth: {groundtruth[i][:5]}, FAISS: {indices[i][:5]}")

# Calculate recall
def calculate_recall(groundtruth, results):
    total_relevant = 0
    total_found = 0
    
    for i in range(len(groundtruth)):
        gt_set = set(groundtruth[i])
        for result in results[i]:
            if result in gt_set:
                total_found += 1
        total_relevant += len(groundtruth[i])
    
    return total_found / total_relevant

recall = calculate_recall(groundtruth, indices)
print(f"\nRecall@{k}: {recall:.3f}")

# Print HNSW stats
print(f"\nFAISS HNSW Index Stats:")
print(f"efConstruction: {index.hnsw.efConstruction}")
print(f"efSearch: {index.hnsw.efSearch}")
print(f"max_level: {index.hnsw.max_level}")
print(f"entry_point: {index.hnsw.entry_point}")
print(f"Total vectors: {index.ntotal}")