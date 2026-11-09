import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# 1. Generate virtual points
X, y = make_blobs(
    n_samples=150,
    centers=3,
    cluster_std=1.0,
    random_state=42
)

# 2. Set number of clusters
k = 3

# 3. Create KMeans model
kmeans = KMeans(
    n_clusters=k,
    random_state=42,
    n_init=10
)

# 4. Train the model
kmeans.fit(X)

# 5. Get cluster labels
labels = kmeans.labels_

# 6. Get cluster centers
centroids = kmeans.cluster_centers_

# 7. Print results
print("Number of clusters:", k)
print("Cluster centers:")
print(centroids)

print("\nCluster assignments:")
for i in range(len(X)):
    print(f"Point {X[i]} -> Cluster {labels[i]}")

# 8. Visualize the results
plt.figure(figsize=(8, 6))

plt.scatter(
    X[:, 0],
    X[:, 1],
    c=labels,
    cmap="viridis",
    s=50
)

plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    c="red",
    marker="X",
    s=200,
    label="Centroids"
)

plt.title("K-Means Clustering on Virtual Points")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True)

plt.show()