import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np

data = pd.read_csv("Datasets/Mall_Customers.csv")
data = data.drop(columns=["CustomerID","Gender"])

inertias = []
for k in range(1, 100):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data)
    inertias.append(kmeans.inertia_)

x = np.arange(1, len(inertias)+1)
y = np.array(inertias)
v1 = np.array([x[-1] - x[0], y[-1] - y[0]])
distances = np.abs(v1[0]*(y[0] - y) - v1[1]*(x[0] - x)) / np.sqrt(v1[0]**2 + v1[1]**2)

best_k = np.argmax(distances) + 1
print(f"Best k (Elbow Method): {best_k}")

plt.plot(range(1, 100), inertias, 'bo-')
plt.xlabel('k'); plt.ylabel('Inertia'); plt.title('Elbow Plot')
plt.savefig("output/kmeans_elbow.png", dpi=300, bbox_inches='tight')
plt.show()

model = KMeans(n_clusters=best_k, random_state=42)
model.fit(data)
cntr = model.cluster_centers_

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data["Age"],data["Annual Income (k$)"], data["Spending Score (1-100)"], c=model.labels_)
ax.scatter(cntr[:, 0], cntr[:, 1], cntr[:, 2], 
           c='black', marker='X', s=200, label='Centroids')
ax.legend()
plt.savefig("output/kmeans_clusters.png", dpi=300, bbox_inches='tight')