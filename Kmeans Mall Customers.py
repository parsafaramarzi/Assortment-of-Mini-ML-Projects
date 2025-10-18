import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = pd.read_csv("Datasets/Mall_Customers.csv")
data = data.drop(columns=["CustomerID","Gender"])

model = KMeans(n_clusters=5)
model.fit(data)
cntr = model.cluster_centers_

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data["Age"],data["Annual Income (k$)"], data["Spending Score (1-100)"], c=model.labels_)
ax.scatter(cntr[:, 0], cntr[:, 1], cntr[:, 2], c='red', marker='X')
plt.show()