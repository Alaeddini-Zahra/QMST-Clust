import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('data.csv')

# Extract the features (X) by dropping the string headers
X = data.drop(columns=data.columns[0])

# Find the optimal number of clusters
distortions = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=0)
    kmeans.fit(X)
    distortions.append(kmeans.inertia_)

# Plot the elbow curve
plt.plot(range(1, 11), distortions, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.title('The Elbow Method')
plt.show()

# Find the optimal number of clusters
optimal_clusters = distortions.index(min(distortions)) + 1
print("Optimal number of clusters:", optimal_clusters)
