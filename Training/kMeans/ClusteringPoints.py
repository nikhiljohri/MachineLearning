import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
X = np.array([[1,1],[2,2.5],[3,1.2],[5.5,6.3],[6,9],[7,6],[8,8]])
k = 2
kmeans = KMeans(n_clusters = k)
kmeans.fit(X);
centroids =  kmeans.cluster_centers_
labels = kmeans.labels_
colors = ['r,','g,']
plt.figure()
for i in range(len(X)):
    plt.plot(X[i,0],X[i,1],colors[labels[i]],markersize = 30)
plt.scatter(centroids[:,0], centroids[:,1], marker="x",s=300,linewidths=5)
plt.show()