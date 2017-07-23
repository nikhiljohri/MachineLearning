import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.cluster import KMeans
X=np.array([[1,2],
           [1.5,1.8],
           [5,8],
           [8,8],
           [1,0.6],
           [9,11]])
plt.scatter(X[:,0], X[:,1], s=150)
plt.show()
clf = KMeans(n_clusters=2)
clf.fit(X)
centroids = clf.cluster_centers_
labels = clf.labels_
colors=10*["g.","r.","c.","b.","k."]
for i in range(len(X)):
    plt.plot(X[i][0],X[i][1],colors[labels[i]],markersize=20)
plt.scatter(centroids[:,0],centroids[:,1],marker = 'x', s=150, linewidths=5)
plt.show()
#
# plt.scatter(X[:,0], X[:,1],s=150)
# plt.show()
#
# color=10*["g.","r.","c.","b.","k."]
#
# class K_Means:
#     def __init__(self,k=2,tol=0.001,max_iter=300):
#         self.k=k
#         self.tol=tol
#         self.max_iter = max_iter
#
#     def fit(self,data):
#         self.centroid ={}
#         for i in range(self.k):
#             self.centroid[i]=data[i]
#         for i in range(self.max_iter):
#             self.classifications={}
#             for i in range(self.k):
#                 self.classifications[i]=[]
#             for featureset in data:
#                 distances  = [np.linalg.norm(featureset-self.centroid[centroid]) for centroid in self.centroid]
#                 classification = distances.index(min(distances))
#                 self.classifications[classification].append(featureset)
#
#             prev_centroids = dict(self.centroid)
#
#             for classification in self.classifications:
#
#
#
#     def predict(self,data):
