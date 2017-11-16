import pandas as pd

df = pd.read_csv('.\Data\oldFaithful.csv', header=None)
X = df.iloc[:,:].values

import matplotlib.pyplot as plt
plt.scatter(X[:, 0],
            X[:, 1],
            c='grey',
            marker='o',
            #edgecolors='black',
            s=50)
plt.title("Old Faithful Eruptions")
plt.xlabel("Eruption Duration (min)")
plt.ylabel("Time to Next Eruption (min)")
plt.grid()
plt.show()

# training with initial random means
from sklearn.cluster import KMeans
km_random = KMeans(n_clusters=2,
            init='random',
            n_init=10,
            max_iter=1000,
            tol=1e-6,
            random_state=0)
y_km_random = km_random.fit_predict(X)
print(y_km_random)
print(km_random.cluster_centers_)

# uses better chosen starting points for the means
km = KMeans(n_clusters=2,
            init='k-means++',
            n_init=10,
            max_iter=1000,
            tol=1e-6,
            random_state=0)
y_km = km.fit_predict(X)
print(y_km)
print(km.cluster_centers_)

plt.clf()
plt.scatter(X[y_km==0, 0],
            X[y_km==0, 1],
            s=50,
            c='deepskyblue',
            marker='s',
            label='cluster 1')
plt.scatter(X[y_km==1, 0],
            X[y_km==1, 1],
            s=50,
            c='orange',
            marker='o',
            label='cluster 2')
plt.scatter(km.cluster_centers_[:, 0],
            km.cluster_centers_[:, 1],
            s=250,
            marker='*',
            c='red',
            label='centroids')
plt.title("Old Faithful Eruptions (k-means)")
plt.xlabel("Eruption Duration (min)")
plt.ylabel("Time to Next Eruption (min)")
plt.legend()
plt.grid()
plt.show()

from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=2,
                     tol=1e-6,
                     max_iter=1000,
                     n_init=10,
                     #weights_init=km.cluster_centers_,
                     means_init=km.cluster_centers_,
                     random_state=0)
gmm.fit(X)
y_gmm = gmm.predict(X)
print(y_gmm)
print(gmm.means_)

plt.clf()
plt.scatter(X[y_gmm==0, 0],
            X[y_gmm==0, 1],
            s=50,
            c='deepskyblue',
            marker='s',
            label='cluster 1')
plt.scatter(X[y_gmm==1, 0],
            X[y_gmm==1, 1],
            s=50,
            c='orange',
            marker='o',
            label='cluster 2')
plt.scatter(gmm.means_[:, 0],
            gmm.means_[:, 1],
            s=250,
            marker='*',
            c='red',
            label='centroids')
plt.title("Old Faithful Eruptions (GMM)")
plt.xlabel("Eruption Duration (min)")
plt.ylabel("Time to Next Eruption (min)")
plt.legend()
plt.grid()
plt.show()
