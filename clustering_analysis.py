from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_boston

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import numpy as np
import pandas as pd


# ==== Preprocessing ====
X = pd.read_csv('txsData/txs_small.csv', header=0, usecols=['n_in', 'n_out', 'amount_in', 'amount_out',
                                                             'change', 'SA', 'diffN'])

X = X.to_numpy()
X = StandardScaler().fit_transform(X)
# ===== K-MEANS =====
# ==== Silhouette analysis ====
range_n_clusters = [2, 3, 4, 5, 6]
silhouette_avg_ls = []
for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1, but in this example it is in a smaller range
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(X)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, cluster_labels)
    silhouette_avg_ls.append(silhouette_avg)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10


    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.1, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(X[:, 2], X[:, 3], marker='.', s=30, lw=0, alpha=0.7,  #a_in and a_out
                c=colors, edgecolor='k')

    # Labeling the clusters
    centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                c="white", alpha=1, s=200, edgecolor='k')

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                    s=50, edgecolor='k')

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')

plt.show()
plt.close(fig)

bestK = silhouette_avg_ls.index(np.max(silhouette_avg_ls))+2
clustering = KMeans(n_clusters=bestK, random_state=10)
clusters_labels = clustering.fit_predict(X)

cls_1 = clustering.cluster_centers_[0]
cls_2 = clustering.cluster_centers_[1]

def getUtility(X, cls1, cls2, clsLabels):
    """Compute the distance (L^2 norm) between a node and the assigned cluster, then it assigns an utility in according
    to such distance. """

    i = 0
    utility = []
    for l in clsLabels:
        if l == 0:
            dist = np.linalg.norm(X[i, :] - cls1)
            utility.append(1 - np.exp(-dist))
            i += 1
        else:
            dist = np.linalg.norm(X[i, :] - cls2)
            utility.append(1 + np.exp(-dist))
            i += 1
    return utility

utility = getUtility(X, cls_1, cls_2, clusters_labels)










