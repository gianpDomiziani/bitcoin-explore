from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

import matplotlib.cm as cm

from dnn_model import DnnModel
from utils import *

LABEL_NAME = 'prediction'
filePath = "txsDataWithLabels/txsWithLabels.csv"

def getDataset(filePath, **kwargs):
    dataset = tf.data.experimental.make_csv_dataset(filePath,
                                                    batch_size=10,
                                                    label_name=LABEL_NAME,
                                                    num_epochs=1,
                                                    header=True)
    return dataset



#Download dataset
(train, train_labels), (test, test_labels) = tf.keras.datasets.boston_housing.load_data()
train_std, mean_train, std_train = scalar(train)
train_labels_std, mean_labels, std_labels = scalar(train_labels)
#test, test_labels = scalar(test), scalar(test_labels)

#KMEANS
range_n_clusters = [2, 3, 4, 5, 6]
silhouette_avg_ls = []
for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns
    #fig, (ax1, ax2) = plt.subplots(1, 2)
    #fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    #ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    #ax1.set_ylim([0, len(train_std) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(train_std)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(train_std, cluster_labels)
    silhouette_avg_ls.append(silhouette_avg)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(train_std, cluster_labels)

    y_lower = 10

    """
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
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(train_std[:, 0], train_std[:, 1], marker='.', s=30, lw=0, alpha=0.7,
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

#plt.savefig('plots/silhouette.png')
#plt.close(fig)
"""


bestK = silhouette_avg_ls.index(np.max(silhouette_avg_ls))

clustering = KMeans(n_clusters=bestK+2, random_state=10)
clusters_labels = clustering.fit_predict(train_std)


#Fully connected Neural Network
dnnModel = DnnModel()
num_epochs = 500
all_mae_history_1 = []
all_mae_history_2 = []
model_1 = dnnModel.getModel(units=64, inputShape=train.shape[1])
model_2 = dnnModel.getModel(units=64, inputShape=train.shape[1])
history_1 = model_1.fit(train_std, train_labels_std,
                        epochs=num_epochs,
                        batch_size=24,
                        verbose=0)

history_2 = model_2.fit(train_std, cluster_labels,
                        epochs=num_epochs,
                        batch_size=24,
                        verbose=0)
mae_history_1 = history_1.history['mae']
mae_history_2 = history_2.history['mae']
all_mae_history_1.append(mae_history_1)
all_mae_history_2.append(mae_history_2)




def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

smoothed_points_1 = smooth_curve(all_mae_history_1[0])
smoothed_points_2 = smooth_curve(all_mae_history_2[0])
plt.plot(range(num_epochs), smoothed_points_1, label='1')
plt.plot(range(num_epochs), smoothed_points_2, label='2')
plt.xlabel('EPOCHS')
plt.ylabel('MAE')
plt.legend()
plt.grid()
plt.savefig('plots/MAE.png')
plt.close()

y_1 = unscalar(model_1.predict(train_std), mean_labels, std_labels)
y_2 = unscalar(model_2.predict(train_std), np.mean(clusters_labels), np.std(clusters_labels))
y_2_e = unscalar(model_2.predict(train_std), mean_labels, std_labels)

y_1_smooth = smooth_curve(y_1)
y_2_smooth = smooth_curve(y_2)
y_2_e_smooth = smooth_curve(y_2_e)
plt.plot(y_1_smooth, label='1')
plt.plot(y_2_smooth, label='2')
plt.plot(y_2_e_smooth, label='3')
plt.grid()
plt.legend()
plt.savefig('plots/modelsError.png')
plt.close()


