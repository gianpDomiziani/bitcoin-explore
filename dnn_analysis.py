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
test_std, mean_test, std_test = scalar(test)
test_labels_std, mean_labels_test, std_labels_test = scalar(test_labels)







#Fully connected Neural Network
dnnModel = DnnModel()
num_epochs = 500
all_mae_history_1 = []
all_mae_history_2 = []
model_1 = dnnModel.getModel(units=64, inputShape=train.shape[1])
model_2 = dnnModel.getModel(units=124, inputShape=train.shape[1])
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

y_1_t = unscalar(model_1.predict(test_std), mean_labels_test, std_labels_test)
y_2_t = unscalar(model_2.predict(test_std), np.mean(clusters_labels), np.std(clusters_labels))

y_1_smooth = smooth_curve(y_1)
y_2_smooth = smooth_curve(y_2)
y_2_e_smooth = smooth_curve(y_2_e)
plt.plot(y_1_smooth, label='1')
plt.plot(y_2_smooth, label='2')
plt.plot(y_2_e_smooth, label='3')
plt.plot(y_1_t, label='1t')
plt.plot(y_2_t, label='2t')
plt.grid()
plt.legend()
plt.savefig('plots/modelsError.png')
plt.close()


