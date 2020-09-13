import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
from functools import partial

LABEL_NAME = 'prediction'
filePath = "txsDataWithLabels/txsWithLabels.csv"

def getDataset(filePath, **kwargs):
    dataset = tf.data.experimental.make_csv_dataset(
    filePath,
    batch_size=10,
    label_name=LABEL_NAME,
    num_epochs=1,
    header=True
    )

    return dataset

def show_batch(dataset):
  for batch, label in dataset.take(1):
    for key, value in batch.items():
      print("{:20s}: {}".format(key,value.numpy()))

dataset = getDataset(filePath)
features, labels = next(iter(dataset))

def pack_features_vector(features, labels):
    """Pack the features into a single array"""
    features_ls = []
    for feature in features.values():

        features_ls.append((tf.dtypes.cast(feature, tf.int64)))
    features = tf.stack(features_ls, axis=1)
    return features, labels

train = dataset.map(pack_features_vector)
#BUILD THE MODEL

RegularizedDense = partial(tf.keras.layers.Dense,
                           activation="relu",
                           kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-4),
                           bias_regularizer=tf.keras.regularizers.l2(1e-4),
                           activity_regularizer=tf.keras.regularizers.l2(1e-5))

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=10, activation='relu', input_shape=(7,),
                          kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-4),
                          bias_regularizer=tf.keras.regularizers.l2(1e-4),
                          activity_regularizer=tf.keras.regularizers.l2(1e-5)),
    #tf.keras.layers.Dropout(0.1),
    RegularizedDense(26),
    RegularizedDense(12),
    tf.keras.layers.Dense(1)
])
print(model.summary())
#TRAIN
loss_obj = tf.keras.losses.MeanAbsoluteError()
def loss(model, x, y, training):
    y_ = model(x, training=training)
    return loss_obj(y_true=y, y_pred=y_)

def grad(model, inputs, targets):
  with tf.GradientTape() as tape:
    loss_value = loss(model, inputs, targets, training=True)
  return loss_value, tape.gradient(loss_value, model.trainable_variables)

def scalar(data):
    data = (data-np.mean(data))/np.std(data)
    return data

optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)

train_loss_results = []
train_accuracy_results = []
num_epochs = 251
for epoch in range(num_epochs):
  epoch_loss_avg = tf.keras.metrics.Mean()
  epoch_accuracy = tf.keras.metrics.MeanAbsoluteError()

  # Training loop - using batches of 32
  for x, y in train:
    x = scalar(x)
    # Optimize the model
    loss_value, grads = grad(model, x, y)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # Track progress
    epoch_loss_avg.update_state(loss_value)  # Add current batch loss
    # Compare predicted label to actual label
    # training=True is needed only if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    epoch_accuracy.update_state(y, model(x, training=True))

  # End epoch
  train_loss_results.append(epoch_loss_avg.result())
  train_accuracy_results.append(epoch_accuracy.result())

  if epoch % 50 == 0:
    print("Epoch {:03d}: AVG Loss: {:.3f}, Loss: {:.3%}".format(epoch,
                                                                epoch_loss_avg.result(),
                                                                epoch_accuracy.result()))


fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
fig.suptitle('Training Metrics')
axes[0].set_ylabel("AVG MAE", fontsize=14)
plt.grid()
axes[0].plot(train_loss_results)


axes[1].set_ylabel("MAE", fontsize=14)
plt.grid()
axes[1].plot(train_accuracy_results)

plt.savefig('plots/training_metrics.png')
plt.show()

ls = []
for t, l in train:
    t = scalar(t)
    p = model(t)
    ls.append(loss(model, t, l, True))
    print(l)
    print(p)