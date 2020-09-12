import tensorflow as tf 
import numpy as np

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

dataset = getDataset(filePath)

def show_batch(dataset):
  for batch, label in dataset.take(1):
    for key, value in batch.items():
      print("{:20s}: {}".format(key,value.numpy()))


print(dataset.cardinality())
print(dataset.shape)


#BUILD THE MODEL
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense()
])