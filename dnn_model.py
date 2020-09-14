import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
from functools import partial

LABEL_NAME = 'prediction'
filePath = "txsDataWithLabels/txsWithLabels.csv"

class DnnModel():

    def __init__(self):


        self.inputShape = 0
        self.regDense = partial(tf.keras.layers.Dense,
                           activation="relu",
                           kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-4),
                           bias_regularizer=tf.keras.regularizers.l2(1e-4),
                           activity_regularizer=tf.keras.regularizers.l2(1e-5))



    @staticmethod
    def pack_features_vector(dataset):
        """Pack the features into a single array"""
        features, labels = next(iter(dataset))
        features_ls = []
        for feature in features.values():
            features_ls.append((tf.dtypes.cast(feature, tf.int64)))
        features = tf.stack(features_ls, axis=1)
        return features, labels

    @staticmethod
    def show_batch(dataset):
        for batch, label in dataset.take(1):
            for key, value in batch.items():
                print("{:20s}: {}".format(key, value.numpy()))

    def getModel(self, units, inputShape):

        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(units=units, activation='relu', input_shape=(inputShape,),
                                  kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                  bias_regularizer=tf.keras.regularizers.l2(1e-4),
                                  activity_regularizer=tf.keras.regularizers.l2(1e-5)),
            self.regDense(units),
            tf.keras.layers.Dense(1)
        ])
        model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
        print(model.summary())
        return model


    def getTrain(self, dataset):
        train = dataset.map(self.pack_features_vector)

    @staticmethod
    def getInputShape(dataset):

        features, labels = next(iter(dataset))
        return len(features)






