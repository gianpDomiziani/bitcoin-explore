"""utils for neural network models"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

loss_obj = tf.keras.losses.MeanAbsoluteError()
def loss(model, x, y, training):
    y_ = model.predict(x, training=training)
    return loss_obj(y_true=y, y_pred=y_)

def grad(model, inputs, targets):
  with tf.GradientTape() as tape:
    loss_value = loss(model, inputs, targets, training=True)
  return loss_value, tape.gradient(loss_value, model.trainable_variables)

def scalar(data):
    data_ = (data-np.mean(data))/np.std(data)
    return data_, np.mean(data), np.std(data)

def unscalar(data, mean, std):
    data_ = data*std+mean
    return data_

def trainStep(model, train):
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