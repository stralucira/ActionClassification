"""Module to train convolutional neural network"""
import dataset
import time
from datetime import timedelta
import math
import random

import numpy as np
from numpy.random import seed
import tensorflow as tf
from tensorflow import set_random_seed

from parameters import CLASSES
from parameters import TRAIN_PATH
from parameters import TEST_PATH
from parameters import IMG_SIZE
from parameters import NUM_CHANNELS
from parameters import BATCH_SIZE

# Adding Seed so that random initialization is consistent
seed(1)
set_random_seed(2)

# Prepare input data
NUM_CLASSES = len(CLASSES)

# 20% of the data will automatically be used for validation
VALIDATION_SIZE = 0.2

# We shall load all the training and validation images and labels into memory using openCV and use that during training
data = dataset.read_train_sets(TRAIN_PATH, IMG_SIZE, CLASSES, validation_size=VALIDATION_SIZE)
test_batches = dataset.load_test(TEST_PATH, IMG_SIZE, CLASSES, NUM_CHANNELS)

print("Complete reading input data. Will Now print a snippet of it")
print("Number of files in Training-set:\t{}".format(len(data.train.labels)))
print("Number of files in Validation-set:\t{}".format(len(data.valid.labels)))
print("Number of files in Testing-set:\t{}".format(len(test_batches)))

session = tf.Session()
x = tf.placeholder(tf.float32, shape=[None, IMG_SIZE, IMG_SIZE, NUM_CHANNELS], name='x')

# labels
y_true = tf.placeholder(tf.float32, shape=[None, NUM_CLASSES], name='y_true')
y_true_cls = tf.argmax(y_true, axis=1)

# Network graph params
filter_size_conv1 = 3
num_filters_conv1 = 32

filter_size_conv2 = 3
num_filters_conv2 = 32

filter_size_conv3 = 3
num_filters_conv3 = 64

fc_layer_size = 128


def create_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))


def create_biases(size):
    return tf.Variable(tf.constant(0.05, shape=[size]))


def create_convolutional_layer(input,
               num_input_channels, 
               conv_filter_size,        
               num_filters):  
    
    # We shall define the weights that will be trained using create_weights function.
    weights = create_weights(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])
    # We create biases using the create_biases function. These are also trained.
    biases = create_biases(num_filters)

    # Creating the convolutional layer
    layer = tf.nn.conv2d(input=input,
                     filter=weights,
                     strides=[1, 1, 1, 1],
                     padding='SAME')

    layer += biases

    ## We shall be using max-pooling.  
    layer = tf.nn.max_pool(value=layer,
                            ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1],
                            padding='SAME')
    ## Output of pooling is fed to Relu which is the activation function for us.
    layer = tf.nn.relu(layer)

    return layer


def create_flatten_layer(layer):
    # We know that the shape of the layer will be [batch_size img_size img_size num_channels] 
    # But let's get it from the previous layer.
    layer_shape = layer.get_shape()

    # Number of features will be img_height * img_width* num_channels. But we shall calculate it in place of hard-coding it.
    num_features = layer_shape[1:4].num_elements()

    # Now, we Flatten the layer so we shall have to reshape to num_features
    layer = tf.reshape(layer, [-1, num_features])

    return layer


def create_fc_layer(input, num_inputs, num_outputs, use_relu=True):
    
    # Let's define trainable weights and biases.
    weights = create_weights(shape=[num_inputs, num_outputs])
    biases = create_biases(num_outputs)

    # Fully connected layer takes input x and produces wx+b.Since, these are matrices, we use matmul function in Tensorflow
    layer = tf.matmul(input, weights) + biases
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer

layer_conv1 = create_convolutional_layer(input=x,
                num_input_channels=NUM_CHANNELS,
                conv_filter_size=filter_size_conv1,
                num_filters=num_filters_conv1)

layer_conv2 = create_convolutional_layer(input=layer_conv1,
                num_input_channels=num_filters_conv1,
                conv_filter_size=filter_size_conv2,
                num_filters=num_filters_conv2)

layer_conv3 = create_convolutional_layer(input=layer_conv2,
                num_input_channels=num_filters_conv2,
                conv_filter_size=filter_size_conv3,
                num_filters=num_filters_conv3)

layer_flat = create_flatten_layer(layer_conv3)

layer_fc1 = create_fc_layer(input=layer_flat,
                num_inputs=layer_flat.get_shape()[1:4].num_elements(),
                num_outputs=fc_layer_size,
                use_relu=True)

layer_fc2 = create_fc_layer(input=layer_fc1,
                num_inputs=fc_layer_size,
                num_outputs=NUM_CLASSES,
                use_relu=False)

y_pred = tf.nn.softmax(layer_fc2, name='y_pred')

y_pred_cls = tf.argmax(y_pred, axis=1)
session.run(tf.global_variables_initializer())
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=layer_fc2,
                                                    labels=y_true)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

session.run(tf.global_variables_initializer())


def show_progress(epoch, feed_dict_train, feed_dict_validate, val_loss):
    acc = session.run(accuracy, feed_dict=feed_dict_train)
    val_acc = session.run(accuracy, feed_dict=feed_dict_validate)
    msg = "Training Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%},  Validation Loss: {3:.3f}"
    print(msg.format(epoch + 1, acc, val_acc, val_loss))

total_iterations = 0
saver = tf.train.Saver()


def train(num_iteration):
    global total_iterations

    # Parameters for performance measurements
    epoch_f_scores = []
    epoch_recalls = []
    epoch_precisions = []
    graph = tf.get_default_graph()
    y_pred = graph.get_tensor_by_name("y_pred:0")
    x = graph.get_tensor_by_name("x:0")
    y_true = graph.get_tensor_by_name("y_true:0")
    y_test_images = np.zeros((1, len(CLASSES)))

    for i in range(total_iterations,
                   total_iterations + num_iteration):

        x_batch, y_true_batch, _, cls_batch = data.train.next_batch(BATCH_SIZE)
        x_valid_batch, y_valid_batch, _, valid_cls_batch = data.valid.next_batch(BATCH_SIZE)

        feed_dict_tr = {x: x_batch, y_true: y_true_batch}
        feed_dict_val = {x: x_valid_batch, y_true: y_valid_batch}

        session.run(optimizer, feed_dict=feed_dict_tr)

        if i % int(data.train.num_examples/BATCH_SIZE) == 0:
            val_loss = session.run(cost, feed_dict=feed_dict_val)
            epoch = int(i / int(data.train.num_examples/BATCH_SIZE))

            show_progress(epoch, feed_dict_tr, feed_dict_val, val_loss)
            saver.save(session, './ucf101-model')

            """Measure convolutional neural network performance each epoch"""
            confusion_matrix = np.zeros((len(CLASSES), len(CLASSES)))
            precisions = []
            recalls = []
            f_scores = []
            actual_class_counter = 0
            for index, test_batch in enumerate(test_batches):
                feed_dict_testing = {x: test_batch, y_true: y_test_images}
                result = session.run(y_pred, feed_dict=feed_dict_testing)
                for index, predicted_class in enumerate(session.run(tf.argmax(result, axis=1))):
                    confusion_matrix[actual_class_counter, predicted_class] += 1
                actual_class_counter += 1
            for index in range(0, len(CLASSES)):
                true_positif = confusion_matrix[index, index]
                precision = true_positif / np.sum(confusion_matrix[index])
                precisions.append(precision)
                recall = true_positif / np.sum(confusion_matrix[:, index])
                recalls.append(recall)
            for index in range(0, len(CLASSES)):
                f_score = (2 * precisions[index] * recalls[index]) / (precisions[index] + recalls[index])
                f_scores.append(f_score)
            epoch_f_scores.append(f_scores)
            epoch_recalls.append(recalls)
            epoch_precisions.append(precisions)

    total_iterations += num_iteration

    np.savetxt('epoch_f_scores.csv', epoch_f_scores, delimiter=",")
    np.savetxt('epoch_recalls.csv', epoch_recalls, delimiter=",")
    np.savetxt('epoch_precisions.csv', epoch_precisions, delimiter=",")

train(num_iteration=3000)
