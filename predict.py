import os, glob
import sys, argparse

import tensorflow as tf
import numpy as np

from dataset import load_test
from parameters import CLASSES
from parameters import TEST_PATH
from parameters import IMG_SIZE

# Initialize the confusion matrix
confusion_matrix = np.zeros((len(CLASSES), len(CLASSES)))

# Counter to loop all over the classes
actual_class_counter = 0

# Load all the test batches
test_batches = load_test(TEST_PATH, IMG_SIZE, CLASSES)

for index, test_batch in enumerate(test_batches):

    # Let us restore the saved model
    sess = tf.Session()
    # Step-1: Recreate the network graph. At this step only graph is created.
    saver = tf.train.import_meta_graph('ucf101-model.meta')
    # Step-2: Now let's load the weights saved using the restore method.
    saver.restore(sess, tf.train.latest_checkpoint('./'))

    # Accessing the default graph which we have restored
    graph = tf.get_default_graph()

    # Now, let's get hold of the op that we can be processed to get the output.
    # In the original network y_pred is the tensor that is the prediction of the network
    y_pred = graph.get_tensor_by_name("y_pred:0")

    # Let's feed the images to the input placeholders
    x = graph.get_tensor_by_name("x:0") 
    y_true = graph.get_tensor_by_name("y_true:0") 
    y_test_images = np.zeros((1, len(CLASSES)))

    # Creating the feed_dict that is required to be fed to calculate y_pred
    feed_dict_testing = {x: test_batch, y_true: y_test_images}
    result = sess.run(y_pred, feed_dict=feed_dict_testing)
    # result is of this format [probabiliy_of_rose probability_of_sunflower]
    print()
    print('Prediction results for ' + str(len(test_batch)) + ' data(s) in class ' + str(actual_class_counter + 1) + ': ' + CLASSES[actual_class_counter])
    print(result)
    print()

    # Fill in confusion matrix based on the actual and predicted labels
    for index2, predicted_class in enumerate(sess.run(tf.argmax(result, axis=1))):
        confusion_matrix[actual_class_counter, predicted_class] += 1

    actual_class_counter += 1

print()
print('Confusion Matrix')
print(confusion_matrix)

PRECISIONS = []
RECALLS = []

for index in range(0, len(CLASSES)):
    true_positif = confusion_matrix[index, index]

    precision = true_positif / np.sum(confusion_matrix[index])
    PRECISIONS.append(precision)

    recall = true_positif / np.sum(confusion_matrix[:, index])
    RECALLS.append(recall)

print()
print('Precisions')
print(PRECISIONS)

print()
print('Recalls')
print(RECALLS)

F_SCORES = []

for index in range(0, len(CLASSES)):
    f_score = (2 * PRECISIONS[index] * RECALLS[index]) / (PRECISIONS[index] + RECALLS[index])
    F_SCORES.append(f_score)

print()
print('F-Scores')
print(F_SCORES)
