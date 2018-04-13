import os, glob
import sys, argparse

import cv2
import tensorflow as tf
import numpy as np

from classes import classes
from classes import test_path

# First, pass the path of the image
# dir_path = os.path.dirname(os.path.realpath(__file__))
# image_path = sys.argv[1]
# filename = dir_path + '/' + image_path
image_size = 128
num_channels = 3

# Initialize the confusion matrix
confusion_matrix = np.zeros((len(classes), len(classes)))

# Counter to loop all over the classes
actual_class_counter = 0

for fields in classes:

    images = []

    path = os.path.join(test_path, fields, '*i')
    files = glob.glob(path)
    for filename in files:

        # Reading the image using OpenCV
        # image = cv2.imread(filename)
        # use videos instead of images          
        vidcap = cv2.VideoCapture(filename)
        vidcap.set(1, (int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)) + 2 // 2) // 2)
        success, image = vidcap.read()
        # print(success)

        # take midframe of each video
        # Resizing the image to our desired size and preprocessing will be done exactly as done during training
        image = cv2.resize(image, (image_size, image_size), 0,0, cv2.INTER_LINEAR)
        images.append(image)
    images = np.array(images, dtype=np.uint8)
    images = images.astype('float32')
    images = np.multiply(images, 1.0/255.0)

    # The input to the network is of shape [None image_size image_size num_channels]. Hence we reshape.
    x_batch = images.reshape(len(images), image_size, image_size, num_channels)

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
    y_test_images = np.zeros((1, len(classes)))

    # Creating the feed_dict that is required to be fed to calculate y_pred
    feed_dict_testing = {x: x_batch, y_true: y_test_images}
    result = sess.run(y_pred, feed_dict=feed_dict_testing)
    # result is of this format [probabiliy_of_rose probability_of_sunflower]
    print()
    print('Prediction results for ' + str(len(files)) + ' data(s) in class ' + str(actual_class_counter + 1) + ': ' + classes[actual_class_counter])
    print(result)
    print()

    # Fill in confusion matrix based on the actual and predicted labels
    for index, predicted_class in enumerate(sess.run(tf.argmax(result, axis=1))):
        confusion_matrix[actual_class_counter, predicted_class] += 1

    actual_class_counter += 1

print()
print('Confusion Matrix')
print(confusion_matrix)