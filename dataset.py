"""Module to generate training and testing datasets"""
import os
import glob
from sklearn.utils import shuffle

import cv2
import numpy as np


def load_train(train_path, image_size, classes):
    """Reads image files for training and returns lists of images and labels"""
    images = []
    labels = []
    img_names = []
    cls = []

    print('Reading training images')
    for fields in classes:
        index = classes.index(fields)
        print('Reading {} files (Index: {})'.format(fields, index))
        path = os.path.join(train_path, fields, '*g')
        files = glob.glob(path)
        for filename in files:

            # Reading the image using OpenCV
            image = cv2.imread(filename)

            # Use midframe of each video instead of images
            # vidcap = cv2.VideoCapture(fl)
            # vidcap.set(1, (int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)) + 2 // 2) // 2)
            # success, image = vidcap.read()
            # print(success)

            image = cv2.resize(image, (image_size, image_size), 0, 0, cv2.INTER_LINEAR)
            image = image.astype(np.float32)
            image = np.multiply(image, 1.0 / 255.0)
            images.append(image)
            label = np.zeros(len(classes))
            label[index] = 1.0
            labels.append(label)
            flbase = os.path.basename(filename)
            img_names.append(flbase)
            cls.append(fields)
    images = np.array(images)
    labels = np.array(labels)
    img_names = np.array(img_names)
    cls = np.array(cls)

    return images, labels, img_names, cls


def load_test(test_path, image_size, classes):
    """Reads image files for testing and returns lists of images and labels"""
    train_batches = []
    image_size = 128
    num_channels = 3

    print('Reading testing images')
    for fields in classes:

        images = []

        path = os.path.join(test_path, fields, '*g')
        files = glob.glob(path)
        for filename in files:

            # Reading the image using OpenCV
            image = cv2.imread(filename)

            # use midframe of each video instead of images
            # vidcap = cv2.VideoCapture(filename)
            # vidcap.set(1, (int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)) + 2 // 2) // 2)
            # success, image = vidcap.read()
            # print(success)

            # Resizing the image to our desired size and preprocessing will be done exactly as done during training
            image = cv2.resize(image, (image_size, image_size), 0, 0, cv2.INTER_LINEAR)
            images.append(image)
        images = np.array(images, dtype=np.uint8)
        images = images.astype('float32')
        images = np.multiply(images, 1.0/255.0)

        # The input to the network is of shape [None image_size image_size num_channels]. Hence we reshape.
        train_batch = images.reshape(len(images), image_size, image_size, num_channels)
        train_batches.append(train_batch)

    return train_batches


class DataSet(object):
    """Dataset Class"""

    def __init__(self, images, labels, img_names, cls):
        self._num_examples = images.shape[0]

        self._images = images
        self._labels = labels
        self._img_names = img_names
        self._cls = cls
        self._epochs_done = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def img_names(self):
        return self._img_names

    @property
    def cls(self):
        return self._cls

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_done(self):
        return self._epochs_done

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size

        if self._index_in_epoch > self._num_examples:
            # After each epoch we update this
            self._epochs_done += 1
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch

        return self._images[start:end], self._labels[start:end], self._img_names[start:end], self._cls[start:end]


def read_train_sets(train_path, image_size, classes, validation_size):
    class DataSets(object):
        pass
    data_sets = DataSets()

    images, labels, img_names, cls = load_train(train_path, image_size, classes)
    images, labels, img_names, cls = shuffle(images, labels, img_names, cls)

    if isinstance(validation_size, float):
        validation_size = int(validation_size * images.shape[0])

    validation_images = images[:validation_size]
    validation_labels = labels[:validation_size]
    validation_img_names = img_names[:validation_size]
    validation_cls = cls[:validation_size]

    train_images = images[validation_size:]
    train_labels = labels[validation_size:]
    train_img_names = img_names[validation_size:]
    train_cls = cls[validation_size:]

    data_sets.train = DataSet(train_images, train_labels, train_img_names, train_cls)
    data_sets.valid = DataSet(validation_images, validation_labels, validation_img_names, validation_cls)

    return data_sets
