import os
import glob
import random
from os import getcwd
from os.path import isfile, join, realpath

import numpy as np
import cv2

from classes import classes
from classes import train_path


def mid_frame_extractor(classes, train_path):

    for fields in classes:

        index = classes.index(fields)
        path = os.path.join(train_path, fields, '*i')
        files = glob.glob(path)

        save_path = realpath(getcwd() + '/frames/ucf-101/' + classes[index])
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for fl in files:

            vidcap = cv2.VideoCapture(fl)
            length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
            vidcap.set(1, (length + 2 // 2) // 2)
            success, image = vidcap.read()
            # print(success)

            save_name = os.path.splitext(os.path.basename(fl))[0]
            print(save_path)
            cv2.imwrite(os.path.join(save_path, save_name + ".jpg"), image)
            print('Length of video ' + os.path.basename(fl) + ': '+  str(length) + ' frames.')            
            vidcap.release()

# mid_frame_extractor(classes, train_path)


def optical_flow_generator(classes, train_path):
    for fields in classes:

        index = classes.index(fields)
        path = os.path.join(train_path, fields, '*i')
        files = glob.glob(path)

        save_path = realpath(getcwd() + '/optical_flow/ucf-101/' + classes[index])
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for fl in files:

            max_magnitude = 0

            vidcap = cv2.VideoCapture(fl)
            success, previous_frame = vidcap.read()
            # print(success)

            previous_frame = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
            is_not_last_frame, next_frame = vidcap.read()

            while is_not_last_frame:
                
                next_frame = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
                optical_flow = cv2.calcOpticalFlowFarneback(previous_frame, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                
                magnitude, angle = cv2.cartToPolar(optical_flow[...,0], optical_flow[...,1])
                magnitude = sum(sum(magnitude))
                if (magnitude > max_magnitude):
                    max_magnitude = magnitude
                    max_magnitude_optical_flow = optical_flow
                
                previous_frame = next_frame
                is_not_last_frame, next_frame = vidcap.read()
            
            save_name = os.path.splitext(os.path.basename(fl))[0]
            np.save(os.path.join(save_path, save_name), max_magnitude_optical_flow)
            print('Done processing ' + os.path.basename(fl) + '.')            
            vidcap.release()

# optical_flow_generator(classes, train_path)


def fold_generator(classes, train_path, fold_size):

    path_class_tuples = []

    for fields in classes:
        index = classes.index(fields)
        path = os.path.join(train_path, fields)
        for file_name in os.listdir(path):
            if isfile(join(path, file_name)):
                path_class_tuples.append((join(path, file_name), index))
    random.shuffle(path_class_tuples)

    data_folds = []
    data_fold_paths = []
    data_fold_classes = []

    for fold_index in range(0, fold_size):
        data_folds.append([])
        data_fold_paths.append([])
        data_fold_classes.append([])

    for data_index, path_class_tuple in enumerate(path_class_tuples):
        data_folds[data_index % fold_size].append(path_class_tuple)

    for fold_index in range(0, fold_size):
        data_fold_paths[fold_index] += list(list(zip(*data_folds[fold_index]))[0])
        data_fold_classes[fold_index] += list(list(zip(*data_folds[fold_index]))[1])

    return data_fold_paths, data_fold_classes

fold_generator(classes, train_path, 10)
