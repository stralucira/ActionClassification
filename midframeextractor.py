import os
import glob
from os import getcwd
from os.path import realpath

import cv2

from classes import classes
from classes import train_path


def mid_frame_extractor(classes, train_path):

    for fields in classes:

        index = classes.index(fields)
        path = os.path.join(train_path, fields, '*i')
        files = glob.glob(path)

        save_path = realpath(getcwd() + '/frames/' + classes[index])
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for fl in files:

            frame_name = os.path.splitext(os.path.basename(fl))[0]
            
            vidcap = cv2.VideoCapture(fl)
            length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
            print('Length of video ' + frame_name + ': '+  str(length) + ' frames.')
            vidcap.set(1, (length + 2 // 2) // 2)
            success, image = vidcap.read()
            # print(success)

            cv2.imwrite(os.path.join(save_path, frame_name + ".jpg"), image)

mid_frame_extractor(classes, train_path)

# def flow_field_generator(all_data_labels, all_data_paths):
#     save_path = realpath(getcwd() + )
