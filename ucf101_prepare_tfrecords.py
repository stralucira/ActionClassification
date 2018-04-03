# save data to tfrecord formats
import tensorflow as tf
import glob
import os
#from scipy.ndimage import imread
import numpy as np
#import utils

import cv2
from os import listdir, getcwd
from os.path import isfile, join, realpath
from labels import dataLabels

def load_pathnlabel(filepath, abs_path=None, strip_symb=None):

    # if not strip_symb:
    #     strip_symb = '\r\n'
    # if not abs_path:
    #     abs_path = ''

    # filenames = []
    # labels = []

    # with open(filepath, 'r') as f:
    #     raw_lines = f.readlines()
    #     for line in raw_lines:
    #         line_content = line.strip(strip_symb).split()
    #         s_path, ext = os.path.splitext(line_content[0])     # keep out of the .jpg

    #         s_label = int(line_content[1])-1        #id starting from 0

    #         if abs_path:
    #             s_path=os.path.join(abs_path, s_path)
    #         filenames.append(s_path)
    #         labels.append(s_label)
    # return filenames, labels

    dataPath = realpath(getcwd() + '/data/training/videos/')
    allDataPaths = []
    allDataLabels = []

    for i in dataLabels.keys():
	    labelPath = join(dataPath, i)
	    for j in listdir(labelPath):
		    if isfile(join(labelPath, j)):
			    allDataPaths.append(join(labelPath, j))
			    allDataLabels.append(dataLabels[i])
    return allDataPaths, allDataLabels

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

flags = tf.app.flags
flags.DEFINE_string('annofile_path', '/Users/basaroguz/Developer/ActionClassification/data/train/trainlist01.txt',
                    'annotation file in format (filename label)[/Users/basaroguz/Developer/ActionClassification/data/train/trainlist01.txt]')
flags.DEFINE_string("abs_path", '/Users/basaroguz/Developer/ActionClassification/data', "added to filename in annotation file to make a valid path")
flags.DEFINE_string('save_dir', '/Users/basaroguz/Developer/ActionClassification/data/train/', 'save to [/Users/basaroguz/Developer/ActionClassification/data/train/]')
flags.DEFINE_string('image_format', 'jpg', 'frame format in directories[jpg]')
flags.DEFINE_string('tf_format', 'tfrecord', 'saved tfrecord format[tfrecord]')
flags.DEFINE_integer('min_len', 16, 'min length of videos [16]')

FLAGS = flags.FLAGS


def main(argv=None):

    #save_dir = utils.get_dir(FLAGS.save_dir)

    file_paths, labels = load_pathnlabel(FLAGS.annofile_path, abs_path=FLAGS.abs_path)
    assert len(file_paths) == len(labels)

    # count = 0
    # for count in range(0, len(file_paths)):
    #     print(file_paths[count])
    #     print(labels[count])
    #     count += 1

    save_dir = realpath(getcwd() + '/data/training/records')
    write_videoframes_to_tfrecord(file_paths, labels, save_dir, frame_limit=FLAGS.min_len)


def write_videoframes_to_tfrecord(file_paths, labels, save_dir, frame_limit=None):

    assert len(file_paths) == len(labels), 'Files and labels mismatch'
    #print 'Total # of files {:d}'.format(len(file_paths))

    #save_dir = utils.get_dir(save_dir)

    # filter away
    # if frame_limit:
    #     filtered_pairs = [(f, l) for f, l in zip(file_paths, labels)
    #                       if len(glob.glob(os.path.join(f,'*.{:s}'.format(FLAGS.image_format)))) >= frame_limit]

    #     file_paths, labels = zip(*filtered_pairs)
    #     assert len(file_paths) == len(labels), 'Checking filtering'

    n_files = len(file_paths)
    #print '# of valid videos: {:d}'.format(n_files)

    saveImagePath = realpath(getcwd() + '/data/training/frames/')
    saveRecordPath = realpath(getcwd() + '/data/training/records/')

    for i, s_label, s_filename in zip(range(n_files), labels, file_paths):

        filename_stem = s_filename.split(os.sep)[-1]
        # print('Processing {:d} | {:d} \t {:s}'.format(i, n_files, filename_stem))
        file_list = glob.glob(os.path.join(s_filename, '*.{:s}'.format(FLAGS.image_format)))
        n_images = len(file_list)
        # images = []

        tf_save_name = os.path.join(saveRecordPath, '{:s}.{:s}'.format(filename_stem, FLAGS.tf_format))
        writer = tf.python_io.TFRecordWriter(tf_save_name)

        # for single_filename in file_list:
        #     img = imread(single_filename, mode='RGB')
        #     images.append(img)

        vidcap = cv2.VideoCapture(s_filename)
        length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        print( 'Frame length of Video ' + str(s_filename) + ' : '+  str(length) + '. Label is ' + str(s_label)) 
        vidcap.set(1,(length + 2 // 2) // 2)
        success, image = vidcap.read()
        #cv2.imwrite(os.path.join(saveImagePath, str(s_filename) + ".jpg"), image)	# save frame as JPEG file

        np_image = np.array(image).astype(np.uint8)
        seq_shape = np_image.shape
        #seq_d = seq_shape    # depth, length of seq
        #assert n_images==seq_d
        seq_h = seq_shape[0]    # height
        seq_w = seq_shape[1]    # width
        seq_c = seq_shape[2]    # channels
        image_raw = np_image.tostring()
        record = tf.train.Example(features=tf.train.Features(feature={
        #     'd': _int64_feature(seq_d),
            'h': _int64_feature(seq_h),
            'w': _int64_feature(seq_w),
            'c': _int64_feature(seq_c),
            'label': _int64_feature(int(s_label)),
            'image': _bytes_feature(image_raw),
            'filename': _bytes_feature(filename_stem.encode('utf-8'))}))

        writer.write(record.SerializeToString())

    print('Done')

if __name__ == "__main__":
    tf.app.run()