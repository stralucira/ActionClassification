import cv2
import os
from os import listdir, getcwd
from os.path import isfile, join, realpath

print(cv2.__version__)

def midFrameExtractor(allDataPaths, allDataLabels):

	savePath = realpath(getcwd() + '/data/training/frames/')
	count = 0

	#for count in range(0,3):
	for count in range(0, len(allDataPaths)):

		print(allDataPaths[count])
		print(allDataLabels[count])

		vidcap = cv2.VideoCapture(allDataPaths[count])
		length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
		print( 'Frame length of Video ' + str(count) + ' : '+  str(length) + '. ') 
		vidcap.set(1,(length + 2 // 2) // 2)
		success, image = vidcap.read()

		cv2.imwrite(os.path.join(savePath, "midframe_" + str(count) + ".jpg"), image)	# save frame as JPEG file
		count += 1