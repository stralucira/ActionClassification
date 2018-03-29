import cv2
import os
from os import listdir, getcwd
from os.path import isfile, join, realpath
from labels import dataLabels

print(cv2.__version__)

dataPath = realpath(getcwd() + '/data/ucf-101/')
savePath = realpath(getcwd() + '/data/frames/')
allDataPaths = []
allDataLabels = []

for i in dataLabels.keys():
	labelPath = join(dataPath, i)
	for j in listdir(labelPath):
		if isfile(join(labelPath, j)):
			allDataPaths.append(join(labelPath, j))
			allDataLabels.append(dataLabels[i])

#print(len(allDataPaths))
#print(len(allDataLabels))
#print(allDataPaths[0])
#print(allDataLabels[0])

count = 0

#for count in range(0,3):
for count in range(0, len(allDataPaths)):
	vidcap = cv2.VideoCapture(allDataPaths[count])
	length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
	print( 'Frame length of Video ' + str(count) + ' : '+  str(length) + '. ') 
	vidcap.set(1,(length + 2 // 2) // 2)
	success, image = vidcap.read()

	cv2.imwrite(os.path.join(savePath, "midframe_" + str(count) + ".jpg"), image)	# save frame as JPEG file
	count += 1