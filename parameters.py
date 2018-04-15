"""Data, tensor, and training parameters"""
# Classes and path to the files
CLASSES = ['BrushingTeeth', 'CuttingInKitchen', 'JumpingJack', 'Lunges', 'WallPushups']
TRAIN_PATH = 'data/ucf-101'
TEST_PATH = 'test/own'

# Convolutional neural network training parameters
IMG_SIZE = 128
NUM_CHANNELS = 3
BATCH_SIZE = 32
