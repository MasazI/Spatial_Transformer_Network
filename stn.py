'''
spatial transfoer network
'''
import os
os.environ['THEANO_FLAGS']='device=gpu0'

# numpy
import numpy as np
np.random.seed(123)

# matplotlib
import matplotlib
import matplotlib.pyplot as plt

# dnn framework
import lasagne
import theano
import theano.tensor as T

# cnn functions
conv = lasagne.layers.Conv2DLayer
pool = lasagne.layers.MaxPool2DLayer

# training parameters
NUM_EPOCHS = 500
BATCH_SIZE = 256
LEARNING_RATE = 0.001
DIM = 60
NUM_CLASSES = 10

# mnist clutterd data
mnist_cluttered = "mnist_cluttered_60x60_6distortions.npz"
# download address
# https://s3.amazonaws.com/lasagne/recipes/datasets/mnist_cluttered_60x60_6distortions.npz
# this is mnist clutterd as 0 to 1 float.


