import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.layers import Conv2D, MaxPooling2D
import pickle
import random
import pandas as pd
import cv2
import requests
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator


np.random.seed(0)
num_of_samples = []
cols = 5
num_classes = 10
num_pixels = 3072


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_cifar10_data():
    unpickled_data = unpickle('cifar-10-batches-py\data_batch_1')
    print(unpickled_data.keys())

def load_cifar100_data():
    unpickled_data = unpickle("cifar-100-python\train")
    print(unpickled_data.keys())


def main():
    load_cifar10_data()
    #load_cifar100_data()
    return 1




if __name__ == '__main__':
    main()