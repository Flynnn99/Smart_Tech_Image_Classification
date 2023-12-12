import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import random
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.layers import Conv2D, MaxPooling2D
import pandas as pd
import cv2
import requests
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from imgaug import augmenters as iaa


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

cifar10_classes = [1, 2, 3, 4, 5, 7]  # classes from CIFAR-10
cifar100_classes = [11, 34, 35, 36, 46, 98, 2, 31, 33, 42, 72, 78, 95, 97]  # classes from CIFAR-100


#unpickle the data
cifar101_data = unpickle('cifar_files/cifar-10-batches-py/data_batch_1')
cifar102_data = unpickle('cifar_files/cifar-10-batches-py/data_batch_2')
cifar103_data = unpickle('cifar_files/cifar-10-batches-py/data_batch_3')
cifar104_data = unpickle('cifar_files/cifar-10-batches-py/data_batch_4')
cifar10_meta = unpickle('cifar_files/cifar-10-batches-py/batches.meta')

combine_cifar10_data = np.concatenate((cifar101_data[b'data'], cifar102_data[b'data'], cifar103_data[b'data'], cifar104_data[b'data']), axis=0)
#add labels to the data
combine_cifar10_data = np.concatenate((combine_cifar10_data, np.concatenate((cifar101_data[b'labels'], cifar102_data[b'labels'],cifar103_data[b'labels'], cifar104_data[b'labels']), axis=0)), axis=1)

cifar100_data = unpickle('cifar_files/cifar-100-python/train')
cifar100_test = unpickle('cifar_files/cifar-100-python/test')
cifar100_meta = unpickle('cifar_files/cifar-100-python/meta')

for item in combine_cifar10_data:
    print(item, type(combine_cifar10_data[item]))


def greyscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def equalize(img):
    img = cv2.equalizeHist(img)
    return img

def preprocessing(img):
    img = greyscale(img)
    img = equalize(img)
    img = img/255
    return img


#Image Augmentation
def flip_random_image(image):
  image = cv2.flip(image, 1)
  return image

def zoom(image):
  zoom = iaa.Affine(scale = (1,1.3))
  image = zoom.augment_image(image)
  return image

def pan(image):
    pan = iaa.Affine(translate_percent = {"x" : (-0.1,0.1), "y" : (-0.1,0.1)})
    image = pan.augment_image(image)
    return image
def img_bright(image):
    brightness = iaa.Multiply((0.2,1.2))
    image = brightness.augment_image(image)
    return image

def image_augment():
    image = mpimg.imread(image)
    if np.random.rand() < 0.5:
        image = zoom(image)
    if np.random.rand() < 0.5:
        image = pan(image)
    if np.random.rand() < 0.5:
        image = img_bright(image)
    if np.random.rand() < 0.5:
        image = flip_random_image(image)
    return image,


def alpha_model():
    model = Sequential()
    model.add(Conv2D(60, (5,5), input_shape=(32,32,1), activation='relu'))
    model.add(Conv2D(60, (5,5), input_shape=(32,32,1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(30, (5,5), input_shape=(32,32,1), activation='relu'))
    model.add(Conv2D(30, (5,5), input_shape=(32,32,1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(500, activation ='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model




