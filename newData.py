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

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

cifar10_data_batch_1 = unpickle('cifar-10-batches-py/data_batch_1')
cifar10_data_batch_2 = unpickle('cifar-10-batches-py/data_batch_2')
cifar10_data_batch_3 = unpickle('cifar-10-batches-py/data_batch_3')
cifar10_data_batch_4 = unpickle('cifar-10-batches-py/data_batch_4')
cifar10_data_batch_5 = unpickle('cifar-10-batches-py/data_batch_5')
cifar10_meta = unpickle('cifar-10-batches-py/batches.meta')


cifar100_data = unpickle('cifar-100-python/train')
cifar100_test = unpickle('cifar-100-python/test')
cifar100_meta = unpickle('cifar-100-python/meta')

print(cifar10_data_batch_1.keys())
print(cifar10_data_batch_2.keys())
print(cifar10_data_batch_3.keys())
print(cifar10_data_batch_4.keys())
print(cifar10_data_batch_5.keys())
print(cifar10_meta.keys())

print(cifar100_data.keys())
print(cifar100_test.keys())
print(cifar100_meta.keys())



combined_data = np.concatenate((cifar10_data_batch_1[b'data'], cifar10_data_batch_2[b'data'], cifar10_data_batch_3[b'data'], cifar10_data_batch_4[b'data']), axis=0)
combined_data = np.concatenate((cifar100_data[b'data'], combined_data), axis=0)
combined_labels = np.concatenate((cifar10_data_batch_1[b'labels'], cifar10_data_batch_2[b'labels'], cifar10_data_batch_3[b'labels'], cifar10_data_batch_4[b'labels']), axis=0)
combined_labels = np.concatenate((cifar100_data[b'fine_labels'], combined_labels), axis=0)

# Shuffle the data
combined_data = np.array(combined_data)
combined_labels = np.array(combined_labels)
shuffle_index = np.random.permutation(len(combined_labels))
combined_data = combined_data[shuffle_index]
combined_labels = combined_labels[shuffle_index]

print(combined_data.shape)

np.save('cifar10_data.npy', combined_data)
np.save('cifar10_labels.npy', combined_labels)

combined_data = np.concatenate((cifar10_data_batch_5[b'data'], cifar100_data[b'data']), axis=0)
combined_labels = np.concatenate((cifar10_data_batch_5[b'labels'], cifar100_data[b'fine_labels']), axis=0)

# Convert Meta Data to a DataFrame
class_names = cifar10_meta[b'label_names'] + cifar100_meta[b'fine_label_names']
class_names = [x.decode('utf-8') for x in class_names]
class_names = np.array(class_names)
class_names = np.unique(class_names)

print(class_names)

# Load the data
data = np.load('cifar10_data.npy')
labels = np.load('cifar10_labels.npy')

# Visualize the data
fig = plt.figure(figsize=(20,5))
for i in range(36):
    ax = fig.add_subplot(3, 12, i + 1, xticks=[], yticks=[])
    ax.imshow(np.transpose(data[i].reshape(3, 32, 32), (1, 2, 0)))
    ax.set_title(labels[i])
plt.show()

# Visualize All Classes
fig = plt.figure(figsize=(20, 100))
for i in range(100):
    ax = fig.add_subplot(20, 5, i + 1, xticks=[], yticks=[])
    ax.imshow(np.transpose(data[labels == i][0].reshape(3, 32, 32), (1, 2, 0)))
    ax.set_title(i)
plt.show()
    

# Split the data into training, validation, and test sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2 / 0.8, random_state=42)

# Normalize the data
X_train = X_train.astype('float32') / 255
X_val = X_val.astype('float32') / 255
X_test = X_test.astype('float32') / 255

# One-hot encode the labels
num_classes = len(np.unique(y_train))
y_train = keras.utils.to_categorical(y_train, num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Define the model





