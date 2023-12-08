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

cifar10_classes = [1, 2, 3, 4, 5, 7]  # classes from CIFAR-10
cifar100_classes = [11, 34, 35, 36, 46, 98, 2, 31, 33, 42, 72, 78, 95, 97]  # classes from CIFAR-100


#unpickle the data
cifar101_data = unpickle('cifar_files/cifar-10-batches-py/data_batch_1')
cifar102_data = unpickle('cifar_files/cifar-10-batches-py/data_batch_2')
cifar103_data = unpickle('cifar_files/cifar-10-batches-py/data_batch_3')
cifar104_data = unpickle('cifar_files/cifar-10-batches-py/data_batch_4')
cifar10_meta = unpickle('cifar_files/cifar-10-batches-py/batches.meta')

cifar100_data = unpickle('cifar_files/cifar-100-python/train')
cifar100_meta = unpickle('cifar_files/cifar-100-python/meta')

#combine the data
combined_data_for_cifar10 = np.concatenate((cifar101_data[b'data'], cifar102_data[b'data'], cifar103_data[b'data'], cifar104_data[b'data']), axis=0)
combined_labels_for_cifar10 = np.concatenate((cifar101_data[b'labels'], cifar102_data[b'labels'], cifar103_data[b'labels'], cifar104_data[b'labels']), axis=0)
combined_data = np.concatenate((combined_data_for_cifar10, cifar100_data[b'data']), axis=0)
combined_labels = np.concatenate((combined_labels_for_cifar10, cifar100_data[b'fine_labels']), axis=0)




# Shuffle the data
combined_data = np.array(combined_data)
combined_labels = np.array(combined_labels)
shuffle_index = np.random.permutation(len(combined_labels))
combined_data = combined_data[shuffle_index]
combined_labels = combined_labels[shuffle_index]

print(combined_data.shape)
print(combined_labels.shape)

np.save('cifar10_data.npy', combined_data)
np.save('cifar10_labels.npy', combined_labels)

# Load the data 
data = np.load('cifar10_data.npy', allow_pickle=True)
labels = np.load('cifar10_labels.npy', allow_pickle=True)

#combined_data = np.concatenate((cifar10_data_batch_5[b'data'], cifar100_data[b'data']), axis=0)
#combined_labels = np.concatenate((cifar10_data_batch_5[b'labels'], cifar100_data[b'fine_labels']), axis=0)

#print(cifar10_meta[b'label_names'])
#print(cifar100_meta['fine_label_names'])

# Convert Meta Data to a DataFrame
class_names = cifar10_meta[b'label_names'] + cifar100_meta[b'fine_label_names']
class_names = [x.decode('utf-8') for x in class_names]
class_names = np.array(class_names)
class_names = np.unique(class_names)

#print(class_names)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2 / 0.8, random_state=42)

X_train = X_train.reshape(X_train.shape[0], 32, 32, 3)




print(X_train.shape)
print(X_val.shape)
print(X_test.shape)

assert(X_train.shape[0] == y_train.shape[0]), "the number training images is different from the number of labels"
assert(X_val.shape[0] == y_val.shape[0]), "the number training images is different from the number of labels"
assert(X_test.shape[0] == y_test.shape[0]), "the number training images is different from the number of labels"
assert(X_train.shape[1:] == (32,32,3)), "Training image is not 32,32,3"

num_samples = []
cols = 5
num_classes = len(class_names)

fig, axs = plt.subplots(nrows=num_classes, ncols=cols, figsize=(20,5))

for i in range(cols):
    for j, row in enumerate(axs):
        x_selected = X_train[y_train == j]
        row[i].imshow(np.transpose(data[i].reshape(3, 32, 32), (1, 2, 0)))
        row[i].axis("off")
       
        if i == 2:
            row[i].set_title(str(j) + "-" + class_names[j])
            num_samples.append(len(x_selected))
#plt.tight_layout()
plt.show()


# Visualize All Classes
fig = plt.figure(figsize=(20, 100))
for i in range(100):
    ax = fig.add_subplot(20, 5, i + 1, xticks=[], yticks=[])
    ax.imshow(np.transpose(data[labels == i][0].reshape(3, 32, 32), (1, 2, 0)))
    ax.set_title(i)
plt.show()