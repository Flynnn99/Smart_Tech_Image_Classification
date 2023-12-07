from tensorflow.keras.datasets import cifar10, cifar100
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
import matplotlib.pyplot as plt
import numpy as np
import cv2


# Load CIFAR-10 and CIFAR-100 data
(x_train_10, y_train_10), (x_test_10, y_test_10) = cifar10.load_data()
(x_train_100, y_train_100), (x_test_100, y_test_100) = cifar100.load_data(label_mode='fine')

# Define the classes of interest for CIFAR-10 and CIFAR-100

cifar10_classes = [1, 2, 3, 4, 5, 7]  # classes from CIFAR-10
cifar100_classes = [11, 34, 35, 36, 46, 98, 2, 31, 33, 42, 72, 78, 95, 97]  # classes from CIFAR-100

# Filter out the classes from CIFAR-10 and CIFAR-100
x_train = np.concatenate((x_train_10[np.isin(y_train_10, cifar10_classes).flatten()], 
                          x_train_100[np.isin(y_train_100, cifar100_classes).flatten()]))
y_train = np.concatenate((y_train_10[np.isin(y_train_10, cifar10_classes).flatten()], 
                          y_train_100[np.isin(y_train_100, cifar100_classes).flatten()]))
x_test = np.concatenate((x_test_10[np.isin(y_test_10, cifar10_classes).flatten()], 
                         x_test_100[np.isin(y_test_100, cifar100_classes).flatten()]))
y_test = np.concatenate((y_test_10[np.isin(y_test_10, cifar10_classes).flatten()], 
                         y_test_100[np.isin(y_test_100, cifar100_classes).flatten()]))

# Convert the labels from 2D to 1D
y_train = y_train.flatten()
y_test = y_test.flatten()

# Shuffle the data
shuffle_train = np.random.permutation(len(x_train))
shuffle_test = np.random.permutation(len(x_test))
x_train = x_train[shuffle_train]
y_train = y_train[shuffle_train]
x_test = x_test[shuffle_test]
y_test = y_test[shuffle_test]

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

# Visualize the data
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.imshow(x_train[i])
    plt.xlabel(y_train[i])
plt.show()

# Resize the images to 32x32x3
x_train = np.array([cv2.resize(img, (32, 32)) for img in x_train])
x_test = np.array([cv2.resize(img, (32, 32)) for img in x_test])

# Visualize the data
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.imshow(x_train[i])
    plt.xlabel(y_train[i])
plt.show()

# Split the data into training and testing
x_train = x_train[:10000]
y_train = y_train[:10000]
x_test = x_test[:2000]
y_test = y_test[:2000]


print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)



# Normalize the data
x_train = x_train / 255.0
x_test = x_test / 255.0 

# One-hot encode the labels
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)





