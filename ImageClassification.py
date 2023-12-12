from keras.datasets import cifar10, cifar100
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.layers import Conv2D, MaxPooling2D
import pickle
import pandas as pd
import random
import cv2
import requests
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator


# load CIFAR-10 data
(x_train_10, y_train_10), (x_test_10, y_test_10) = cifar10.load_data()
print("x_train_10 shape:", x_train_10.shape, "y_train_10 shape:", y_train_10.shape)
# load CIFAR-100 data
(x_train_100, y_train_100), (x_test_100, y_test_100) = cifar100.load_data(label_mode='fine')
print("x_train_100 shape:", x_train_100.shape, "y_train_100 shape:", y_train_100.shape)
class_names_cifar10 = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]
cifar_10_classes_needed = ['automobile', 'bird', 'cat, deer', 'dog', 'truck']
# Create a dictionary mapping numerical labels to class names
label_to_class_name = {i: class_names_cifar10[i] for i in range(len(class_names_cifar10))}
# Apply the mapping to the training and testing labels
y_train_10 = [label_to_class_name[label] for label in y_train_10.flatten()]
y_test_10 = [label_to_class_name[label] for label in y_test_10.flatten()]

class_names = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle',
    'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
    'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
    'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
    'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree',
    'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket',
    'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
    'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor',
    'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
]

classes_needed_cifar100 = ["cattle", "fox", "baby", "boy", "girl", "man", "woman", "rabbit", "squirrel", "maple", "oak", "palm", "pine", "willow", "bicycle", "bus", "motorcycle", "pickup truck", "train", "lawn-mower", "tractor"]

# Create a dictionary mapping numerical labels to class names
label_to_class_name = {i: class_names[i] for i in range(len(class_names))}
# Apply the mapping to the training and testing labels
y_train_100 = [label_to_class_name[label] for label in y_train_100.flatten()]
y_test_100 = [label_to_class_name[label] for label in y_test_100.flatten()]
# Only keep the images that are in the classes_needed
x_train_10 = x_train_10[np.isin(y_train_10, cifar_10_classes_needed).flatten()]
y_train_10 = np.array(y_train_10)[np.isin(y_train_10, cifar_10_classes_needed).flatten()]
x_test_10 = x_test_10[np.isin(y_test_10, cifar_10_classes_needed).flatten()]
y_test_10 = np.array(y_test_10)[np.isin(y_test_10, cifar_10_classes_needed).flatten()]
# Only keep the images that are in the classes_needed
x_train_100 = x_train_100[np.isin(y_train_100, classes_needed_cifar100).flatten()]
y_train_100 = np.array(y_train_100)[np.isin(y_train_100, classes_needed_cifar100).flatten()]
x_test_100 = x_test_100[np.isin(y_test_100, classes_needed_cifar100).flatten()]
y_test_100 = np.array(y_test_100)[np.isin(y_test_100, classes_needed_cifar100).flatten()]

# Resize the images to 32x32x3
x_train_10 = np.array([cv2.resize(img, (32, 32)) for img in x_train_10])
x_test_10 = np.array([cv2.resize(img, (32, 32)) for img in x_test_10])
x_train_100 = np.array([cv2.resize(img, (32, 32)) for img in x_train_100])
x_test_100 = np.array([cv2.resize(img, (32, 32)) for img in x_test_100])

# Print the shape of the data Before
print(x_train_10.shape, y_train_10.shape, x_test_10.shape, y_test_10.shape)
print(x_train_100.shape, y_train_100.shape, x_test_100.shape, y_test_100.shape)

# Join the two datasets
x_train = np.concatenate((x_train_10, x_train_100), axis=0)
y_train = np.concatenate((y_train_10, y_train_100), axis=0)
x_test = np.concatenate((x_test_10, x_test_100), axis=0)
y_test = np.concatenate((y_test_10, y_test_100), axis=0)

# Print the shape of the data After
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

# Visualise the Classes
num_of_samples = []
cols = 5
unique_classes = np.unique(y_train)
fig, axs = plt.subplots(nrows=len(unique_classes), ncols=cols, figsize=(5, 50))
fig.tight_layout()
for i in range(cols):
    for j in unique_classes:
        x_selected = x_train[y_train == j]
        axs[np.where(unique_classes == j)[0][0]][i].imshow(
            x_selected[random.randint(0, len(x_selected) - 1), :, :], cmap=plt.get_cmap("gray")
        )
        axs[np.where(unique_classes == j)[0][0]][i].axis("off")
        if i == 2:
            num_of_samples.append(len(x_selected))
            axs[np.where(unique_classes == j)[0][0]][i].set_title(str(j))
plt.show()

# Data Augmentation 


