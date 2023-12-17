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
from imgaug import augmenters as iaa # pip install imgaug
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
import matplotlib.image as mpimg
from imgaug import augmenters as iaa



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

cifar_10_classes_needed = ["automobile", "bird", "cat", "deer", "dog","horse", "truck"]
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

classes_needed_cifar100 = ["cattle", "fox", "baby", "boy", "girl", "man", "woman", "rabbit", "squirrel", "maple_tree", "oak_tree", "palm_tree", "pine_tree", "willow_tree", "bicycle", "bus", "motorcycle", "pickup_truck", "train", "lawn_mower", "tractor"]

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

print(set(y_train))  # This will show all unique classes in your training set

# Visualize the Classes
num_of_samples = []
cols = 5
num_classes = np.unique(y_train)
fig, axs = plt.subplots(nrows=len(num_classes), ncols=cols, figsize=(5, 50))
fig.tight_layout()

for i in range(cols):
    for j, label in enumerate(num_classes):
        x_selected = x_train[y_train == label]
        if len(x_selected) > 0:
            img_index = random.randint(0, len(x_selected) - 1)
            axs[j][i].imshow(x_selected[img_index, :, :], cmap=plt.get_cmap('gray'))
            axs[j][i].axis("off")
            if i == 2:
                num_of_samples.append(len(x_selected))
                axs[j][i].set_title(str(label))

plt.show()
#Image Preprocessing
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



img = x_train[random.randint(0, len(x_train) - 1)]
img = cv2.resize(img, (300, 300))
cv2.imshow("Original", img)
img = preprocessing(img)
cv2.imshow("Preprocessing", img)
cv2.waitKey(0)

# Data Exploration

def plot_distribution(y_train):
    # Convert string labels to numeric values
    unique_classes = np.unique(y_train)
    class_to_int = {label: index for index, label in enumerate(unique_classes)}
    y_train_numeric = np.array([class_to_int[label] for label in y_train])

    num_classes = len(unique_classes)  # Number of unique classes

    # Plotting the distribution
    plt.figure(figsize=(15, 15))
    plt.hist(y_train_numeric, bins=num_classes)  # Use numeric labels here
    plt.title("Distribution of the training data")
    plt.xlabel("Class number")
    plt.ylabel("Number of images")
    plt.show()

# Call the function with y_train
plot_distribution(y_train)


def check_image_dimensions(x_train):
    assert (x_train.shape[1:] == (32, 32, 3)), "The dimensions of the images are not 32 x 32 x 3"
    assert (x_train.shape[0] == len(y_train)), "Number of training images is not equal to the number of labels"
    assert (x_test.shape[0] == len(y_test)), "Number of testing images is not equal to the number of labels"

check_image_dimensions(x_train)

def reshape_images(x_train, x_test):
    x_train = x_train.reshape(x_train.shape[0], 32, 32, 3)
    x_test = x_test.reshape(x_test.shape[0], 32, 32, 3)
    return x_train, x_test

x_train, x_test = reshape_images(x_train, x_test)

check_image_dimensions(x_train)


# Data Augmentation 
def flip_random_image(image):
  image = cv2.flip(image, 1)
  return image

img = x_train[random.randint(0, len(x_train) - 1)]
img = cv2.resize(img, (300, 300))
cv2.imshow("Original", img)
img = flip_random_image(img)
cv2.imshow("Flipped", img)
cv2.waitKey(0)



def zoom(image):
  zoom = iaa.Affine(scale = (1,1.3))
  image = zoom.augment_image(image)
  return image

img = x_train[random.randint(0, len(x_train) - 1)]
img = cv2.resize(img, (300, 300))
cv2.imshow("Original", img)
img = zoom(img)
cv2.imshow("Zoomed", img)
cv2.waitKey(0)

def pan(image):
    pan = iaa.Affine(translate_percent = {"x" : (-0.1,0.1), "y" : (-0.1,0.1)})
    image = pan.augment_image(image)
    return image

img = x_train[random.randint(0, len(x_train) - 1)]
img = cv2.resize(img, (300, 300))
cv2.imshow("Original", img)
img = pan(img)
cv2.imshow("Panned", img)
cv2.waitKey(0)

def img_bright(image):
    brightness = iaa.Multiply((0.2,1.2))
    image = brightness.augment_image(image)
    return image

img = x_train[random.randint(0, len(x_train) - 1)]
img = cv2.resize(img, (300, 300))
cv2.imshow("Original", img)
img = img_bright(img)
cv2.imshow("Bright", img)
cv2.waitKey(0)

def image_augment(image):
    # Assuming 'image' is a numpy array representing an image
    if np.random.rand() < 0.5:
        image = zoom(image)
    if np.random.rand() < 0.5:
        image = pan(image)
    if np.random.rand() < 0.5:
        image = img_bright(image)
    if np.random.rand() < 0.5:
        image = flip_random_image(image)
    return image

# Visualize the augmented images
plt.figure(figsize=(10, 5))
for i in range(10):
    augmented_image = image_augment(x_train[random.randint(0, len(x_train) - 1)])
    plt.subplot(2, 5, i + 1)
    plt.imshow(augmented_image)
    plt.axis('off')
plt.tight_layout()
plt.show()

# After data augmentation but before model definition
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)


# Augment the data

datagen = ImageDataGenerator(
    preprocessing_function=image_augment,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    shear_range=0.1,
    rotation_range=10
)

datagen.fit(x_train)

# Visualize the augmented data
plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_train[random.randint(0, len(x_train) - 1)])
    plt.axis('off')
plt.tight_layout()
plt.show()



# Check the dimensions of the data
print(x_train.shape, x_test.shape)

# Visualize the augmented data
plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_train[random.randint(0, len(x_train) - 1)])
    plt.axis('off')
plt.tight_layout()
plt.show()

# Count the occurrence of each class
unique_classes, counts = np.unique(y_train, return_counts=True)

plt.figure(figsize=(15, 10))  # Increase figure size
plt.bar(unique_classes, counts, color='skyblue')
plt.title('Distribution of Classes in the Training Set')
plt.xlabel('Class')
plt.ylabel('Number of Images')
plt.xticks(unique_classes, rotation=90, fontsize=8)  # Rotate labels and adjust font size
plt.tight_layout()  # Adjust layout to fit everything
plt.show()




