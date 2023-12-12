from tensorflow.keras.datasets import cifar10, cifar100
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
import matplotlib.pyplot as plt
import numpy as np
import cv2
import random
from imgaug import augmenters as iaa

# Load CIFAR-10
(x_train_10, y_train_10), (x_test_10, y_test_10) = cifar10.load_data()
print(x_train_10.shape, y_train_10.shape, x_test_10.shape, y_test_10.shape)

# Load CIFAR-100
(x_train_100, y_train_100), (x_test_100, y_test_100) = cifar100.load_data()
print(x_train_100.shape, y_train_100.shape, x_test_100.shape, y_test_100.shape)



y_train_10 = y_train_10.reshape(-1,)
print(y_train_10[:5])

cifar_10_classes = ['airplane','automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
cifar_10_classes_needed = ['automobile', 'bird', 'cat, deer', 'dog', 'truck']


#cifar10_classes = [1, 2, 3, 4, 5, 7]  # classes from CIFAR-10
#cifar100_classes = [11, 34, 35, 36, 46, 98, 2, 31, 33, 42, 72, 78, 95, 97]  # classes from CIFAR-100

#dictionary to store the classes
cifar_10_classes_dict = {i:cifar_10_classes[i] for i in range(len(cifar_10_classes))}

#apply dic to the testing and training
y_train_10 = [cifar_10_classes_dict[label] for label in y_train_10.flatten()]
y_test_10 = [cifar_10_classes_dict[label] for label in y_test_10.flatten()]

#filter the data
x_train_10 = x_train_10[np.isin(y_train_10, cifar_10_classes_needed)]
y_train_10 = y_train_10[np.isin(y_train_10, cifar_10_classes_needed)]


#https://www.geeksforgeeks.org/image-classification-using-cifar-10-and-cifar-100-dataset-in-tensorflow/
def plot_sample(X,y,index):
    plt.figure(figsize=(15,2))
    plt.imshow(X[index])
    plt.xlabel(cifar_10_classes_needed[y[index]])
    plt.show()

plot_sample(x_train_10,y_train_10,2)

#normalise
x_train_10 = x_train_10 / 255
x_test_10 = x_test_10 / 255

#Displaying the images
number_of_samples = []
cols = 5
num_classes = 27

X_train = np.concatenate((x_train_10, x_train_100), axis=0)
y_train = np.concatenate((y_train_10, y_train_100), axis=0)


fig,axs = plt.subplots(nrows = num_classes, ncols = cols, figsize = (5,50))
fig.tight_layout()
for i in range(cols):
  for j in num_classes:
    x_selected = X_train[y_train == j]
    axs[j][i].imshow(x_selected[random.randint(0,len(x_selected)-1), :, :], cmap=plt.get_cmap('gray'))
    axs[j][i].axis("off")
    if i == 2:
      number_of_samples.append(len(x_selected))
      axs[j][i].set_title(str(j))
plt.show()


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

