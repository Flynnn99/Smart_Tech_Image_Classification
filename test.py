from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
import matplotlib.pyplot as plt
import numpy as np
import cv2
import random
import matplotlib.image as mpimg
from imgaug import augmenters as iaa
from sklearn.model_selection import train_test_split
import pickle

 
# Function to load a CIFAR batch
def load_cifar_batch(file_path):
    with open(file_path, 'rb') as file:
        batch = pickle.load(file, encoding='latin1')
    return batch
 
# Load all CIFAR-10 batches
cifar10_batches = [load_cifar_batch(f'cifar_files/cifar-10-batches-py/data_batch_{i}') for i in range(1, 6)]
 
# Load CIFAR-100 data
cifar100_data = load_cifar_batch('cifar_files/cifar-100-python/train')
 
# Load CIFAR-10 test batch
cifar10_test_batch = load_cifar_batch('cifar_files/cifar-10-batches-py/test_batch')
 
# Combine data from CIFAR-10 and CIFAR-100
x_train_10 = np.concatenate([batch['data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1) for batch in cifar10_batches], axis=0)
y_train_10 = np.concatenate([batch['labels'] for batch in cifar10_batches])
 
x_train_100 = np.array(cifar100_data['data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1))
y_train_100 = np.array(cifar100_data['fine_labels'])
 
# Combine CIFAR-10 and CIFAR-100 labels
y_train_combined = np.concatenate([y_train_10, y_train_100 + 10])  # Offset CIFAR-100 labels by 10
 
# Concatenate the data
x_train = np.concatenate((x_train_10, x_train_100), axis=0)
x_test = cifar10_test_batch['data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
y_test = cifar10_test_batch['labels']
 
# Class names for CIFAR-10 and CIFAR-100
cifar_10_class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
cifar_100_class_names = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle',
                         'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar',
                         'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab',
                         'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl',
                         'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
                         'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom',
                         'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree',
                         'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket',
                         'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
                         'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television',
                         'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree',
                         'wolf', 'woman', 'worm']
 
# Visualize random images
def show_random_images(x, y, class_names, num_images=5):
    plt.figure(figsize=(10, 5))
    for i in range(num_images):
        index = np.random.randint(0, len(x))
        label = y[index]  
        plt.subplot(1, num_images, i + 1)
        plt.imshow(x[index])
        plt.title(class_names[label])
        plt.axis('off')
    plt.show()
 
# Visualize random images
show_random_images(x_train, y_train_combined, cifar_10_class_names + cifar_100_class_names, num_images=5)

# Function to plot the distribution of the data
def plot_distribution(labels, title='Distribution of the training set', xlabel='Class Number', ylabel='Number of images'):
    # Count the number of images per class
    unique, counts = np.unique(labels, return_counts=True)
    # Plot the distribution
    plt.figure(figsize=(15, 5))
    plt.bar(unique, counts, color='blue')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(unique)  # Ensure all class numbers are shown on the x-axis
    plt.show()

plot_distribution(y_train_combined, title='Distribution of the training set', xlabel='Class Number', ylabel='Number of images')

# Print the shape of the data After
print(x_train.shape)

#split the data
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train_combined, test_size=0.2, random_state=42)
print(x_train.shape, y_train.shape, x_val.shape, y_val.shape)

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

# Preprocess the data
x_train = np.array(list(map(preprocessing, x_train)))
x_val = np.array(list(map(preprocessing, x_val)))
x_test = np.array(list(map(preprocessing, x_test)))

plt.imshow(x_train[random.randint(0, len(x_train)-1)])
plt.axis("off")
plt.show()
print(x_train.shape)

#reshape 
x_train = x_train.reshape(x_train.shape[0],32,32,1)
X_val = x_val.reshape(x_val.shape[0],32,32,1)
X_train = x_train.reshape(x_train.shape[0],32,32,1)

print(x_train.shape)
print(x_val.shape)
print(x_test.shape)


# #Image Augmentation
# def flip_random_image(image):
#   image = cv2.flip(image, 1)
#   return image

# def zoom(image):
#     zoom = iaa.Affine(scale = (1,1.3))
#     image = zoom.augment_image(image)
#     return image

# def pan(image_to_pan):
#   pan_func = iaa.Affine(translate_percent ={"x":(-0.1, 0.1), "y":(-0.1,0.1)})
#   panned_image = pan_func.augment_image(image_to_pan)
#   return panned_image

# def img_bright(image):
#     brightness = iaa.Multiply((0.2,1.2))
#     image = brightness.augment_image(image)
#     return image

# def image_augment(image):
#     image = mpimg.imread(image)
#     if np.random.rand() < 0.5:
#         image = zoom(image)
#     if np.random.rand() < 0.5:
#         image = pan(image)
#     if np.random.rand() < 0.5:
#         image = img_bright(image)
#     if np.random.rand() < 0.5:
#         image = flip_random_image(image)
#     return image

# # Image Augmentation
# figs, axs = plt.subplots(1, 5, figsize=(20, 2))
# figs.tight_layout()
# for i in range(5):
#     rand_num = x_train[random.randint(0, len(x_train)-1)] 
#     original_image = mpimg.imread(rand_num)
#     augmented_image = image_augment(original_image)
#     axs[i][0].set_title("Original_image")
#     axs[i][0].imshow(original_image)
#     axs[i][0].axis('off')
#     axs[i][0].set_title("Augmented_image")
#     axs[i][1].imshow(augmented_image)
#     axs[i][1].axis('off')

#create the model
#based off the lenet model

num_classes = 27
num_samples = []
cols = 5




def modified_model():
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

model = modified_model()
print(model.summary())

history = model.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val), verbose=1, shuffle=1)












