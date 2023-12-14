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
from sklearn.preprocessing import LabelEncoder



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
plt.show()



print(set(y_train))
x_train = np.array(list(map(preprocessing, x_train)))
x_test = np.array(list(map(preprocessing, x_test)))

# After preprocessing, add a channel dimension to x_train and x_test
x_train = x_train.reshape(x_train.shape[0], 32, 32, 1)
x_test = x_test.reshape(x_test.shape[0], 32, 32, 1)

# Step 1: Convert string labels to integer labels
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)


unique_classes = np.unique(y_train)
num_classes = len(unique_classes)
# Step 2: Apply to_categorical
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)
# Assuming y_train is a numpy array of labels

print(x_train.shape), print(y_train.shape), print(x_test.shape), print(y_test.shape)

#First Iteration of Our Model Based of the one we used in Class 
def alpha_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(32, 32, 1), activation='relu'))  # Adjust input_shape here
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(500, activation ='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))  # Use num_classes here
    model.compile(Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = alpha_model()
history = model.fit(x_train, y_train, epochs=1, validation_data=(x_test, y_test), 
                    batch_size=100, verbose=1, shuffle=1)
model.summary()

# Plotting our loss charts
plt.figure(0)
plt.plot(history.history['loss'], 'r')
plt.plot(history.history['val_loss'], 'g')
plt.xticks(np.arange(0, 10, 1.0))
plt.rcParams['figure.figsize'] = (8, 6)
plt.xlabel("Num of Epochs")
plt.ylabel("Loss")
plt.title("Training Loss vs Validation Loss")
plt.legend(['train', 'validation'])
plt.show()  

# Plotting our accuracy charts
plt.figure(1)
plt.plot(history.history['accuracy'], 'r')
plt.plot(history.history['val_accuracy'], 'g')
plt.xticks(np.arange(0, 10, 1.0))
plt.rcParams['figure.figsize'] = (8, 6)
plt.xlabel("Num of Epochs")
plt.ylabel("Accuracy")
plt.title("Training Accuracy vs Validation Accuracy")
plt.legend(['train', 'validation'])
plt.show()

# Predictions
predictions = model.predict(x_test)
predicted_labels = np.argmax(predictions, axis=1)

# Initialize the dictionary for counting classes
# Convert labels in label_encoder.classes_ to integers if they are not already
class_indices = range(len(label_encoder.classes_))  # Assuming label_encoder.classes_ are in order
class_counts = {label_index: 0 for label_index in class_indices}

# Count the number of images in each class
for label_array in y_test:
    label = np.argmax(label_array)  # Extracting the label index
    class_counts[label] += 1

# Convert counts to a list for plotting
labels = label_encoder.classes_  # Class names
counts = [class_counts[index] for index in class_indices]

# Plotting
plt.figure(figsize=(20, 10))
plt.bar(labels, counts)
plt.xlabel('Class Labels')
plt.ylabel('Number of Images')
plt.title('Distribution of Classes in the CIFAR Dataset')
plt.xticks(labels, rotation=90)
plt.show()

# Show images from each class
# Convert one-hot encoded y_test to class indices
class_indices = np.argmax(y_test, axis=1)

plt.figure(figsize=(20, 10))
for i in range(num_classes):
    # Select images of class i
    x_selected = x_test[class_indices == i]

    # In case there are no images of a certain class, continue to the next
    if len(x_selected) == 0:
        continue

    img = x_selected[0]  # Select the first image of class i
    plt.subplot(1, num_classes, i + 1)
    plt.imshow(img.squeeze(), cmap=plt.get_cmap('gray'))
    plt.axis('off')
    plt.title("Label: {}".format(label_encoder.classes_[i]))
plt.show()

