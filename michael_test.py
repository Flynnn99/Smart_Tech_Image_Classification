import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
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
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split



np.random.seed(0)
num_of_samples = []
cols = 5
num_classes = 109
num_pixels = 3072


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def extract_relevant_data(unpickled_data, relevant_classes):
    images = unpickled_data[b'data']
    labels = unpickled_data[b'labels'] if b'labels' in unpickled_data else unpickled_data[b'fine_labels']
    relevant_images = []
    relevant_labels = []
    for i, label in enumerate(labels):
        if label in relevant_classes:
            relevant_images.append(images[i])
            relevant_labels.append(label)
    return np.array(relevant_images), np.array(relevant_labels)


def test_model():
    model = Sequential()
    model.add(Conv2D(30, (5,5), input_shape=(32,32,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(15, (3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(500,  activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(Adam(learning_rate=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def preprocess_data(images, labels, num_classes):
    # Normalize pixel values to 0-1
    images = images.astype('float32') / 255.0

    # Reshape data for CNN (32x32 pixels with 3 channels)
    # The correct order is (number of images, height, width, channels)
    images = images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

    # One-hot encode labels
    labels = to_categorical(labels, num_classes)

    return images, labels

def show_images(images, labels, class_names, num_images=5):
    fig, axes = plt.subplots(1, num_images, figsize=(15, 3))
    for i, ax in enumerate(axes):
        ax.imshow(images[i])
        ax.set_title(class_names[labels[i]])
        ax.axis('off')
    plt.show()

def plot_class_representations(images, labels, class_names, samples_per_class=5):
    num_classes = len(class_names)
    fig, axes = plt.subplots(nrows=num_classes, ncols=samples_per_class, figsize=(15, 2 * num_classes))
    fig.tight_layout()
    for class_idx in range(num_classes): 
        idxs = np.where(labels == class_idx)[0][:samples_per_class] 
        for plot_idx, img_idx in enumerate(idxs): #img
            ax = axes[class_idx, plot_idx] if samples_per_class > 1 else axes[plot_idx] 
            ax.imshow(images[img_idx], cmap=plt.get_cmap('gray')) 
            ax.axis('off')
           # ax.set_title(f'{class_idx}-{class_names[class_idx]}')        
    plt.show()

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



def main():
    
    cifar101_data = unpickle('cifar_files/cifar-10-batches-py/data_batch_1')
    cifar102_data = unpickle('cifar_files/cifar-10-batches-py/data_batch_2')
    cifar103_data = unpickle('cifar_files/cifar-10-batches-py/data_batch_3')
    cifar104_data = unpickle('cifar_files/cifar-10-batches-py/data_batch_4')
    cifar100_data = unpickle('cifar_files/cifar-100-python/train')



    # Load label names from metadata files
    cifar10_meta = unpickle('cifar_files/cifar-10-batches-py/batches.meta')
    cifar100_meta = unpickle('cifar_files/cifar-100-python/meta')

    # Extract label names
    cifar10_label_names = [t.decode('utf8') for t in cifar10_meta[b'label_names']]
    cifar100_label_names = [t.decode('utf8') for t in cifar100_meta[b'fine_label_names']]

    print("---------------------------------------------")
    print("CIFAR-10 label names:", cifar10_label_names)
    print("---------------------------------------------")
    print("CIFAR-100 label names:", cifar100_label_names)
    print("---------------------------------------------")

     # Combined class mapping
    combined_class_mapping = {
        # CIFAR-10 classes
        'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4,
        'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9,
        # CIFAR-100 classes, starting from 10
        'cattle': 10, 'fox': 11, 'baby': 12, 'boy': 13, 'girl': 14,
        'man': 15, 'woman': 16, 'rabbit': 17, 'squirrel': 18, 'bicycle': 19,
        'bus': 20, 'motorcycle': 21, 'pickup_truck': 22, 'train': 23,
        'cockroach': 24, 'couch': 25

      #  'cattle': 19, 'fox': 34, 'baby': 2, 'boy': 11, 'girl': 35,
      #   'man': 47, 'woman': 99, 'rabbit': 65, 'squirrel': 80, 'bicycle': 8,
      #   'bus': 13, 'motorcycle': 49, 'pickup_truck': 59, 'train': 90, 'lawn_mower': 41,
      #    'tractor': 96,  
    }
    
    
     # Convert label names to label lists
    cifar10_relevant_labels = [combined_class_mapping[name] for name in cifar10_label_names if name in combined_class_mapping]
    cifar100_relevant_labels = [combined_class_mapping[name] for name in cifar100_label_names if name in combined_class_mapping]

    # Extract relevant data
    cifar101_images, cifar101_labels = extract_relevant_data(cifar101_data, cifar10_relevant_labels)
    cifar102_images, cifar102_labels = extract_relevant_data(cifar102_data, cifar10_relevant_labels)
    cifar103_images, cifar103_labels = extract_relevant_data(cifar103_data, cifar10_relevant_labels)
    cifar104_images, cifar104_labels = extract_relevant_data(cifar104_data, cifar10_relevant_labels)

    cifar100_images, cifar100_labels = extract_relevant_data(cifar100_data, cifar100_relevant_labels)

    # Preprocess data
    cifar101_images, cifar101_labels = preprocess_data(cifar101_images, cifar101_labels, len(combined_class_mapping))
    cifar102_images, cifar102_labels = preprocess_data(cifar102_images, cifar102_labels, len(combined_class_mapping))
    cifar103_images, cifar103_labels = preprocess_data(cifar103_images, cifar103_labels, len(combined_class_mapping))
    cifar104_images, cifar104_labels = preprocess_data(cifar104_images, cifar104_labels, len(combined_class_mapping))
    
    cifar100_images, cifar100_labels = preprocess_data(cifar100_images, cifar100_labels, len(combined_class_mapping))

    # Combine datasets
    combined_images = np.concatenate([cifar101_images, cifar102_images, cifar103_images, cifar104_images, cifar100_images])
    combined_labels = np.concatenate([cifar101_labels, cifar102_labels, cifar103_labels, cifar104_labels, cifar100_labels])

     # Data Exploration
    print("Size of each image: 32x32 pixels")
    
    # Count the number of images in each class before preprocessing
    cifar10_label_counts = {label: cifar101_data[b'labels'].count(label) for label in cifar10_relevant_labels}
    cifar100_label_counts = {label: cifar100_data[b'fine_labels'].count(label) for label in cifar100_relevant_labels}

    print("Number of images per class in CIFAR-10 before preprocessing:", cifar10_label_counts)
    print("Number of images per class in CIFAR-100 before preprocessing:", cifar100_label_counts)

    # Count the number of images in each class after preprocessing
    post_processed_label_counts = np.sum(combined_labels, axis=0)
    print("Number of images per class after preprocessing:", post_processed_label_counts)

    # Extract class names for visualization
    class_names = [name for name, index in sorted(combined_class_mapping.items(), key=lambda item: item[1])]

    # Show a few images from each class
    show_images(combined_images, np.argmax(combined_labels, axis=1), class_names)
    plot_class_representations(combined_images, np.argmax(combined_labels, axis=1), class_names)

    class_numbers = np.argmax(combined_labels, axis=1)
    plot_distribution(class_numbers)
    
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(combined_images, combined_labels, test_size=0.2, random_state=0)

    # Split the training set into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

    # Check the shape of the training, validation, and test sets
    print("Training set shape:", X_train.shape, y_train.shape)
    print("Validation set shape:", X_val.shape, y_val.shape)
    print("Test set shape:", X_test.shape, y_test.shape)

   
    # Create the model
    # model = test_model()
    # print(model.summary())

    # history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=500, verbose=1, shuffle=1)
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.legend(['training', 'validation'])
    # plt.title("loss")
    # plt.xlabel("plot")

if __name__ == '__main__':
    main()