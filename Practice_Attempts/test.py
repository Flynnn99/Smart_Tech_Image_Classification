import tensorflow as tf
from tensorflow.keras.datasets import cifar10, cifar100
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
import cv2

class_mapping = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}
num_classes = len(class_mapping)

#cifar10_classes = [1, 2, 3, 4, 5, 7]  # classes from CIFAR-10
#cifar100_classes = [11, 34, 35, 36, 46, 98, 2, 31, 33, 42, 72, 78, 95, 97]

selected_classes_cifar10 = [1, 2, 3,]
selected_classes_cifar100 = [4, 5, 6]

#Preprocessing
def greyscale(img):
    img = np.float32(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

# Histogram equalisation aims to standardise the lighting in all the images, enhaces contrast
def equalize(img):
  img = cv2.equalizeHist(img)
  return img

def preprocessing(img):
    greyimg = greyscale(img)#
    histimg = equalize(greyimg)
    normimg = histimg / 255.0
    #image = cv2.equalizeHist(greyscale_img)
    return normimg

# Function to filter dataset based on selected classes
def filter_dataset(images, labels, selected_classes):
    filter_indices = np.isin(labels.flatten(), selected_classes)
    filtered_images = images[filter_indices]
    filtered_labels = labels[filter_indices]
    return filtered_images, filtered_labels

# Function to Combine datasets
def combine_datasets(cifar10_train_filtered, cifar10_labels_train_filtered, cifar100_train_filtered, cifar100_labels_train_filtered, cifar10_test_filtered, cifar10_labels_test_filtered, cifar100_test_filtered, cifar100_labels_test_filtered):
    train_images = np.concatenate((cifar10_train_filtered, cifar100_train_filtered))
    train_labels = np.concatenate((cifar10_labels_train_filtered, cifar100_labels_train_filtered))
    test_images = np.concatenate((cifar10_test_filtered, cifar100_test_filtered))
    test_labels = np.concatenate((cifar10_labels_test_filtered, cifar100_labels_test_filtered))
    return train_images, train_labels, test_images, test_labels


# Normalize pixel values to be between 0 and 1
def normalise(train_images, test_images):
    train_images, test_images = train_images / 255.0, test_images / 255.0
    return train_images, test_images

#Data Model
def alpha_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu')),
    model.add(Flatten()),
    model.add(Dense(64, activation='relu')),
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(Adam(learning_rate=0.01),loss='categorical_crossentropy', metrics=['accuracy'])
    return model

#Plotting the Loss
def plot_loss(model, train_images, train_labels_categorical, test_images, test_labels_categorical):
    history = model.fit(train_images, train_labels_categorical, epochs=10, 
                        validation_data=(test_images, test_labels_categorical))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['loss', 'val_loss'])
    plt.title('loss')
    plt.xlabel('epoch')
    plt.show()
    return history

#Evaluate the Model
def evaulute_model(model, X_test, y_test):
    score = model.evaluate(X_test, y_test, verbose=1)
    print('Test Score', score[0])
    print('Test Accuracy', score[1])

def main():
    #load the data in
    (cifar10_train, cifar10_labels_train), (cifar10_test, cifar10_labels_test) = cifar10.load_data()
    (cifar100_train, cifar100_labels_train), (cifar100_test, cifar100_labels_test) = cifar100.load_data()

    # Define selected classes (Replace these with our chosen classes)
    cifar10_train_filtered, cifar10_labels_train_filtered = filter_dataset(cifar10_train, cifar10_labels_train, selected_classes_cifar10)
    cifar10_test_filtered, cifar10_labels_test_filtered = filter_dataset(cifar10_test, cifar10_labels_test, selected_classes_cifar10)
    cifar100_train_filtered, cifar100_labels_train_filtered = filter_dataset(cifar100_train, cifar100_labels_train, selected_classes_cifar100)
    cifar100_test_filtered, cifar100_labels_test_filtered = filter_dataset(cifar100_test, cifar100_labels_test, selected_classes_cifar100)

    # Combine datasets
    train_images, train_labels, test_images, test_labels = combine_datasets(cifar10_train_filtered, cifar10_labels_train_filtered, cifar100_train_filtered, cifar100_labels_train_filtered, cifar10_test_filtered, cifar10_labels_test_filtered, cifar100_test_filtered, cifar100_labels_test_filtered)

    # Normalize pixel values to be between 0 and 1
    train_images, test_images = normalise(train_images, test_images)


    # Map class labels to a new range
    train_labels_mapped = np.vectorize(class_mapping.get)(train_labels)
    test_labels_mapped = np.vectorize(class_mapping.get)(test_labels)

    #print(set(train_labels_mapped))

    
    plt.imshow(train_images[1])
    plt.xlabel(train_labels_mapped[1])
    plt.show()
    print(train_images.shape)

    
    img = preprocessing(train_images[1])
    plt.imshow(img)
    plt.xlabel(train_labels_mapped[1])
    plt.show()

    # Apply one-hot encoding
    #num_classes = len(class_mapping)
    train_labels_categorical = to_categorical(train_labels_mapped, num_classes)
    test_labels_categorical = to_categorical(test_labels_mapped, num_classes)

     # Compile the model
    model = alpha_model()
    model.summary()
   
    # Train the Model
    history = plot_loss(model, train_images, train_labels_categorical, test_images, test_labels_categorical)
    
    # Evaluate the Model
    evaulute_model(model, test_images, test_labels_categorical)

    # Get predictions
    predictions = model.predict(test_images)
    predicted_labels = np.argmax(predictions, axis=1)

    # Print some examples
    print('Examples:')

    for i in range(10):
        print(f'Predicted: {predicted_labels[i]}, Actual: {test_labels_mapped[i]}')

    # Plot training history
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.show()

    
   

if __name__ == "__main__":
    main()
    