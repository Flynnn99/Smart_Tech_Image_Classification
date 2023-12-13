import tensorflow as tf
from tensorflow.keras.datasets import cifar10, cifar100
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.utils import to_categorical
import numpy as np

# Load CIFAR-10 and CIFAR-100 data
(cifar10_train, cifar10_labels_train), (cifar10_test, cifar10_labels_test) = cifar10.load_data()
(cifar100_train, cifar100_labels_train), (cifar100_test, cifar100_labels_test) = cifar100.load_data()

# Define selected classes (Replace these with our chosen classes)
selected_classes_cifar10 = [1, 2, 3]
selected_classes_cifar100 = [4, 5, 6]

# Function to filter dataset based on selected classes
def filter_dataset(images, labels, selected_classes):
    filter_indices = np.isin(labels.flatten(), selected_classes)
    filtered_images = images[filter_indices]
    filtered_labels = labels[filter_indices]
    return filtered_images, filtered_labels

# Filter datasets
cifar10_train_filtered, cifar10_labels_train_filtered = filter_dataset(cifar10_train, cifar10_labels_train, selected_classes_cifar10)
cifar10_test_filtered, cifar10_labels_test_filtered = filter_dataset(cifar10_test, cifar10_labels_test, selected_classes_cifar10)
cifar100_train_filtered, cifar100_labels_train_filtered = filter_dataset(cifar100_train, cifar100_labels_train, selected_classes_cifar100)
cifar100_test_filtered, cifar100_labels_test_filtered = filter_dataset(cifar100_test, cifar100_labels_test, selected_classes_cifar100)

# Combine datasets
train_images = np.concatenate((cifar10_train_filtered, cifar100_train_filtered))
train_labels = np.concatenate((cifar10_labels_train_filtered, cifar100_labels_train_filtered))
test_images = np.concatenate((cifar10_test_filtered, cifar100_test_filtered))
test_labels = np.concatenate((cifar10_labels_test_filtered, cifar100_labels_test_filtered))

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# Map class labels to a new range
class_mapping = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}
train_labels_mapped = np.vectorize(class_mapping.get)(train_labels)
test_labels_mapped = np.vectorize(class_mapping.get)(test_labels)

print(set(train_labels_mapped))

# Display Image
import matplotlib.pyplot as plt
plt.imshow(train_images[0])
plt.xlabel(train_labels_mapped[0])
plt.show()

# Apply one-hot encoding
num_classes = len(class_mapping)
train_labels_categorical = to_categorical(train_labels_mapped, num_classes)
test_labels_categorical = to_categorical(test_labels_mapped, num_classes)

# Build the CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train the Model
history = model.fit(train_images, train_labels_categorical, epochs=10, validation_data=(test_images, test_labels_categorical))

# Evaluate the Model
test_loss, test_acc = model.evaluate(test_images,  test_labels_categorical, verbose=2)
print('\nTest accuracy:', test_acc)

# Get predictions
predictions = model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)

# Print some examples
print('Examples:')
for i in range(10):
    print(f'Predicted: {predicted_labels[i]}, Actual: {test_labels_mapped[i]}')

# Plot training history
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

