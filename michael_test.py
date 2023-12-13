from tensorflow.keras.datasets import cifar10, cifar100
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
import pickle
from sklearn.model_selection import train_test_split

#unpickle
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

classes_for_cifar_10 = ['automobile', 'bird', 'cat', 'deer', 'dog','horse','truck']
classes_for_cifar_100 = ['cattle', 'fox', 'baby', 'boy', 'girl', 'man', 'woman', 'rabbit', 'squirrel',
                         'maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'
                        'bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train', 'lawn_mower' ,'tractor']

def get_class_names(x_test, y_test, x_train, y_train, classes):
    selected_classes_for_testing = np.isin(y_test, classes).flatten()
    selected_classes_for_training = np.isin(y_train, classes).flatten()

    x_test = x_test[selected_classes_for_testing]
    y_test = y_test[selected_classes_for_testing]
    x_train = x_train[selected_classes_for_training]
    y_train = y_train[selected_classes_for_training]

    return x_test, y_test, x_train, y_train


def combine(x_train,y_train, x_test, y_test, x_train_2, y_train_2, x_test_2, y_test_2):
    x_train = np.concatenate((x_train, x_train_2))
    y_train = np.concatenate((y_train, y_train_2))
    x_test = np.concatenate((x_test, x_test_2))
    y_test = np.concatenate((y_test, y_test_2))
    return (x_train, y_train), (x_test, y_test)

def load_cifar_10():
    cifar_10_1_data = unpickle('cifar_files/cifar-10-batches-py/data_batch_1')  
    cifar_10_2_data = unpickle('cifar_files/cifar-10-batches-py/data_batch_2')
    cifar_10_3_data = unpickle('cifar_files/cifar-10-batches-py/data_batch_3')
    cifar_10_4_data = unpickle('cifar_files/cifar-10-batches-py/data_batch_4')
    cifar_10_5_data = unpickle('cifar_files/cifar-10-batches-py/data_batch_5')

    cifar_10_meta = unpickle('cifar_files/cifar-10-batches-py/batches.meta')
    labels = cifar_10_meta[b'label_names']

    X_train = np.concatenate((cifar_10_1_data[b'data'], cifar_10_2_data[b'data'], cifar_10_3_data[b'data'], cifar_10_4_data[b'data'], cifar_10_5_data[b'data']), axis=0)
    y_train = np.concatenate((cifar_10_1_data[b'labels'], cifar_10_2_data[b'labels'], cifar_10_3_data[b'labels'], cifar_10_4_data[b'labels'], cifar_10_5_data[b'labels']), axis=0)
    x_test = unpickle('cifar_files/cifar-10-batches-py/test_batch')

    y_train_desc = [labels[i].decode('utf-8') for i in y_train]
    y_test_desc = [labels[i].decode('utf-8') for i in x_test[b'labels']]

    #return X_train, y_train_desc, x_test, y_test_desc
    return (X_train, np.asarray(y_train_desc)), (x_test[b'data'], np.asarray(y_test_desc))


def load_cifar_100():
    meta = unpickle('cifar_files/cifar-100-python/meta')
    train = unpickle('cifar_files/cifar-100-python/train')
    x_test = unpickle('cifar_files/cifar-100-python/test')

    x_train = train[b'data']
    y_train = train[b'fine_labels']

    label_names = meta[b'fine_label_names']

    y_train_desc = [label_names[label_index].decode('utf-8') for label_index in y_train]
    y_test_desc = [label_names[label_index].decode('utf-8') for label_index in x_test[b'fine_labels']]
    return (x_train, np.asarray(y_train_desc)), (x_test[b'data'], np.asarray(y_test_desc))
    #return x_train, y_train_desc, x_test, y_test_desc



(x_train2, y_train2), (x_test2, y_test2) = load_cifar_100()

(x_train1, y_train1), (x_test1, y_test1) = load_cifar_10()

x_train1, y_train1, x_test1, y_test1 = get_class_names(x_train1, y_train1, x_test1, y_test1, classes_for_cifar_10)
x_train2, y_train2, x_test2, y_test2 = get_class_names(x_train2, y_train2, x_test2, y_test2, classes_for_cifar_100)

(x_train, y_train), (x_test, y_test) = combine(x_train1, y_train1, x_test1, y_test1, x_train2, y_train2, x_test2, y_test2)

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)


# Visualise the Classes
num_of_samples = []
cols = 5
num_classes = np.unique(y_train)

#split the data
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

print(x_train.shape)
print(x_val.shape)
print(x_test.shape)

x_train = x_train.reshape(x_train.shape[0],32,32,3)
x_test = x_test.reshape(x_test.shape[0],32,32,3)

assert(x_train.shape[0] == y_train.shape[0]), "the number training images is different from the number of labels"
assert(x_val.shape[0] == y_val.shape[0]), "the number training images is different from the number of labels"
assert(x_test.shape[0] == y_test.shape[0]), "the number training images is different from the number of labels"
assert(x_train.shape[1:] == (32,32,3)), "Training image is not 32,32,3"


print(x_train.shape)
print(x_val.shape)
print(x_test.shape)


number_of_samples = []
cols = 5
num_classes = len(classes_for_cifar_10 + classes_for_cifar_100)

plt.imshow(x_train[20])
plt.xlabel(y_train[20])
print(x_train[20].shape)
print(y_train[20])
plt.show()



