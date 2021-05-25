# loading necessary libraries
import os
import tensorflow as tf
from tensorflow import keras
import tempfile

#_>print(tf.version.VERSION)

# load the dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0

# Save a dataset

path = os.path.join(tempfile.gettempdir(), "saved_data")

dataset = tf.keras.datasets.mnist.load_data()
tf.data.experimental.save(dataset, path)
#_>new_dataset = tf.data.experimental.load(path)

# load the dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0