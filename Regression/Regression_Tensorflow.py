# import librabries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
#print(tf.__version__)

# download the dataset
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Origin']

raw_ds = pd.read_csv(url, names=column_names,
                     na_values='?', comment='\t',
                     sep=' ', skipinitialspace=True)

dataset = raw_ds.copy()
#print(dataset.tail())

# prepare dataset
#print(dataset.isna().sum()) # to check zero values in dataset

dataset = dataset.dropna() # to drop zero values

dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'}) # to change categorical data 
dataset = pd.get_dummies(dataset, columns=['Origin'], prefix='', prefix_sep='') # to change categorical data
#print(dataset.tail())

# split dataset for training and testing
train_ds = dataset.sample(frac=0.8, random_state=0)
test_ds = dataset.drop(train_ds.index)

# split features from labels
train_features = train_ds.copy()
test_features = test_ds.copy()

train_labels = train_features.pop('MPG')
test_labels = test_features.pop('MPG')

# normalize dataset
normalizer = preprocessing.Normalization()
normalizer.adapt(np.array(train_features))

# to check normalized data
first = np.array(train_features[:1])
with np.printoptions(precision=2, suppress=True):
    print("First exampe: ", first)
    print("<><><><><><><><>")
    print("First example after normalized: ", normalizer(first).numpy())



#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

# THE LINEAR MODEL


# LINEAR REGRESSION WITH ONE INDEPENDENT INPUT VARIABLE

horsepower = np.array(train_features['Horsepower'])
horsepower_normalized = preprocessing.Normalization(input_shape=[1,])
horsepower_normalized.adapt(horsepower)

# build the model
horsepower_model = tf.keras.Sequential([
    horsepower_normalized,
    layers.Dense(units=1)
    ])

horsepower_model.summary()

# compile the model
horsepower_model.compile(
    optimizer = tf.optimizers.Adam(learning_rate=0.1),
    loss = 'mean_absolute_error')


history = horsepower_model.fit(
    train_features['Horsepower'], train_labels,
    epochs=50,
    verbose=0,
    validation_split=0.2)

# to visualize the model performance
def plot_his(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0,10])
    plt.xlabel('Epoch')
    plt.ylabel('Error [MPG]')
    plt.legend()
    plt.grid(True)


plot_his(history)

# LINEAR REGRESSION WITH MULTIPLE INDEPENDENT INPUT VARIABLE

# create the model
linear_model = tf.keras.Sequential([
    normalizer,
    layers.Dense(units=1)
])

# compile the model
linear_model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error')

# train the model
history = linear_model.fit(
    train_features, train_labels, 
    epochs=100,
    verbose=0,
    validation_split = 0.2)

# plot the history
plot_loss(history)

#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

# A DNN REGRESSION

# SINGLE VARIABLE

# build the model function
def build_and_compile_model(norm):
  model = keras.Sequential([
      norm,
      layers.Dense(64, activation='relu'),
      layers.Dense(64, activation='relu'),
      layers.Dense(1)
  ])

  model.compile(loss='mean_absolute_error',
                optimizer=tf.keras.optimizers.Adam(0.001))
  return model

dnn_horsepower_model = build_and_compile_model(horsepower_normalizer)
dnn_horsepower_model.summary()

# train the model
history = dnn_horsepower_model.fit(
    train_features['Horsepower'], train_labels,
    validation_split=0.2,
    verbose=0, epochs=100)


# MULTIPLE VARIABLE
dnn_model = build_and_compile_model(normalizer)
dnn_model.summary()

history = dnn_model.fit(
    train_features, train_labels,
    validation_split=0.2,
    verbose=0, epochs=100)

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#$$$ THIS IS REGRESSION MODEL
#$$$ LINEAR MODEL:  1)SINGLE VARIABLE    2)MULTIPLE VARIABLE
#$$$ DNN MODEL:     1)SINGLE VARIABLE    2)MULTIPLE VARIABLE
 