# import libraries
import warnings
warnings.filterwarnings('ignore')

import numpy as np
np.random.seed(123)

from keras.models import Sequential
from keras.layers import Flatten, MaxPool2D, Conv2D, Dense, Reshape, Dropout
from keras.utils import np_utils
from PIL import Image
from numpy import asarray

from keras.datasets import mnist

# load data and prepare data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /=255

Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

# create model
model = Sequential()
model.add(Conv2D(32, 3, 3, activation='relu', input_shape=(28,28,1)))
model.add(Conv2D(32, 2, 2, activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

# compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# train the model
model.fit(X_train, Y_train, batch_size=32, epochs=10, verbose=1)


# actually i tired to predict ^-^
# so predict part unfinished
data_path = 'G:/rauf/STEPBYSTEP/Data/numbers/test/2.jpg'

custom_data = Image.open(data_path)
data = asarray(custom_data)
data = data.reshape(28, 28, 1)
data = data.astype('float32')
data /= 255

model.predict(data)

