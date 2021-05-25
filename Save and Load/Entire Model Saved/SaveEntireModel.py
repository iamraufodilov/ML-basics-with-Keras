# loading necessary libraries
import tensorflow as tf
from tensorflow import keras



# load the dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0


def create_model():
  model = tf.keras.models.Sequential([
    keras.layers.Dense(512, activation='relu', input_shape=(784,)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10)
  ])

  model.compile(optimizer='adam',
                loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[tf.metrics.SparseCategoricalAccuracy()])

  return model

# Create and train a new model instance.
#_>model = create_model()
#_>model.fit(train_images, train_labels, epochs=5)

# Save the entire model as a SavedModel.
#_>model.save('G:/rauf/STEPBYSTEP/Tutorial/TensorFlowCore/Save and Load/Entire Model Saved/saved_model/my_model')


# load entire model
#_>new_model = tf.keras.models.load_model('G:/rauf/STEPBYSTEP/Tutorial/TensorFlowCore/Save and Load/Entire Model Saved/saved_model/my_model')

# Check its architecture
#_>new_model.summary()

# Evaluate the restored model
#_>loss, acc = new_model.evaluate(test_images, test_labels, verbose=2)
#_>print('Restored model, accuracy: {:5.2f}%'.format(100 * acc)) # here we go we have 86% accuracy with loaded model



# HDF5 format

# Create and train a new model instance.
#_>model = create_model()
#_>model.fit(train_images, train_labels, epochs=5)

# Save the entire model to a HDF5 file.
#_>model.save('G:/rauf/STEPBYSTEP/Tutorial/TensorFlowCore/Save and Load/Entire Model Saved/saved_model/my_model.h5')


# Recreate the exact same model, including its weights and the optimizer
new_model = tf.keras.models.load_model('G:/rauf/STEPBYSTEP/Tutorial/TensorFlowCore/Save and Load/Entire Model Saved/saved_model/my_model.h5')

# Show the model architecture
#_>new_model.summary()

loss, acc = new_model.evaluate(test_images, test_labels, verbose=2)
print('Restored model, accuracy: {:5.2f}%'.format(100 * acc)) # gain we got 86% accuracy, but this time we lod model through h5 file