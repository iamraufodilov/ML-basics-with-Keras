# loading necessary libraries
import tensorflow as tf
from tensorflow import keras

from SaveLoad import create_model

# load the dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0


# 2) Manually save weights

#_>manual_model = create_model()

'''

manual_model.fit(train_images, 
          train_labels,  
          epochs=10,
          validation_data=(test_images, test_labels))

'''

# Save the weights
#_>manual_model.save_weights('G:/rauf/STEPBYSTEP/Tutorial/TensorFlowCore/Save and Load/Saved Weights/manually_checkpoint')

# Create a new model instance
new_model = create_model()

# Restore the weights
new_model.load_weights('G:/rauf/STEPBYSTEP/Tutorial/TensorFlowCore/Save and Load/Saved Weights/manually_checkpoint')

# Evaluate the model
loss, acc = new_model.evaluate(test_images, test_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc)) # here we go we got 86% of ccuracy with untrained model, because of saved weights

# Nice work