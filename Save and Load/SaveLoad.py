# pip install -q pyyaml h5py  
# Required to save models in HDF5 format

# loading necessary libraries
import os
import tensorflow as tf
from tensorflow import keras
print(tf.version.VERSION)



# crete function to return entire datset
def get_dataset(x_train, y_train, x_test, y_test):
    x_train = train_images
    y_train = train_labels

    x_test = test_images
    y_test = test_labels

    return (x_train, y_train), (x_test, y_test)


# Define a simple sequential model
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

# Create a basic model instance
#_>model = create_model()

# Display the model's architecture
#_>model.summary()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#now we come to fun part.
#First of all we created the model
#Our goal is train the model with train dataset, and for test evaluation we create another new model as same old one but without training
#for new model we load weights from old trained model
#here we have several ways to do this challenge, we will see one by one

# 1) Checkpoint callback options  

# Save Checkpoints during training
checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

# Create a checkpoint model instance
#_>checkpoint_model = create_model()

# Train the model with the new callback
"""
checkpoint_model.fit(train_images, 
          train_labels,  
          epochs=10,
          validation_data=(test_images, test_labels),
          callbacks=[cp_callback])  # Pass callback to training
"""

# This may generate warnings related to saving the state of the optimizer.
# These warnings (and similar warnings throughout this notebook)
# are in place to discourage outdated usage, and can be ignored.



#_>print(os.listdir(checkpoint_dir)) # to check checkpoint directory

# Create a new model instance
#_>stupid_model = create_model()

# Evaluate the model
#_>loss, acc = stupid_model.evaluate(test_images, test_labels, verbose=2)
#_>print("Untrained model, accuracy: {:5.2f}%".format(100 * acc)) # as we seen it has only 6.5% accuracy, bacause of untrained model

# Loads the weights
#stupid_model.load_weights(checkpoint_path)

# Re-evaluate the model
#_>loss, acc = stupid_model.evaluate(test_images, test_labels, verbose=2)
#_>print("Restored model, accuracy: {:5.2f}%".format(100 * acc)) # so far so good when we load weights from our first trained model, our second model is reaching 86% of accuracy


# 2) Manually save weights

#_>manual_model = create_model()

'''
manual_model.fit(train_images, 
          train_labels,  
          epochs=10,
          validation_data=(test_images, test_labels))


'''

# Save the weights
#_>manual_model.save_weights('./checkpoints/my_checkpoint')
