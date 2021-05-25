#sonofsunra

# importing libraries
import tensorflow as tf

# loading sataset
mnist = tf.keras.datasets.mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train, X_test = X_train / 255., X_test / 255.0

# create model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
    ])

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# compile model
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

# train the model
model.fit(X_train, y_train, epochs=5)

# get the report
model.evaluate(X_test, y_test, verbose=2)

# Great work.
# Our model trained with 97.68% accurcy.