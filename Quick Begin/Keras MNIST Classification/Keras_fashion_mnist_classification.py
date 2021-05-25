# loading libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt 
print(tf.__version__)

# loading dtaset
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print(train_images.shape)
print(len(train_labels))

# visualize data example 
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

# preprocessing dataset
train_images = train_images / 255.0
test_images = test_images / 255.0

# visualize dataset again
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

# creating the model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
    ])

# compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# train the model
model.fit(train_images, train_labels, epochs=10)

# evalute accuracy
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print("Test accuracy: ", test_acc)

# make prediction
probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])

predictions = probability_model.predict(test_images)
print(predictions[0]) ### here you can see 10 array number which holds confidence it means tht which array has higher confidence it represend class number
print(np.argmax(predictions[0])) ### here it represend 9 which is the class number

# make single prediction
img = test_images[1]
img = (np.expand_dims(img,0))

pred_single = probability_model.predict(img)
print(pred_single)
print(np.argmax(pred_single[0]))