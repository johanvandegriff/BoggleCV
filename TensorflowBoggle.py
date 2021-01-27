#https://www.tensorflow.org/tutorials/keras/classification

from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

import json

print(tf.__version__)

DATA_FILE="/home/johanv/downloads/labelled.json"
#DATA_FILE="/home/johanv/Nextcloud/Projects/Boggle2.0/labelled.json"
#DATA_FILE="/home/johanv/Nextcloud/Projects/Boggle2.0/labelled-20200124_155523.mp4.json"

#fashion_mnist = keras.datasets.fashion_mnist
#(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
#class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
#               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

with open(DATA_FILE, 'r') as f:
    data = json.load(f)

IMG_DIM = 30

images_in = data["imgs"]
labels_in = data["labels"]

images = []
labels = []
for rot in range(4):
    for i in range(len(images_in)):
    #https://artemrudenko.wordpress.com/2014/08/28/python-rotate-2d-arraymatrix-90-degrees-one-liner/
        images_in[i] = list(zip(*images_in[i][::-1])) #rotate 90 degrees (still the same letter!)
    images.extend(images_in)
    labels.extend(labels_in)


images = np.array(images, dtype=np.uint8).reshape((-1, IMG_DIM, IMG_DIM, 1))

split = int(0.15 * len(images))

train_images = np.array(images[split:],dtype=np.uint8)
train_labels = np.array(labels[split:],dtype=np.uint8)

test_images = np.array(images[:split],dtype=np.uint8)
test_labels = np.array(labels[:split],dtype=np.uint8)

class_names = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

L = len(class_names)

print(train_images.shape)
print(train_labels.shape)
print(test_images.shape)
print(test_labels.shape)

# plt.figure()
# plt.imshow(train_images[1])
# plt.colorbar()
# plt.grid(False)
# plt.show()

train_images = train_images / 255.0
test_images = test_images / 255.0

plt.figure(figsize=(10,10))
for i in range(25):
    print("{}/25".format(i))
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i].reshape((IMG_DIM, IMG_DIM)), cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

#model = keras.Sequential([
#    keras.layers.Flatten(input_shape=(IMG_DIM,IMG_DIM)),
#    keras.layers.Dense(128, activation="sigmoid"),
#    keras.layers.Dense(64, activation="sigmoid"),
#    keras.layers.Dense(L, activation="softmax")
#])

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_DIM, IMG_DIM, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(L, activation="softmax"))

#model.compile(
#    loss='sparse_categorical_crossentropy',
#    optimizer='adam',
#    metrics=['accuracy']
#)
#
#model.fit(train_images, train_labels, epochs=7)

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10,
                    validation_data=(test_images, test_labels))

print(history)

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

print("\n\nacc: ", test_acc)

predictions = model.predict(test_images)
print("predictions[0]: ", predictions[0])
print("np.argmax(predictions[0]): ", np.argmax(predictions[0]))
print("test_labels[0]: ", test_labels[0])

def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array, true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img.reshape((IMG_DIM, IMG_DIM)), cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array, true_label[i]
  plt.grid(False)
  plt.xticks(range(L))
  plt.yticks([])
  thisplot = plt.bar(range(L), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()

i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()

# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions[i], test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()

# Grab an image from the test dataset.
img = test_images[1]

print(img.shape)

# Add the image to a batch where it's the only member.
img = (np.expand_dims(img,0))

print(img.shape)

predictions_single = model.predict(img)

print(predictions_single)

plot_value_array(1, predictions_single[0], test_labels)
_ = plt.xticks(range(L), class_names, rotation=45)

print("np.argmax(predictions_single[0]): ", np.argmax(predictions_single[0]))



