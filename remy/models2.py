import tensorflow as tf
import numpy as np
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# Set seeds
tf.random.set_seed(94)
np.random.seed(94)

# Load data
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

y_train = tf.squeeze(tf.one_hot(y_train, 10))
y_test = tf.squeeze(tf.one_hot(y_test, 10))


# Normalize pixel values to be between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0


# Define model
model = models.Sequential(name='cifar')
model.add(layers.Conv2D(32, (4, 4), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (4, 4), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (4, 4), activation='relu'))
model.add(layers.AveragePooling2D())
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))


model.summary()

loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)


# Compile and fit
model.compile(optimizer='adam',
              loss=loss,
              metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=15,
                    validation_data=(x_test, y_test))

# Plot accuracy
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(x_test, y_test)
