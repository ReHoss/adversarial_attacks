import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers

(x_train, y_train), (
        x_test, y_test) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i])
    # The CIFAR labels happen to be arrays,
    # which is why you need the extra index
    plt.xlabel(class_names[y_train[i][0]])
plt.show()


class ConvBlock(layers.Layer):

    def __init__(self):
        super(ConvBlock, self).__init__()
        self.conv = layers.Conv2D(32, 3, padding='same')
        self.relu = layers.ReLU()
        self.max_pool = layers.MaxPool2D()

    # noinspection PyMethodOverriding
    def call(self, inputs):
        x = self.conv(inputs)
        x = self.relu(x)
        x = self.max_pool(x)
        return self.max_pool(x)



class EndBlock(layers.Layer):

    def __init__(self):
        super(EndBlock, self).__init__()
        self.avg_pool = layers.AveragePooling2D()
        self.conv = layers.Conv2D(32, 3)
        self.relu1 = layers.ReLU()
        self.relu2 = layers.ReLU()
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(128)
        self.dense2 = layers.Dense(10)

    # noinspection PyMethodOverriding
    def call(self, inputs):
        x = self.conv(inputs)
        x = self.relu1(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.relu2(x)
        x = self.dense2(x)
        return x


class MyNet(tf.keras.Model):

    def __init__(self):
        super(MyNet, self).__init__()
        self.conv_block_1 = ConvBlock()
        self.conv_block_2 = ConvBlock()
        self.end_block = EndBlock()model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

    # noinspection PyMethodOverriding
    def call(self, inputs):
        x = self.conv_block_1(inputs)
        x = self.conv_block_2(x)
        x = self.end_block(x)
        return x


#
test = ConvBlock()
img = next(iter(train_dataset))
layers.Dense(10)(test(img[0]))

with tf.GradientTape() as tape:
    # Run the forward pass of the layer.
    # The operations that the layer applies
    # to its inputs are going to be recorded
    # on the GradientTape.
    logits = layers.Dense(10)(layers.Flatten()(test(img[0])))  # Logits for this minibatch

    # Compute the loss value for this minibatch.
    loss_value = loss_fn(img[1], logits)

# Use the gradient tape to automatically retrieve
# the gradients of the trainable variables with respect to the loss.
grads = tape.gradient(loss_value, model.trainable_weights)




#
# model(next(iter(train_dataset))).shape

if __name__ == '__main__':

    model = MyNet()
    # model = Con()
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    loss_metric = tf.keras.metrics.Mean()

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)
    train_dataset = train_dataset.take(2000)

    epochs = 1

    for epoch in range(epochs):
        print('Start of epoch %d' % (epoch,))
        # Iterate over the batches of the dataset.
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):

            # Open a GradientTape to record the operations run
            # during the forward pass, which enables autodifferentiation.
            with tf.GradientTape() as tape:

                # Run the forward pass of the layer.
                # The operations that the layer applies
                # to its inputs are going to be recorded
                # on the GradientTape.
                logits = model(x_batch_train,
                               training=True)  # Logits for this minibatch

                # Compute the loss value for this minibatch.
                loss_value = loss_fn(y_batch_train, logits)

            # Use the gradient tape to automatically retrieve
            # the gradients of the trainable variables with respect to the loss.
            grads = tape.gradient(loss_value, model.trainable_weights)

            # Run one step of gradient descent by updating
            # the value of the variables to minimize the loss.
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            # Log every 200 batches.
            if step % 10 == 0:
                print('Training loss (for one batch) at step %s: %s' % (
                step, float(loss_value)))
                print('Seen so far: %s samples' % ((step + 1) * 64))


