import tensorflow as tf
# import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import datasets, layers


# noinspection SpellCheckingInspection,PyShadowingNames
class ConvBlock(tf.keras.Model):

    def __init__(self, filters=(32, 64, 128)):
        super(ConvBlock, self).__init__()
        filter_1, filter_2, filter_3 = filters
        self.conv1 = layers.Conv2D(filter_1, 3, padding='same')
        self.conv2 = layers.Conv2D(filter_2, 3, padding='same')
        self.conv3 = layers.Conv2D(filter_3, 3, padding='same')
        self.relu1 = layers.ReLU()
        self.relu2 = layers.ReLU()
        self.relu3 = layers.ReLU()
        self.max_pool1 = layers.MaxPool2D()
        self.max_pool2 = layers.MaxPool2D()

    # noinspection PyMethodOverriding
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.relu1(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.max_pool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        return x


class EndBlock(tf.keras.Model):

    def __init__(self):
        super(EndBlock, self).__init__()
        self.avg_pool = layers.AveragePooling2D()
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(128)
        self.relu = layers.ReLU()
        self.dense2 = layers.Dense(10)
        self.softmax = layers.Softmax()

    # noinspection PyMethodOverriding
    def call(self, inputs):
        x = self.avg_pool(inputs)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dense2(x)
        x = self.softmax(x)
        return x


class MyNet(tf.keras.Model):

    def __init__(self):
        super(MyNet, self).__init__()
        self.conv_block = ConvBlock()
        self.end_block = EndBlock()

    # noinspection PyMethodOverriding
    def call(self, inputs):
        x = self.conv_block(inputs)
        x = self.end_block(x)
        return x


def get_gradient(x, y, model, loss_fn):
    """Compute the gradient of the loss as a function of x

    Args:
        x: tf.Tensor
        y: tf.Tensor
        model: tf.Model
        loss_fn: tf.keras.losses

    Returns: tf.Tensor

    """
    with tf.GradientTape() as tape:
        tape.watch(x)
        prediction = model(x)
        loss = loss_fn(y, prediction)

    # Get the gradients of the loss w.r.t to the input image.
    gradient = tape.gradient(loss, x)
    return gradient


def sign_gradient(gradient):
    """Get the sign of the gradients to create the perturbation

    Args:
        gradient: tf.Tensor

    Returns: tf.Tensor

    """
    return tf.sign(gradient)


def generate_sign_perturbation(x, eta, sign_gradient):
    """Generates perturbated elements from an initial tensor x

    Args:
        x: tf.Tensor
        eta: scalar
        sign_gradient: tf.Tensor

    Returns:

    """

    x_perturbated = x + eta * sign_gradient
    return x_perturbated


# noinspection PyUnusedLocal
def fgsm(x, y, model, loss_fn, eta=0.01, **kwargs):
    gradient = get_gradient(x, y, model, loss_fn)
    signed_gradient = sign_gradient(gradient)
    x_adv = generate_sign_perturbation(x, eta, signed_gradient)
    return x_adv


def pgd_infinity(x, y, model, loss_fn, eta=0.01, eps=0.1, n_steps=2):
    x_adv = x
    for i in range(n_steps):
        # Perturbation
        perturbation = fgsm(x_adv, y, model, loss_fn, eta)
        # Projection
        x_adv = tf.clip_by_value(perturbation, x - eps, x + eps)
    return x_adv


def pgd_ininity_random(x, y, model, loss_fn, eta=0.01, eps=0.1, n_steps=2):
    """PGD l-infinity with random init uniform centered in x and of radius
    eps / 2.

    Args:
        x:
        y:
        model:
        loss_fn:
        eta:
        eps:
        n_steps:

    Returns:

    """
    x_random = x + tf.random.uniform(shape=x.shape,
                                     minval=-eps / 2,
                                     maxval=eps / 2,
                                     seed=94,
                                     dtype=tf.dtypes.float64)
    x_adv = pgd_infinity(x_random, y, model, loss_fn, eta, eps, n_steps)
    return x_adv


def projection_l2(x, x_adv, eps):
    """Project x_adv onto the l2-ball centered in x

    Args:
        x: tf.Tensor, ball center.
        x_adv: tf.Tensor to be projected.
        eps: scalar, radius,

    Returns:
        tf.Tensor projected onto the ball.
    """
    # Reshape tensor to compute the norm easily, keep batch axis
    delta = tf.reshape(x - x_adv, [x.shape[0], -1])
    norm_x = tf.norm(delta, axis=1, ord='euclidean')
    scale = tf.math.minimum(1, eps / norm_x)
    scale = tf.reshape(scale, (-1, 1, 1, 1))
    return x + scale * (x - x_adv)

    # norm_x = tf.norm(delta, axis=1, ord='euclidean')
    # tf.constant([1.0, 2.0]) * tf.ones((10,2,2,3))
    # tst = tf.reshape(tf.constant([1.0, .0], dtype=tf.dtypes.float64), (-1,1,1,1))
    # tf.reduce_mean(x_adv, axis=(1, 2, 3))
    # tf.reduce_mean(tf.multiply(x_adv, tst), axis=(1, 2, 3))
    # pgd_l2(x, y, model, loss_fn, n_steps=3)
    # pgd_infinity(x, y, model, loss_fn, n_steps=3)


def pgd_l2(x, y, model, loss_fn, eta=0.01, eps=0.1, n_steps=2):
    """PGD

    Args:
        x: tf.Tensor, input to attack.
        y: tf.Tensor, associated labels.
        model: tf.Model, model to attack.
        loss_fn: tf.losses, loss to attack.
        eta: scalar, step size.
        eps: scalar, radius of the ball.
        n_steps: int, number of iterations.
    """
    x_adv = x
    for i in range(n_steps):
        # Perturbation
        perturbation = fgsm(x_adv, y, model, loss_fn, eta)
        x_adv = projection_l2(x, perturbation, eps)
    return x_adv


def pgd_l2_random(x, y, model, loss_fn, eta=0.01, eps=0.1, n_steps=2):
    """PGD with random start.

    Args:
        x: tf.Tensor, input to attack.
        y: tf.Tensor, associated labels.
        model: tf.Model, model to attack.
        loss_fn: tf.losses, loss to attack.
        eta: scalar, step size.
        eps: scalar, radius of the ball.
        n_steps: int, number of itertions.
    """
    x_random = x + tf.random.uniform(shape=x.shape,
                                     minval=-eps / 2,
                                     maxval=eps / 2,
                                     seed=94,
                                     dtype=tf.dtypes.float64)

    x_adv = pgd_l2(x_random, y, model, loss_fn, eta, eps, n_steps)
    return x_adv


if __name__ == '__main__':

    np.random.seed(94)
    tf.random.set_seed(94)
    (x_train, y_train), (
            x_test, y_test) = datasets.cifar10.load_data()

    # One hot conversion
    y_train = tf.one_hot(y_train, 10)
    y_test = tf.one_hot(y_test, 10)

    # Normalize pixel values to be between 0 and 1
    x_train, x_test = x_train / 255.0, x_test / 255.0

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    # Model def
    model = MyNet()
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    # loss_fn = tf.keras.losses.MeanSquaredError()
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)
    train_dataset = train_dataset.take(500)

    epochs = 1
    train_loss_results = []
    train_accuracy_results = []
    # Training
    for epoch in range(epochs):
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_accuracy = tf.keras.metrics.CategoricalAccuracy()
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
                softmax = model(x_batch_train,
                                training=True)  # Logits for this minibatch

                # Compute the loss value for this minibatch.
                loss_value = loss_fn(y_batch_train, softmax)

            # Use the gradient tape to automatically retrieve
            # the gradients of the trainable variables with respect to the loss.
            grads = tape.gradient(loss_value, model.trainable_weights)

            # Run one step of gradient descent by updating
            # the value of the variables to minimize the loss.
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            # End epoch

            # Track progress
            epoch_loss_avg.update_state(loss_value)  # Add current batch loss
            # Compare predicted label to actual label
            # training=True is needed only if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            epoch_accuracy.update_state(y_batch_train,
                                        softmax)

            train_loss_results.append(epoch_loss_avg.result())
            train_accuracy_results.append(epoch_accuracy.result())

            if step % 200 == 0:
                print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(
                        epoch,
                        epoch_loss_avg.result(),
                        epoch_accuracy.result()))

    #   ###############

    x = x_test[1:3]
    x = tf.constant(x)
    # x_adv = tf.expand_dims(tf.constant(x_adv), 0)
    # plt.imshow(x_adv[0])
    y = y_train[1:3]
    y = tf.constant(y)

    loss_fn = tf.keras.losses.MeanSquaredError()

    # def pgd_infinity(x, y, model, loss_fn, eta=0.01, eps=0.1, n_step=2):

    eta = 0.01
    gradient = get_gradient(x, y, model, loss_fn)
    signed_gradient = sign_gradient(gradient)
    # pgd_infinity(x, y, model, loss_fn, eta=0.01, eps=0.1,
    #                   n_step=1) == fgsm(x, y, model, loss_fn, eta=0.01)

    tf.reduce_sum(fgsm(x, y, model, loss_fn, eta=0.01) > 1)

    tf.reduce_sum(
            tf.cast(generate_sign_perturbation(x, 0.2, signed_gradient) > 1,
                    dtype=tf.int32))

    list_eta = [0, 0.001, 0.01, 0.1]

    x_adv = fgsm(x, y, model, loss_fn, eta=0.01)
    x_adv2 = fgsm(x_adv, y, model, loss_fn, eta=0.01)

    # tst = [x_adv + eta * signed_grad for eta in list_eta]
    # tst = [tf.clip_by_value(x, 0, 1) for x in tst]

    # Print images
    #
    # for i in range(len(tst)):
    #     plt.subplot(5, 5, i + 1)
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.grid(False)
    #     plt.imshow(tst[i][0])
    #
    #     pred = tf.nn.softmax(model.predict(tst[i]))
    #     print(tf.math.argmax(pred))
    #     print(np.argmax(pred))
    #     print(pred)
    #     # The CIFAR labels happen to be arrays,
    #     # which is why you need the extra index
    #     # plt.xlabel(class_names[y_train[i][0]])
    #
