import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import load_model


def fgsm(x, y, model, loss, epsilon=0.031):
    x = tf.constant(x, dtype=tf.float32)
    y = tf.constant(y)
    with tf.GradientTape() as g:
        g.watch(x)
        loss_fn = loss(y, model(x))
    perturbation = np.sign(g.gradient(loss_fn, x))
    x_attack = x + epsilon * perturbation
    x_attack = np.clip(x_attack, 0, 1)
    return x_attack


def pgd(x, y, model, loss, alpha=0.01, epsilon=0.031, eta=0.011):
    x_origin = x
    i = 0
    evaluation = model.evaluate(x, y, verbose=0)
    value_previous_loss = evaluation[0]
    value_loss = evaluation[0] + alpha + 0.1
    list_loss, list_accuracy, list_interation = [evaluation[0]], [evaluation[1]], [i]
    print(i)
    print(evaluation[0], ' ', evaluation[1])
    i += 1
    while np.abs(value_loss - value_previous_loss) >= alpha:
        x = fgsm(x, y, model, loss, eta)
        x = np.clip(x, x_origin-epsilon, x_origin+epsilon)
        evaluation = model.evaluate(x, y, verbose=0)
        value_previous_loss, value_loss = value_loss, evaluation[0]
        list_loss.append(evaluation[0])
        list_accuracy.append(evaluation[1])
        list_interation.append(i)
        print(i)
        print(evaluation[0], ' ', evaluation[1])
        i += 1
    return x, list_loss, list_accuracy, list_interation


if __name__ == '__main__':

    tf.random.set_seed(94)
    np.random.seed(94)

    (x_train_full, y_train_full), (x_test, y_test) = cifar10.load_data()
    x_train, x_valid = x_train_full[:40000] / 255.0, x_train_full[40000:] / 255.0
    y_train, y_valid = y_train_full[:40000], y_train_full[40000:]
    x_test = x_test / 255.0

    model = load_model("theo/model_32_64_128_dropout_14e")

    loss_fn = tf.keras.losses.CategoricalCrossentropy()

    x_fgsm = fgsm(x_test, tf.keras.utils.to_categorical(y_test, 10), model, loss_fn)
    x_pgd = pgd(x_test, tf.keras.utils.to_categorical(y_test, 10), model, loss_fn)

    model.evaluate(x_fgsm, tf.keras.utils.to_categorical(y_test, 10))
    model.evaluate(x_pgd, tf.keras.utils.to_categorical(y_test, 10))
    model.evaluate(x_test, tf.keras.utils.to_categorical(y_test, 10))
