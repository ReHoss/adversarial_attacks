import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import load_model


def fgsm(x, y, model, loss, epsilon=0.1):
    x = tf.constant(x)
    y = tf.constant(y)
    with tf.GradientTape() as g:
        g.watch(x)
        loss_fn = loss(y, model(x))
    perturbation = np.sign(g.gradient(loss_fn, x))
    x_attack = x + epsilon * perturbation
    x_attack = np.clip(x_attack, 0, 1)
    return x_attack


def pgd(x, y, model, loss, nb_step=5, epsilon=0.1, eta=0.01):
    x = tf.constant(x)
    y = tf.constant(y)
    for _ in range(nb_step):
        x_iter = fgsm(x, y, model, loss, eta)
        x_iter = np.clip(x_iter, x-epsilon, x+epsilon)
    return x_iter


if __name__ == '__main__':

    tf.random.set_seed(94)
    np.random.seed(94)

    (X_train_full, y_train_full), (X_test, y_test) = cifar10.load_data()
    X_train, X_valid = X_train_full[:40000] / 255.0, X_train_full[40000:] / 255.0
    y_train, y_valid = y_train_full[:40000], y_train_full[40000:]
    X_test = X_test / 255.0

    model = load_model("model_20200418-172129")

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    x_attack = fgsm(X_test, y_test, model, loss_fn)
    x_pgd = pgd(X_test, y_test, model, loss_fn)

    print(x_attack.shape)
    print(x_pgd.shape)

    model.evaluate(x_attack, y_test)
    model.evaluate(x_pgd, y_test)
    model.evaluate(X_test, y_test)
