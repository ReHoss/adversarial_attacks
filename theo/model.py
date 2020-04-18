import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, BatchNormalization, MaxPooling2D
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Model
from tensorflow import keras
import numpy as np
import datetime
import typing
import matplotlib.pyplot as plt


class ModelConfig(typing.NamedTuple):
    conv_layers: typing.List
    dense_layers: typing.List
    epochs: int


def create_model(config: ModelConfig):
    inp = Input(shape=(32, 32, 3))

    np.random.seed(94)
    tf.random.set_seed(94)

    x = inp

    for filters, kernel_size in config.conv_layers:
        x = Conv2D(filters=filters, kernel_size=(kernel_size, kernel_size), activation='relu')(x)
        x = MaxPooling2D((2, 2))(x)

    x = Flatten()(x)

    for nb_units, activation in config.dense_layers:
        x = Dense(units=nb_units, activation=activation)(x)

    model = Model(
        inputs=inp,
        outputs=x,
        name="model"
    )
    ####

    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer='adam',
        metrics=['accuracy']
    )
    return model


if __name__ == '__main__':

    tf.random.set_seed(94)
    np.random.seed(94)

    (X_train_full, y_train_full), (X_test, y_test) = cifar10.load_data()
    X_train, X_valid = X_train_full[:40000] / 255.0, X_train_full[40000:] / 255.0
    y_train, y_valid = y_train_full[:40000], y_train_full[40000:]
    X_test = X_test / 255.0

    config = ModelConfig(
        conv_layers=[(64, 4), (128, 4), (256, 4)],
        dense_layers=[(256, "relu"), (10, None)],
        epochs=20
    )

    model = create_model(config)
    model.summary()

    # log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # tensorboard_callback = keras.callbacks.TensorBoard(log_dir)

    history = model.fit(X_train, y_train, epochs=config.epochs, validation_data=(X_valid, y_valid))

    plt.plot(pd.DataFrame(history.history))
    plt.grid(True)
    plt.gca().set_ylim(0, 2.2)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    plt.savefig("loss_evolution.png")

    test_loss, test_acc = model.evaluate(X_test, y_test)
    print("loss {}, accuracy {} on test set".format(test_loss, test_acc))
    model.save("model_{}".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
