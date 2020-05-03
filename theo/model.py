import numpy as np
import datetime
import typing
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, AveragePooling2D, Dropout
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping


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
        # x = Dropout(0.2, seed=94)(x)
        x = MaxPooling2D((2, 2))(x)

    x = AveragePooling2D()(x)
    x = Flatten()(x)

    for nb_units, activation in config.dense_layers:
        x = Dense(units=nb_units, activation=activation)(x)

    model = Model(
        inputs=inp,
        outputs=x,
        name="model"
    )

    model.compile(
        loss="categorical_crossentropy",
        optimizer='adam',
        metrics=['accuracy']
    )
    return model


if __name__ == '__main__':

    now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    (x_train_full, y_train_full), (x_test, y_test) = cifar10.load_data()
    x_train, x_valid = x_train_full[:40000] / 255.0, x_train_full[40000:] / 255.0
    y_train, y_valid = y_train_full[:40000], y_train_full[40000:]
    x_test = x_test / 255.0

    config = ModelConfig(
        conv_layers=[(64, 3), (128, 3), (256, 3)],
        dense_layers=[(256, "relu"), (10, "softmax")],
        epochs=20
    )

    model = create_model(config)
    model.summary()

    early_stopping_cb = EarlyStopping(patience=4, restore_best_weights=True)

    history = model.fit(x_train, tf.keras.utils.to_categorical(y_train, 10), epochs=config.epochs,
                        validation_data=(x_valid, tf.keras.utils.to_categorical(y_valid, 10)))
                        # callbacks=[early_stopping_cb])

    test_loss, test_acc = model.evaluate(x_test, tf.keras.utils.to_categorical(y_test, 10))
    print("loss {}, accuracy {} on test set".format(test_loss, test_acc))
    model.save("model_{}".format(now))

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.gca().set_ylim(0, 1.0)
    plt.title('Accuracy evolution')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    plt.savefig("accuracy_evolution_{}.png".format(now))

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.gca().set_ylim(0, 2.2)
    plt.title('Loss evolution')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    plt.savefig("loss_evolution_{}.png".format(now))
