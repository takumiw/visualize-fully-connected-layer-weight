import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    BatchNormalization,
    Dropout,
    Input,
    Activation,
)

weight_decay = 1e-4


def vgg_block(x, filters, layers):
    for _ in range(layers):
        x = Conv2D(
            filters,
            (3, 3),
            padding="same",
            kernel_initializer="he_normal",
            kernel_regularizer=regularizers.l2(weight_decay),
        )(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

    return x


def vgg8():
    input = Input(shape=(28, 28, 1))

    x = vgg_block(input, 16, 2)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = vgg_block(x, 32, 2)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = vgg_block(x, 64, 2)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Flatten()(x)
    x = Dense(
        2,
        kernel_initializer="he_normal",
        kernel_regularizer=regularizers.l2(weight_decay),
    )(x)
    x = BatchNormalization()(x)
    output = Dense(
        10,
        activation="softmax",
        kernel_regularizer=regularizers.l2(weight_decay),
        use_bias=False,
    )(x)

    return Model(input, output)
