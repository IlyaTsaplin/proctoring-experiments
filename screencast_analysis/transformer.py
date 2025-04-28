import random

import numpy as np
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization
from tensorflow.keras.layers import MultiHeadAttention, GlobalAveragePooling1D
from matplotlib import pyplot as plt

import dataset_constants

# Model parameters
HEAD_SIZE = 64
NUM_HEADS = 4
FF_DIM = 64
NUM_TRANSFORMER_BLOCKS = 4
DROPOUT_RATE = 0.2

# Training parameters
WINDOW_SIZE = 16
EPOCH_COUNT = 30
VALIDATION_SPLIT = 0.2
TRANSFORMER_PATH = "model_weights/transformer_model_{}.keras"


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(
        x, x
    )
    x = Dropout(dropout)(x)
    res = x + inputs

    x = LayerNormalization(epsilon=1e-6)(res)
    x = Dense(ff_dim, activation="relu")(x)
    x = Dropout(dropout)(x)
    x = Dense(inputs.shape[-1])(x)
    return x + res


class Transformer_Creator:
    def create_model(self, input_shape, num_classes):
        # Build the model
        inputs = Input(shape=input_shape)
        x = inputs
        for _ in range(NUM_TRANSFORMER_BLOCKS):
            x = transformer_encoder(x, HEAD_SIZE, NUM_HEADS, FF_DIM, DROPOUT_RATE)

        x = GlobalAveragePooling1D()(x)
        x = Dropout(DROPOUT_RATE)(x)
        x = Dense(FF_DIM, activation="relu")(x)
        x = Dropout(DROPOUT_RATE)(x)
        outputs = Dense(num_classes, activation="softmax")(x)

        model = Model(inputs, outputs)
        return model


def get_model(class_key_template: str | None = None):
    """
    Retrieves a transformer model, optionally loading weights from a file.

    Args:
        class_key_template (str | None): A template string containing a type of data for a model (e.g. counts or text lengths).

    Returns:
        Model: A transformer model for a specific type of data
    """
    model = Transformer_Creator().create_model(
        input_shape=(WINDOW_SIZE * len(dataset_constants.CLASSES), 1), num_classes=2
    )
    if class_key_template is not None:
        model.load_weights(TRANSFORMER_PATH.format(class_key_template[3:]))
    return model


def train_transformer(values_key_template: str, show_plots: bool = False):
    """
    Trains a transformer model.

    Args:
        values_key_template (str): A template string containing a type of data for a model (e.g. counts or text lengths).
        show_plots (bool, optional): Whether to display plots of the training and validation accuracy. Defaults to False.
    """
    df = pd.read_csv(dataset_constants.DATASET_PATH)
    model = get_model()

    X_train = []
    y_train = []
    for _, video_row in df.iterrows():
        for window_index in range(
            int(video_row[dataset_constants.VIDEO_LENGTH_KEY] - WINDOW_SIZE)
        ):
            X_values = []
            window_start = window_index
            window_end = window_start + WINDOW_SIZE

            for class_name in dataset_constants.CLASSES:
                class_values = eval(video_row[values_key_template.format(class_name)])[
                    window_start:window_end
                ]
                X_values.extend((int(x) for x in class_values))

            X_train.append(X_values)
            y_train.append(int(video_row[dataset_constants.IS_CHEATED_KEY]))

    # shuffle data
    values = [(x1, y1) for x1, y1 in zip(X_train, y_train)]
    random.shuffle(values)
    X_train, y_train = zip(*values)

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    print(X_train.shape)

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    history = model.fit(
        X_train, y_train, epochs=EPOCH_COUNT, validation_split=VALIDATION_SPLIT
    )

    if show_plots:
        print(history.history["accuracy"])
        plt.plot(history.history["accuracy"])
        plt.title("Transformer accuracy")
        plt.show()

        print(history.history["val_accuracy"])
        plt.plot(history.history["val_accuracy"])
        plt.title("Transformer validation accuracy")
        plt.show()

    model.save(TRANSFORMER_PATH.format(values_key_template[3:]))


if __name__ == "__main__":
    train_transformer(dataset_constants.COUNTS_KEY, show_plots=True)
    train_transformer(dataset_constants.TEXT_LENGTHS_KEY, show_plots=True)
