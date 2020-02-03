# | Created by Ar4ikov
# | Время: 04.02.2020 - 01:29

from keras.layers import Input, Dense, Dropout, Embedding, Conv1D, MaxPool1D, Flatten, LSTM, Concatenate
from keras.models import Model
from keras.activations import relu, softmax, sigmoid
from keras.optimizers import Adam


def cnn_lstm(seq_length, vocab_size, conv_layers=4, use_lstm=True, is_binary=True, compile=True):
    """
    Архитектура сети CNN + LSTM
    :param seq_length: Длина вектора
    :param vocab_size: Размер словаря
    :param conv_layers: Количество сверточных слоёв на уровне
    :param use_lstm: bool Нужен ли LSTM слой на каждом свёрточном слое
    :param is_binary: bool Тип выхода (True - бинарный, False - двоичный (categorical) )
    :param compile: bool Компиляция модели
    :return: keras.models.Model
    """

    input_layer = Input(shape=seq_length)
    embedding = Embedding(input_dim=vocab_size + 1, output_dim=64)(input_layer)
    dropout = Dropout(0.25)(embedding)

    conv_blocks = []
    for _ in range(conv_layers):
        conv = Conv1D(128, 5, padding="valid", activation=relu)(dropout)
        pooling = MaxPool1D(pool_size=4)(conv)

        if use_lstm:
            conv_out = LSTM(70)(pooling)
        else:
            conv_out = Flatten()(pooling)

        conv_blocks.append(conv_out)

    dense = Concatenate()(conv_blocks)
    dense = Dropout(0.2)(dense)
    dense = Dense(50, activation=relu)(dense)

    output_layer = Dense(1 if is_binary else 2, activation=sigmoid if is_binary else softmax)(dense)

    model = Model(input_layer, output_layer, name=f"Sentiment_CNN_{conv_layers}{'_LSTM' if use_lstm else ''}")

    if compile:
        model.compile(optimizer=Adam(), loss="binary_crossentropy" if is_binary else "categorical_crossentropy")

    return model
