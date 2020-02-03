from os import path
from statistics import mean

import numpy as np
from tensorflow.keras.activations import relu, sigmoid, softmax
from tensorflow.keras.layers import Input, Embedding, Dropout, Conv1D, MaxPool1D, LSTM, Flatten, Concatenate, Dense
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.optimizers import Adam


class SentimentClassifizer:
    def __init__(self, use_trained_model=True):
        if use_trained_model:
            self.model = load_model(path.join(path.dirname(__file__), "model/sentiment_filter.h5"))
        else:
            self.model = None

    @staticmethod
    def build_net(seq_length, vocab_size, conv_layers=4, use_lstm=True, is_binary=True, compile=True):
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

    @staticmethod
    def reshape_x(x, seq_length=100):
        zeros = np.zeros([seq_length])
        for idx, item in enumerate(x):
            zeros[idx] = item

        return zeros

    @staticmethod
    def divide_by_k(x, y):
        k = x // y if x % y == 0 else x // y + 1

        part = x // k

        if part * k < x:
            response = [part for _ in range(k)]
            while sum(response) < x:
                for i in range(len(response) if len(response) <= (x - sum(response)) else x - sum(response)):
                    response[i] += 1

            return response
        else:
            return [part for _ in range(k)]

    def predict(self, x, seq_length=100):
        if self.model is None:
            raise ValueError("Model is not trained or not announced.")

        k = self.divide_by_k(len(x), seq_length)
        parts = []
        intervals = [[0, k[0] - 1]]
        [intervals.append([intervals[-1][1], intervals[-1][1] + i]) for i in k[1:] if len(k) > 1]

        if len(k) > 1:
            for s, f in intervals:
                parts.append(self.reshape_x(x[s:f]))

        else:
            parts.append(x)

        response = round(mean([x[0] for x in self.model.predict(np.array(parts))]), 3)

        return response
