# | Created by Ar4ikov
# | Время: 05.01.2020 - 11:54

from os import mkdir, listdir, path
from json import loads, dumps

from re import compile, findall, match
from nltk.stem.snowball import RussianStemmer
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer

import numpy as np
import pandas as pd


class SentimentDataset:
    def __init__(self, positive_path=None, negative_path=None, vocab_size=None, delimiter=None, regex=None):
        if vocab_size is None:
            vocab_size = -1

        if regex is None:
            self.regex = r"""[а-яА-Яa-zA-Zё]*"""
        else:
            self.regex = regex

        if delimiter is None:
            self.delimiter = ";"
        else:
            self.delimiter = delimiter

        if positive_path is None:
            self.positive_path = ""
        else:
            self.positive_path = positive_path

        if negative_path is None:
            self.negative_path = ""
        else:
            self.negative_path = negative_path

        def skip_loader(pos_path, neg_path):
            if (len(pos_path) == 0) or (len(neg_path) == 0):
                return True
            else:
                return False

        self.data = self.prepare_data(skip_loader=skip_loader(self.positive_path, self.negative_path))
        self.stemmed_words = self.tokenize_by_frequency(self.data)
        self.tokens = self.tokenize_by_iter(vocab_size, self.stemmed_words)
        self.default_vocab_path = path.join(path.dirname(__file__), "model", "vocab.json")

    def prepare_data(self, skip_loader=False):
        if not skip_loader:
            positive = pd.read_csv(self.positive_path, header=None, delimiter=self.delimiter)[[3]]
            positive = list(zip([tweet[3] for _, tweet in positive.iterrows()], [0 for _ in range(len(positive))]))

            negative = pd.read_csv(self.negative_path, header=None, delimiter=self.delimiter)[[3]]
            negative = list(zip([tweet[3] for _, tweet in negative.iterrows()], [1 for _ in range(len(negative))]))

            self.data = []
            self.data.extend(positive)
            self.data.extend(negative)

            print(self.data[0])

            return self.data
        else:
            return []

    def save_vocab(self, filename):
        with open(filename, "w") as file:
            file.write(dumps(self.tokens, indent=4))

        return self.tokens

    def load_vocab(self, filename):
        with open(filename, "r") as file:
            self.tokens = loads(file.read())

        return self.tokens

    @staticmethod
    def tokenize_by_frequency(data, count_: int = -1) -> dict:
        tokenize = {}
        for idx, item in enumerate([SentimentDataset.to_regex(x[0]) for x in data]):
            for word in item:
                stem = SentimentDataset.get_to_stem(word)
                if not stem in tokenize:
                    tokenize.update({stem: 1})
                else:
                    tokenize[stem] += 1

            if (idx + 1) % 100 == 0:
                print(f"{idx + 1} is done!", len(tokenize.keys()))

        tokenize = dict(sorted(tokenize.items(), key=lambda x: x[1], reverse=True)[:count_ if count_ > -1 else None])
        return tokenize

    @staticmethod
    def get_summary_words(wordlist: list, data: dict) -> int:
        return sum([v for k, v in data.items() if k in wordlist])

    @staticmethod
    def get_language(string) -> str:
        if findall(r"""[а-яА-Я]""", str(string)):
            return "russian"
        else:
            return "english"

    @staticmethod
    def get_to_stem(string: str):
        lang = {
            "russian": RussianStemmer,
            "english": PorterStemmer
        }

        return lang[SentimentDataset.get_language(string)]().stem(string)

    @staticmethod
    def tokenize_by_iter(count_: int, data: dict) -> dict:
        if count_ <= -1:
            count_ = len(data.keys())

        return {k: idx + 1 for idx, k in enumerate(data.keys()) if idx + 1 <= count_}

    @staticmethod
    def to_regex(phrase, regex=None, replace_none_word=True):
        if regex is None:
            regex = r"""[а-яА-Яa-zA-Zё]*"""

        if replace_none_word:
            return [x for x in findall(regex, str(phrase)) if x != ""]
        else:
            return [x for x in findall(regex, str(phrase))]

    @staticmethod
    def to_input_dim(phrase: str, vocab: dict):
        regex_list = SentimentDataset.to_regex(phrase)
        stem_list = [SentimentDataset.get_to_stem(x) for x in regex_list]

        dim = np.zeros([len(vocab.keys())], dtype=np.int_)

        for stem in stem_list:
            if stem in vocab:
                dim[list(vocab.keys()).index(stem)] = 1

        return dim

    @staticmethod
    def to_categorical(temp):
        func = lambda x: [0, 1] if x == 1 else [1, 0]
        return func(temp)

    @staticmethod
    def to_binary(temp: list):
        func = lambda x: 1 if np.array(temp, dtype=np.int_).tolist() == np.array([0, 1], dtype=np.int_).tolist() else 0
        return func(temp)

    @staticmethod
    def vocab_with_unknown_word(vocab):
        vocab = dict((k, v) for k, v in vocab.items() if v < len(vocab.items()))
        vocab.update({"_____UNK": len(vocab.keys()) + 1})

        return vocab

    @staticmethod
    def to_embedding_dim(phrase: str, vocab: dict, seq_length, use_unknown_word=True):
        regex = SentimentDataset.to_regex(phrase)
        stem = [SentimentDataset.get_to_stem(x) for x in regex]

        dim = np.zeros([seq_length], dtype=np.int_)
        for idx, item in enumerate(stem):
            if item in vocab:
                dim[idx] = vocab[item]
            else:
                if use_unknown_word:
                    dim[idx] = len(vocab.keys())

        return dim

    @staticmethod
    def train_data(word_list: list, vocab: dict):
        x, y = [], []
        for idx, item in enumerate(word_list):
            token, temp = item
            x.append(SentimentDataset.to_input_dim(token, vocab))
            y.append(SentimentDataset.to_categorical(temp))

            if (idx + 1) % 100 == 0:
                print(f"{idx + 1} is done!")

        x, y = np.array(x, dtype=np.int_), np.array(y, dtype=np.int_)

        return x, y

    @staticmethod
    def embedding_data(word_list: list, vocab: dict, seq_length=None, use_unknown_word=True):
        x, y = [], []
        print(word_list[1])

        if seq_length is None:
            seq_length = 0
            for z, _ in word_list:
                temp_ = len([SentimentDataset.get_to_stem(i) for i in SentimentDataset.to_regex(z)])
                if seq_length < temp_:
                    seq_length = temp_

        for idx, item in enumerate(word_list):
            token, temp = item
            x.append(SentimentDataset.to_embedding_dim(token, vocab, seq_length, use_unknown_word))
            y.append(SentimentDataset.to_categorical(temp))

            if (idx + 1) % 100 == 0:
                print(f"{idx + 1} is done!")

        x, y = np.array(x, dtype=np.int_), np.array(y, dtype=np.int_)

        return x, y, seq_length
