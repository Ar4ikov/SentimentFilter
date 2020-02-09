# | Created by Ar4ikov
# | Время: 03.02.2020 - 00:47

from sentiment_filter.net import SentimentClassifizer
from sentiment_filter.dataset import SentimentDataset
from enum import Enum


class SentimentFilter:
    class Sentiment(Enum):
        POSITIVE = "positive"
        NEGATIVE = "negative"
        NEUTRAL = "neutral"

    def __init__(self):
        self.network = SentimentClassifizer()
        self.dataset = SentimentDataset()

        # Loading vocabulary
        self.dataset.load_vocab(self.dataset.default_vocab_path)

    def get_vector(self, data, seq_length=100):
        regex = self.dataset.to_regex(data)
        size = [self.dataset.get_to_stem(x) for x in regex]

        vector = self.dataset.to_embedding_dim(data, self.dataset.tokens,
                                               len(size) if seq_length < len(size) else seq_length)

        return vector

    def is_negative(self, data, score=0.67, seq_length=100):
        vector = self.get_vector(data, seq_length)

        response = self.network.predict(vector, seq_length=seq_length)

        if response >= score:
            return True
        else:
            return False

    def is_positive(self, data, score=0.45, seq_length=100):
        vector = self.get_vector(data, seq_length)

        response = self.network.predict(vector, seq_length=seq_length)

        if response >= score:
            return False
        else:
            return True

    def is_neutral(self, data, scores=None, seq_length=100):
        if scores is None:
            scores = [0.45, 0.67]

        vector = self.get_vector(data, seq_length)

        response = self.network.predict(vector, seq_length=seq_length)

        if scores[0] <= response <= scores[1]:
            return True
        else:
            return False

    def get_analysis(self, data, scores=None, seq_length=100):
        if scores is None:
            scores = [0.45, 0.67]

        vector = self.get_vector(data, seq_length)

        response = self.network.predict(vector, seq_length=seq_length)

        result = {"result": None, "score": response}

        if response >= scores[1]:
            result.update({"result": self.Sentiment.NEGATIVE})
        elif response <= scores[0]:
            result.update({"result": self.Sentiment.POSITIVE})
        else:
            result.update({"result": self.Sentiment.NEUTRAL})

        return result
