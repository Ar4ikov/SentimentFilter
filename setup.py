# | Created by Ar4ikov
# | Время: 06.01.2020 - 14:08

from setuptools import setup
from os import path
from sys import argv

USE_GPU = True


class SentimentSetup:
    def __init__(self):
        self.package = "sentiment_filter"
        self.__version__ = open(path.join(path.dirname(__file__), self.package, "version.txt"), "r").read()
        with open("requirements.txt", "r") as file:
            self.reqs = [x.replace("\n", "") for x in file.readlines()]

        if USE_GPU is True:
            index_ = [(idx, x) for idx, x in enumerate(self.reqs) if "tensorflow" in x]
            self.reqs[index_[0][0]] = index_[0][1].replace("tensorflow-cpu", "tensorflow-gpu")
            print("USE_GPU")

    def setup(self):
        setup(
            name="sentiment_filter",
            version=self.__version__,
            install_requires=self.reqs,
            packages=[self.package],
            include_package_data=True,
            package_data={self.package: ["*.txt", "*.json", "*.h5", "model/*"]},
            license="MIT Licence",
            author="Nikita Archikov",
            author_email="bizy18588@gmail.com",
            description="Neural Network Filter that gives score to input text",
            keywords="opensource sentiment sentimental analysis neural network net tensorflow filter score"
        )


SentimentSetup().setup()
