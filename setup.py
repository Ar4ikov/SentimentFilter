# | Created by Ar4ikov
# | Время: 06.01.2020 - 14:08


from setuptools import setup
from os import path


class SentimentSetup:
    def __init__(self):
        self.package = "sentiment_filter"
        self.__version__ = open(path.join(path.dirname(__file__), self.package, "version.txt"), "r").read()

    def setup(self):
        setup(
            name="sentiment_filter",
            version=self.__version__,
            install_requires=["tensorflow", "pandas", "nltk"],
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
