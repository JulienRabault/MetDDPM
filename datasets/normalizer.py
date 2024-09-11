from abc import ABC, abstractmethod


class Normalizer(ABC):
    @abstractmethod
    def normalize(self, data):
        pass

    @abstractmethod
    def denormalize(self, data):
        pass


class MinMaxNormalizer(Normalizer):
    def __init__(self, min_values, max_values):
        self.min_values = min_values
        self.max_values = max_values

    def normalize(self, data):
        return (data - self.min_values) / (self.max_values - self.min_values)

    def denormalize(self, data):
        return data * (self.max_values - self.min_values) + self.min_values


class MeanStdNormalizer(Normalizer):
    def __init__(self, mean_values, std_values):
        self.mean_values = mean_values
        self.std_values = std_values

    def normalize(self, data):
        return (data - self.mean_values) / self.std_values

    def denormalize(self, data):
        return data * self.std_values + self.mean_values
