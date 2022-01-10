import abc
from itertools import permutations
import math
import random
import torch
from torch.utils.data import IterableDataset


class BinaryOpTorchDataset(IterableDataset):
    def __init__(self, dataset, type="train"):
        super(BinaryOpTorchDataset, self).__init__()
        assert type in ["train", "test"]
        self.type = type
        self.dataset = dataset

    def __iter__(self):
        return self

    def __next__(self):
        return self.dataset.get_sample(self.type)


class BinaryOpDataset(metaclass=abc.ABCMeta):
    def __init__(self, elements1, elements2, train_size, outlier_size):
        assert 0 <= train_size <= 1, "train_size must be in [0, 1]"
        assert 0 <= outlier_size <= 1, "outlier_size must be in [0, 1]"
        self.train_size = train_size
        self.outlier_size = outlier_size
        self.elements1 = elements1
        self.elements2 = elements2
        self.elements = ["o", "="] + elements1 + elements2
        self.idx_to_symbol_map = {i: sym for i, sym in enumerate(self.elements)}
        self.symbol_to_idx_map = {sym: i for i, sym in self.idx_to_symbol_map.items()}
        self.vocab_size = len(self.elements)
        self.output_dim = len(self.elements)
        self.num_equations = len(elements1) * len(elements2)
        self.train_data_n = int(train_size * self.num_equations)
        self.test_data_n = self.num_equations - self.train_data_n
        self.train_data_idxs, self.test_data_idxs = self.init_train_test_idxs()
        random.seed(99)

    def init_train_test_idxs(self):
        idxs = list(range(self.num_equations))
        random.shuffle(idxs)
        train_data_idxs = idxs[: self.train_data_n]
        test_data_idxs = idxs[self.train_data_n :]
        return train_data_idxs, test_data_idxs

    def embed_sequence(self, sequence):
        return [self.symbol_to_idx_map[sym] for sym in sequence]

    def unembed_sequence(self, sequence):
        return [self.idx_to_symbol_map[idx] for idx in sequence]

    @abc.abstractmethod
    def compute(self, x1, x2):
        pass

    def create_equation(self, x1, x2, y):
        return [x1, "o", x2, "=", y]

    def get_sample(self, type="train"):
        data_idxs = self.train_data_idxs if type == "train" else self.test_data_idxs
        idx = random.choice(data_idxs)
        x1 = self.elements1[idx // len(self.elements1)]
        x2 = self.elements2[idx % len(self.elements2)]
        y = self.compute(x1, x2)
        if type == "train" and random.random() <= self.outlier_size:
            x1_random = random.choice(self.elements1)
            x2_random = random.choice(self.elements2)
            y = self.compute(x1_random, x2_random)
        equation = self.create_equation(x1, x2, y)
        x_seq = self.embed_sequence(equation[:-1])
        y_seq = self.embed_sequence(equation[-1:])
        return torch.tensor(x_seq), torch.tensor(y_seq)


class SumModDataset(BinaryOpDataset):
    def __init__(self, high_val, train_size=1.0, outlier_size=0.0):
        self.high_val = high_val
        elements = list(range(high_val))
        super(SumModDataset, self).__init__(
            elements, elements, train_size, outlier_size
        )

    def compute(self, x1, x2):
        return (x1 + x2) % self.high_val


class SubtractModDataset(BinaryOpDataset):
    def __init__(self, high_val, train_size=1.0, outlier_size=0.0):
        self.high_val = high_val
        elements = list(range(high_val))
        super(SubtractModDataset, self).__init__(
            elements, elements, train_size, outlier_size
        )

    def compute(self, x1, x2):
        return (x1 - x2) % self.high_val


class DivideModDataset(BinaryOpDataset):
    def __init__(self, high_val, train_size=1.0, outlier_size=0.0):
        self.high_val = high_val
        elements = list(range(high_val))
        super(DivideModDataset, self).__init__(
            elements, elements, train_size, outlier_size
        )

    def compute(self, x1, x2):
        return (x1 * pow(x2, self.high_val - 2, self.high_val)) % self.high_val


class SquareSumModDataset(BinaryOpDataset):
    def __init__(self, high_val, train_size=1.0, outlier_size=0.0):
        self.high_val = high_val
        elements = list(range(high_val))
        super(SquareSumModDataset, self).__init__(
            elements, elements, train_size, outlier_size
        )

    def compute(self, x1, x2):
        return (x1 ** 2 + x2 ** 2) % self.high_val


class CubeSumModDataset(BinaryOpDataset):
    def __init__(self, high_val, train_size=1.0, outlier_size=0.0):
        self.high_val = high_val
        elements = list(range(high_val))
        super(CubeSumModDataset, self).__init__(
            elements, elements, train_size, outlier_size
        )

    def compute(self, x1, x2):
        return (x1 ** 3 + x2 ** 3) % self.high_val


class PermutationsDataset(BinaryOpDataset):
    def __init__(self, high_val, train_size=1.0, outlier_size=0.0):
        self.high_val = high_val
        elements = list(map(tuple, permutations(range(high_val))))
        super(PermutationsDataset, self).__init__(
            elements, elements, train_size, outlier_size
        )

    def compute(self, x1, x2):
        return tuple([x1[x2[i]] for i in range(len(x2))])
