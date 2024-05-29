import numpy as np
import pandas as pd
from collections import defaultdict


class BaseLetter:

    def __init__(self, name):
        self.name = name
        self.followers = defaultdict(float)

    def random_next(self):
        rand_float = np.random.random()
        for follower in self.followers:
            rand_float -= self.followers[follower]
            if rand_float <= 0:
                return follower

    @staticmethod
    def _relative_frequency(letter, data):
        n_first = np.sum(data[data['first'] == letter]['count'])
        n_second = np.sum(data[data['second'] == letter]['count'])
        return n_first, n_second


class Letter(BaseLetter):
    def __init__(self, name, data, eps=0.01):
        super(Letter, self).__init__(name)

        n_first, n_second = self._relative_frequency(name, data)
        f_eos = max(0, (n_second - n_first)) + eps * min(n_second, n_first)

        data = data[data['first'] == name]

        if 'medial' in name:
            f_eos = 0.0

        self.followers['EOW'] = f_eos
        n_first += f_eos

        for follower, freq in zip(data['second'], data['count']):
            self.followers[follower] += freq

        for follower in self.followers:
            self.followers[follower] /= n_first


class EOW(BaseLetter):
    def __init__(self, name, data, eps=0.01):
        super(EOW, self).__init__(name)
        total = 0
        let_set = list(set(data['first']))

        for letter in let_set:
            n_first, n_second = self._relative_frequency(letter, data)
            self.followers[letter] += (max(0, (n_first - n_second))
                                       + eps * min(n_second, n_first))
            total += self.followers[letter]

        for follower in self.followers:
            self.followers[follower] /= total
