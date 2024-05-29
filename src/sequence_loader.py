import torch
from torch.utils.data import Dataset, DataLoader
from letter import EOW, Letter
from typing import List, Tuple
import numpy as np
import pandas as pd
import os
import PIL.Image as Image


def get_sequence_loader(bigram_data, image_dir, length, transform, batch_size):
    dataset = SequenceDataset(bigram_data, image_dir, length, transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


class SequenceDataset(Dataset):
    """

    """

    # TODO generate function
    # TODO load images
    # TODO load expected distribution for regularization

    def __init__(self, bigrams_data, image_dir, length, transform=None, epsilon=0.05):
        self.image_dir = image_dir
        self.transform = transform
        bigrams_data = pd.read_csv(bigrams_data)

        self.letters = {}
        self.image_files = {}
        self.set_letters = set(bigrams_data['first']) | set(bigrams_data['second'])
        self.set_letters = sorted(list(self.set_letters))
        self.length = length

        self.letters['EOW'] = EOW('EOW', bigrams_data, epsilon)

        for letter in self.set_letters:
            self.letters[letter] = Letter(letter, bigrams_data, epsilon)

        self.not_end = []
        for letter in self.letters:
            if self.letters[letter].followers['EOW'] != 1.0:
                self.not_end.append(letter)

        self.index_dict = {}
        for idx, letter in enumerate(self.set_letters):
            self.index_dict[letter] = idx
            files = os.listdir(os.path.join(self.image_dir, letter))
            self.image_files[letter] = files

        self.index_dict['EOW'] = len(self.set_letters)

        num_letters = len(self.set_letters)
        self.expected_dist = torch.zeros((num_letters+1, num_letters))
        for letter in self.letters:
            followers = self.letters[letter].followers
            for follower in followers:
                if follower == 'EOW':
                    continue
                l_idx = self.index_dict[letter]
                f_idx = self.index_dict[follower]
                self.expected_dist[l_idx, f_idx] = followers[follower]

    def __len__(self):
        return self.length

    def __getitem__(
            self,
            idx,
    ) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._load_trigram()

    def _load_trigram(
            self
    ) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        trigram, end = self._generate_trigram()
        images = []
        for letter in trigram:
            img = self._get_image(letter)
            images.append(img)
        end = torch.tensor(end).float()
        end = end.unsqueeze(0)
        target = self._one_hot(trigram[1])
        dist = self.expected_dist[self.index_dict[trigram[0]]]
        images = torch.stack(images, dim=0)
        return images, target, end, dist

    def _generate_trigram(self) -> Tuple[List[str], bool]:
        # TODO add end + rand letter
        index = np.random.randint(len(self.not_end))
        current_letter = self.not_end[index]
        next_letter = self.letters[current_letter].random_next()
        ending = False
        if next_letter == 'EOW':
            # Reduce the chance of EOW letter EOW
            next_letter = self.letters[current_letter].random_next()

            if next_letter == 'EOW':
                next_letter = self._draw_none_eow_start()
                ending = True

            trigram = ['EOW', current_letter, next_letter]
        else:
            last_letter = self.letters[next_letter].random_next()
            if last_letter == 'EOW':
                ending = True
                last_letter = self._draw_none_eow_start()
            trigram = [current_letter, next_letter, last_letter]

        return trigram, ending

    def _draw_none_eow_start(self) -> str:
        """Draws a random letter that is not EOW."""
        index = np.random.randint(len(self.not_end))
        if self.not_end[index] == 'EOW':
            index = ((index + np.random.randint(1, len(self.not_end)))
                     % len(self.not_end))
        return self.not_end[index]

    def _get_image(self, letter: str) -> torch.Tensor:
        if letter == 'EOW':
            return self.transform(torch.zeros(1, 32, 32))
        rand_file = np.random.choice(self.image_files[letter])
        path = os.path.join(self.image_dir, letter, rand_file)
        img = Image.open(path).convert('L')
        img = torch.from_numpy(np.array(img)).float()
        img = img.unsqueeze(0)
        img = self.transform(img)
        return img

    def _one_hot(self, letter: str) -> torch.Tensor:
        ret = torch.zeros((len(self.set_letters)), dtype=torch.float)
        ret[self.index_dict[letter]] = 1

        return ret


if __name__ == '__main__':
    bigrams = pd.read_csv('../data/bigrams.csv')
    dataset = SequenceDataset(bigrams, './bigrams_data', 100)

    for _ in range(10):
        print(dataset._generate_trigram())
