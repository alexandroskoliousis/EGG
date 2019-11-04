# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

import pathlib
import numpy as np
import torch
import torch.utils.data as data

def findposition(value, alist):
    if value==alist[0]:
        return 0
    if value==alist[-1]:
        return 1

    for i, element in enumerate(alist):
        if value >= element and value < alist[i+1]:
            return element

def build_distance_matrix(dataset):
    n = len(dataset)
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            color_i, color_j = dataset[i], dataset[j]
            x_i, y_i, z_i = color_i[1], color_i[2], color_i[3]
            x_j, y_j, z_j = color_j[1], color_j[2], color_j[3]

            dist_x = (x_i - x_j)**2
            dist_y = (y_i - y_j)**2
            dist_z = (z_i - z_j)**2
            dist = np.sqrt(dist_x + dist_y + dist_z)

            id_i = int(color_i[0].item())
            id_j = int(color_j[0].item())
            distance_matrix[id_i, id_j] = dist
            distance_matrix[id_j, id_i] = dist
    return distance_matrix

def ColorData(chip_file=None):
    if chip_file is None:
        chip_file = pathlib.Path(__file__).parent / 'data/lab_coor.txt'
    else:
        chip_file = pathlib.Path(chip_file)

    if not chip_file.exists():
        raise FileNotFoundError(f'Cannot find chip file {chip_file}')

    data = np.loadtxt(chip_file, dtype=str, delimiter='\t')
    data = [torch.tensor([int(index)-1, float(x_value), float(y_value), float(z_value)]) for (index, x_value, y_value, z_value) \
                                        in zip(data[:,0], data[:,6], data[:,7], data[:,8])]
    return data

def sample_min(_list, min_value, n, random_state, distance_list):
    distractors = []
    for _ in range(n):
        potential_distr = random_state.choice(len(_list), replace=False, size=1)[0]
        potential_distr = _list[potential_distr]
        id_distr= int(potential_distr[0])
        distance = distance_list[id_distr]
        while distance < min_value:
            potential_distr = random_state.choice(len(_list), replace=False, size=1)[0]
            potential_distr = _list[potential_distr]
            id_distr= int(potential_distr[0])
            distance = distance_list[id_distr]
        distractors.append(potential_distr.unsqueeze(0))
    return distractors

class _ColorIterator:

    def __init__(self, n_distractor, n_batches_per_epoch, batch_size, distance_matrix, min_value, data, train=True, seed=None):
        self.n_batches_per_epoch = n_batches_per_epoch
        self.n_distractor = n_distractor
        self.batch_size = batch_size
        self.batches_generated = 0
        self.random_state = np.random.RandomState(seed)
        self.distance_matrix = distance_matrix
        self.data = data
        self.min_value = min_value
        self.train = train

    def __iter__(self):
        return self

    def __next__(self):
        if self.batches_generated >= self.n_batches_per_epoch:
            raise StopIteration()
        if self.train:
            targets_nb = self.random_state.choice(len(self.data), replace=True, size=self.batch_size)
        else:
            targets_nb = range(len(self.data))
        # create distractors
        batch_distractor = []
        batch_targets = []
        for target_nb in targets_nb:
            target = self.data[target_nb]
            batch_targets.append(target.unsqueeze(0))
            id_target = int(target[0].item())
            distances = self.distance_matrix[id_target]
            # Sample n distractor with a disntace > min_value
            distractor_chips = sample_min(self.data, self.min_value, self.n_distractor, self.random_state, distances)
            batch_distractor.append(distractor_chips)

        # get label
        labels = []
        inputs = []
        for target, distractors in zip(batch_targets, batch_distractor):
            distractors = torch.cat(distractors, 0)
            receiver_inp = torch.cat((target, distractors), 0)
            # Shuffle target/distractor order
            indexes = self.random_state.permutation(receiver_inp.size()[0])
            #indexes = torch.randperm(receiver_inp.size()[0])
            shuffled_inp = receiver_inp[indexes]
            inputs.append(shuffled_inp)
            label = indexes.argmin() # target was on position 0 by construction
            labels.append(label)

        self.batches_generated += 1
        return torch.cat(batch_targets).float(), torch.LongTensor(labels), torch.stack(inputs,0)


class ColorIterator:
    def __init__(self, n_distractor, n_batches_per_epoch, batch_size, distance_matrix, min_value, data, train=True, seed=None):
        self.seed = seed
        self.n_batches_per_epoch = n_batches_per_epoch
        self.n_distractor = n_distractor
        self.batch_size = batch_size
        self.distance_matrix = distance_matrix
        self.data = data
        self.min_value = min_value
        self.train = train

    def __iter__(self):
        if self.seed is None:
            seed = np.random.randint(0, 2 ** 32)
        else:
            seed = self.seed

        return _ColorIterator(n_distractor=self.n_distractor, n_batches_per_epoch=self.n_batches_per_epoch, \
                                batch_size=self.batch_size, distance_matrix=self.distance_matrix, min_value=self.min_value, \
                                train=self.train, data=self.data, seed=self.seed)


if __name__ == '__main__':
    d = ColorData()
    print(build_distance_matrix(d).max())
