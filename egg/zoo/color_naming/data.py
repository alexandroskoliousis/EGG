# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

import pathlib
import numpy as np
import torch
import torch.utils.data as data

def findposition(value, alist):
    if value==alist[0]:
        return(0)

    for i, element in enumerate(alist):
        if value > element and value <= alist[i+1]:
            return(i+1)

def build_distance_matrix(dataset):
    n = len(dataset.data)
    distance_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i):
            color_i, color_j = dataset.data[i], dataset.data[j]
            x_i, y_i, z_i = color_i[1], color_i[2], color_i[3]
            x_j, y_j, z_j = color_j[1], color_j[2], color_j[3]

            dist_x = np.abs(x_i - x_j)
            dist_y = np.abs(y_i - y_j)
            dist_z = np.abs(z_i - z_j)

            dist = dist_x + dist_y + dist_z

            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist
    return distance_matrix

def build_distance_matrix_old(dataset):
    n = len(dataset.data)
    distance_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i):
            color_i, color_j = dataset.data[i], dataset.data[j]
            x_i, y_i = color_i[1], color_i[2]
            x_j, y_j = color_j[1], color_j[2]

            dist_x = np.abs(x_i - x_j)
            dist_y = min(np.abs(y_i - y_j), np.abs(y_i - y_j + 40), np.abs(y_i - y_j - 40))
            dist = dist_x + dist_y

            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist

    return distance_matrix


def clipped_distance_matrix(dataset, division):
    distance_matrix = build_distance_matrix(dataset)
    distance_matrix = distance_matrix/np.max(distance_matrix)

    n = len(dataset.data)
    clipped_matrix = np.zeros((n, n))
    interval = [k/division for k in range(int(division)+1)]
    for i in range(n):
        for j in range(i):
            clipped_matrix[i,j] = findposition(distance_matrix[i,j], interval)/division
            clipped_matrix[j,i] = clipped_matrix[i,j]

    return clipped_matrix



class ColorData:
    def __init__(self, chip_file=None, scaler=1):
        self.scaler = scaler

        if chip_file is None:
            chip_file = pathlib.Path(__file__).parent / 'data/lab_coor.txt'
        else:
            chip_file = pathlib.Path(chip_file)

        if not chip_file.exists():
            raise FileNotFoundError(f'Cannot find chip file {chip_file}')

        data = np.loadtxt(chip_file, dtype=str, delimiter='\t')
        self.data = [torch.tensor([int(index)-1, float(x_value), float(y_value), float(z_value)]) for (index, x_value, y_value, z_value) \
                                            in zip(data[:,0], data[:,6], data[:,7], data[:,8])]

    def __len__(self):
        return self.scaler * len(self.data)

    def __getitem__(self, i):
        i = i % len(self.data)
        return self.data[i], torch.zeros(1)

class ColorData_old:
    def __init__(self, chip_file=None, scaler=1):
        self.scaler = scaler

        if chip_file is None:
            chip_file = pathlib.Path(__file__).parent / 'data/chip.txt'
        else:
            chip_file = pathlib.Path(chip_file)

        if not chip_file.exists():
            raise FileNotFoundError(f'Cannot find chip file {chip_file}')

        data = np.loadtxt(chip_file, dtype=str, delimiter='\t')
        # the first row is row id, the last is the concatenation of
        # the 2nd and the 3rd, don't need those
        data = data[:, 1:3]

        x = [ord(v) - ord('A') for v in data[:, 0]]
        y = [int(v) for v in data[:, 1]]

        self.data = [torch.tensor([i, x_value, y_value]).long() for i, (x_value, y_value) in enumerate(zip(x, y))]

    def __len__(self):
        return self.scaler * len(self.data)

    def __getitem__(self, i):
        i = i % len(self.data)
        return self.data[i], torch.zeros(1)

if __name__ == '__main__':
    #for i in data.DataLoader(ColorData(), batch_size=8):
    #    print(i)

    d = ColorData()

    print(build_distance_matrix(d).max())
