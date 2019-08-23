# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
import torch
from torch.distributions import Bernoulli


class Receiver(nn.Module):
    def __init__(self, n_features, n_hidden):
        super(Receiver, self).__init__()
        self.fc1 = nn.Linear(n_hidden, n_features)

    def forward(self, x, _input):
        x = self.fc1(x)
        return x.sigmoid()

class CompoReceiver(nn.Module):
    def __init__(self, n_features, n_hidden, partition):
        super(CompoReceiver, self).__init__()
        self.fc1 = nn.Linear(n_hidden, n_features)
        self.partition = partition

    def forward(self, x, _input):
        x = self.fc1(x)
        start = 0
        outputs = []
        for p in self.partition:
            outputs.append(x[:, start:(start+p)].softmax())
            start += p
        return torch.cat(outputs,1)


class Sender(nn.Module):
    def __init__(self, n_hidden, n_features):
        super(Sender, self).__init__()
        self.fc1 = nn.Linear(n_features, n_hidden)

    def forward(self, x):
        x = self.fc1(x)
        return x
