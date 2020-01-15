# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree
import copy
import torch
import torch.nn as nn
import egg.core as core
import torch.nn.functional as F



class Sender2(nn.Module):
    def __init__(self, vocab_size):
        super(Sender, self).__init__()
        self.fc1 = nn.Linear(3, 500) # LAB is a 3D space
        self.fc2 = nn.Linear(500, vocab_size)

    def forward(self, x):
        x = x[:, 1:] # not taking the ID but only the LAB system
        x = self.fc1(x)
        x = F.leaky_relu(x)
        x = self.fc2(x)
        return x.log_softmax(dim=-1)


class Receiver2(nn.Module):
    def __init__(self, hidden):
        super(Receiver, self).__init__()
        self.fc = nn.Linear(3, hidden)


    def forward(self, x, _input):
        _input = _input[:,:,1:]  # not taking the ID but only the LAB system
        embedded_input = self.fc(_input)
        energies = torch.matmul(embedded_input, torch.unsqueeze(x, dim=-1))
        return energies.squeeze()

class Sender(nn.Module):
    def __init__(self, vocab_size, n_colors=330, ids=False):
        super(Sender, self).__init__()
        if ids:
            self.emb = nn.Embedding(n_colors, 3)
        self.ids = ids
        """
        # V1
        self.fc1 = nn.Linear(3, 500) # LAB is a 3D space (originally 500)
        self.fc2 = nn.Linear(500, vocab_size)
        # V2
        self.fc1 = nn.Linear(3, 1000) # LAB is a 3D space
        self.fc2 = nn.Linear(1000, vocab_size)
        """
        # V3
        self.fc1 = nn.Linear(3, 1000) # LAB is a 3D space (originally 500)
        self.fc2 = nn.Linear(1000, 1000) ## New
        self.fc3 = nn.Linear(1000, vocab_size)


    def forward(self, x):
        if self.ids:
            x = x[:, 0:1].long() # only color-id
            x = self.emb(x)
        else:
            x = x[:, 1:] # not taking the ID but only the LAB system
        x = self.fc1(x)
        x = F.leaky_relu(x)
        x = self.fc2(x)
        # V3
        x = F.leaky_relu(x) # NEW
        x = self.fc3(x)
        return x.log_softmax(dim=-1)


class Receiver(nn.Module):
    def __init__(self, hidden, n_colors=330, ids=False):
        super(Receiver, self).__init__()
        if ids:
            self.emb = nn.Embedding(n_colors, hidden)
        else:
            self.fc = nn.Linear(3, hidden)
        self.ids = ids


    def forward(self, x, _input):
        if self.ids:
            _input = _input[:,:,0:1].long() # only color-id
            embedded_input = self.emb(_input)
        else:
            _input = _input[:,:,1:]  # not taking the ID but only the LAB system
            embedded_input = self.fc(_input)

        energies = torch.matmul(embedded_input, torch.unsqueeze(x, dim=-1))
        return energies.squeeze()
