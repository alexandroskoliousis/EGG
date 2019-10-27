# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree
import torch
import torch.nn as nn
import egg.core as core
import torch.nn.functional as F



class Sender(nn.Module):
    def __init__(self, vocab_size):
        super(Sender, self).__init__()
        self.fc = nn.Linear(3, vocab_size) # LAB is a 3D space

    def forward(self, x):
        x = x[:, 1:] # not taking the ID but only the LAB system
        x = self.fc(x)
        return x.log_softmax(dim=-1)


class Receiver(nn.Module):
    def __init__(self, hidden):
        super(Receiver, self).__init__()
        self.fc = nn.Linear(3, hidden)
        #self.fc2 = nn.Linear(hidden, 3)
        #self.fc_message = nn.Linear(hidden, 3)


    def forward(self, x, _input):
        #x = self.fc_message(x)
        _input = _input[:,:,1:]  # not taking the ID but only the LAB system
        embedded_input = self.fc(_input)
        #embedded_input = _input
        energies = torch.matmul(embedded_input, torch.unsqueeze(x, dim=-1))
        return energies.squeeze()
