# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree
import torch
import torch.nn as nn
import egg.core as core
import torch.nn.functional as F


class Sender2(nn.Module):
    def __init__(self, n_colors, vocab_size):
        super(Sender, self).__init__()
        self.emb = nn.Embedding(n_colors, 100)
        self.fc1 = nn.Linear(100, 1000)
        self.fc2 = nn.Linear(1000, vocab_size)
        #self.emb = nn.Embedding(n_colors, vocab_size)

    def forward(self, x):
        x = x[:, 0:1].long() # only color-id at the moment
        x = self.emb(x)
        x = F.leaky_relu(x)
        #x = torch.sigmoid(x)
        x = self.fc1(x)
        x = F.leaky_relu(x)
        x = self.fc2(x)
        return x.log_softmax(dim=-1)

class Sender3(nn.Module):
    def __init__(self, n_colors, vocab_size):
        super(Sender, self).__init__()
        #self.emb = nn.Embedding(n_colors, 100)
        #self.fc = nn.Linear(100, vocab_size)
        self.emb = nn.Embedding(n_colors, vocab_size)

    def forward(self, x):
        x = x[:, 0:1].long() # only color-id at the moment
        x = self.emb(x)
        #x = F.leaky_relu(x)
        #x = torch.sigmoid(x)
        #x = self.fc(x)
        return x.log_softmax(dim=-1)

class Sender(nn.Module):
    def __init__(self, n_colors, vocab_size):
        super(Sender, self).__init__()
        self.emb = nn.Embedding(n_colors, 100)
        self.fc = nn.Linear(100, vocab_size)
        #self.emb = nn.Embedding(n_colors, vocab_size)

    def forward(self, x):
        x = x[:, 0:1].long() # only color-id at the moment
        x = self.emb(x)
        x = F.leaky_relu(x)
        #x = torch.sigmoid(x)
        x = self.fc(x)
        return x.log_softmax(dim=-1)


class Receiver(nn.Module):
    def __init__(self, n_colors):
        super(Receiver, self).__init__()

    def forward(self, x, _input):
        return x.squeeze(1)

class Receiver2(nn.Module):
    def __init__(self, n_colors):
        super(Receiver, self).__init__()
        self.fc1 = nn.Linear(n_colors, 100)
        self.fc2 = nn.Linear(100, n_colors)


    def forward(self, x, _input):
        x = F.leaky_relu(x)
        x = self.fc1(x)
        x = F.leaky_relu(x)
        x = self.fc2(x)
        return x.squeeze(1)
