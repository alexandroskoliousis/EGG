# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
import torch
from torch.distributions import Categorical
from torch.nn import functional as F


class Receiver(nn.Module):
    def __init__(self, n_features, n_hidden):
        super(Receiver, self).__init__()
        self.fc1 = nn.Linear(n_hidden, n_features)

    def forward(self, x, _input):
        x = self.fc1(x)
        return x.sigmoid()

class CompoReceiver(nn.Module):
    def __init__(self, n_features, n_hidden):
    #def __init__(self, n_features, n_hidden, partition):
        super(CompoReceiver, self).__init__()
        self.fc1 = nn.Linear(n_hidden, n_features)
        #self.partition = partition

    def forward(self, x, _input):
        x = self.fc1(x)
        """
        start = 0
        outputs = []
        for p in self.partition:
            outputs.append(F.softmax(x[:, start:(start+p)], 1))
            start += p
        return torch.cat(outputs,1)
        """
        return x

class ReinforcedReceiver(nn.Module):
    def __init__(self, n_features, n_hidden, partition):
        super(ReinforcedReceiver, self).__init__()
        self.fc1 = nn.Linear(n_hidden, n_features)
        self.partition = partition

    def forward(self, x, _input):
        x = self.fc1(x)
        start = 0
        samples = []
        log_probs = []
        entropies = []

        for p in self.partition:
            probs = F.softmax(x[:, start:(start+p)], 1)
            distr = Categorical(probs=probs)
            entropy = distr.entropy()
            if self.training:
                sample = distr.sample()
            else:
                sample = probs.argmax(1)
                #sample = torch.zeros(probs.shape).to(probs.device).scatter(1, tmp_tensor.unsqueeze (1), 1.0).float()
            log_prob = distr.log_prob(sample)
            samples.append(sample)
            log_probs.append(log_prob)
            entropies.append(entropy)
            start += p

        return torch.cat([x.unsqueeze(0) for x in samples],0), torch.cat([x.unsqueeze(1) for x in log_probs],1).mean(1), torch.cat([x.unsqueeze(1) for x in entropies],1).mean(1)


class Sender(nn.Module):
    def __init__(self, n_hidden, n_features):
        super(Sender, self).__init__()
        self.fc1 = nn.Linear(n_features, n_hidden)

    def forward(self, x):
        x = self.fc1(x)
        return x
