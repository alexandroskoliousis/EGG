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

class Sender1(nn.Module):
    def __init__(self, vocab_size, n_colors=330, ids=3):
        super(Sender, self).__init__()
        if ids==1:
            self.emb = nn.Embedding(n_colors, 3)
        elif ids==2:
            self.emb = nn.Linear(ids, 3)
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
        if self.ids<3:
            if self.ids==1:
                x = x[:, 0:1].long() # only color-id
            else:
                x = x[:, 1:] # not taking the ID
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


class Receiver1(nn.Module):
    def __init__(self, hidden, n_colors=330, ids=3):
        super(Receiver, self).__init__()
        if ids==1:
            self.emb = nn.Embedding(n_colors, hidden)
        else:
            self.fc = nn.Linear(ids, hidden)
        self.ids = ids


    def forward(self, x, _input):
        if self.ids==1:
            _input = _input[:,:,0:1].long() # only color-id
            embedded_input = self.emb(_input)
        else:
            _input = _input[:,:,1:].float()  # not taking the ID but only the LAB system
            embedded_input = self.fc(_input)

        energies = torch.matmul(embedded_input, torch.unsqueeze(x, dim=-1))
        return energies.squeeze()

class Sender(nn.Module):
    def __init__(self, vocab_size, num_layers=2, hidden_size=1000, n_colors=330, ids=3):
        super(Sender, self).__init__()
        if ids==1:
            self.emb = nn.Embedding(n_colors, 3)
        elif ids==2:
            self.emb = nn.Linear(ids, 3)
        self.ids = ids

        fcs = []
        fcs.append(nn.Linear(3, hidden_size))
        for _ in range(num_layers-1):
            fcs.append(nn.Linear(hidden_size, hidden_size))
        fcs.append(nn.Linear(hidden_size, vocab_size))
        self.fcs = nn.ModuleList(fcs)

    def forward(self, x, debug=False):
        if self.ids<3:
            if self.ids==1:
                x = x[:, 0:1].long() # only color-id
            else:
                x = x[:, 1:] # not taking the ID
            x = self.emb(x)
        else:
            x = x[:, 1:] # not taking the ID but only the LAB system
        for fc in self.fcs[:-1]:
            x = fc(x)
            x = F.leaky_relu(x)

        x = self.fcs[-1](x)

        # Modified by akoliousis for debugging purposes
        result = x.log_softmax(dim=-1)
        if debug:
            print("[DBG] Sender output tensor has shape", result.shape)
            print(result)
            # expected = x.softmax(dim=-1)
            # esum = expected.sum(dim=-1)
            # print(esum.shape, esum)
            actual = torch.exp(result)
            asum = torch.sum(actual, dim=-1)
            print(asum.shape, asum)
            # print(expected)
            # print(actual)
            # assert torch.allclose(expected, actual)
        return result


class Receiver(nn.Module):
    def __init__(self, hidden, num_layers=1, n_colors=330, ids=3):
        super(Receiver, self).__init__()
        fcs = []
        if ids==1:
            fcs.append(nn.Embedding(n_colors, hidden))
        else:
            fcs.append(nn.Linear(ids, hidden))

        for _ in range(num_layers-1):
            fcs.append(nn.Linear(hidden, hidden))

        self.fcs = nn.ModuleList(fcs)
        self.ids = ids


    def forward(self, x, _input):
        if self.ids==1:
            y = _input[:,:,0:1].long() # only color-id
        else:
            y = _input[:,:,1:].float()  # not taking the ID but only the LAB system
        for fc in self.fcs[:-1]:
            y = fc(y)
            y = F.leaky_relu(y)

        embedded_input = self.fcs[-1](y)

        energies = torch.matmul(embedded_input, torch.unsqueeze(x, dim=-1))
        return energies.squeeze()


class ContinousGame(nn.Module):
    def __init__(self, sender, receiver, loss):
        """
        :param sender: Sender agent. sender.forward() has to output log-probabilities over the vocabulary.
        :param receiver: Receiver agent. receiver.forward() has to accept two parameters: message and receiver_input.
        `message` is shaped as (batch_size, vocab_size).
        :param loss: Callable that outputs differentiable loss, takes the following parameters:
          * sender_input: input to Sender (comes from dataset)
          * message: message sent from Sender
          * receiver_input: input to Receiver from dataset
          * receiver_output: output of Receiver
          * labels: labels that come from dataset
        """
        super(ContinousGame, self).__init__()
        self.sender = sender
        self.receiver = receiver
        self.loss = loss

    def forward(self, sender_input, labels, receiver_input=None):
        message = self.sender(sender_input)
        receiver_output = self.receiver(message.softmax(dim=1), receiver_input)

        loss, rest_info = self.loss(sender_input, message, receiver_input, receiver_output, labels)
        for k, v in rest_info.items():
            if hasattr(v, 'mean'):
                rest_info[k] = v.mean().item()

        return loss.mean(), rest_info
