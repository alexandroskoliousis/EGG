# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F
import torch.distributions
import egg.core as core
import argparse
import numpy as np
from sklearn.metrics import mutual_info_score

import egg.core as core
from egg.zoo.color_naming.data import ColorData, build_distance_matrix
from egg.zoo.color_naming.archs import Sender, Receiver


N_COLOR_IDS = 330

def get_params(params):
    parser = argparse.ArgumentParser()
    parser.add_argument('--sender_hidden', type=int, default=10,
                        help='Size of the hidden layer of Sender (default: 10)')
    parser.add_argument('--receiver_hidden', type=int, default=10,
                        help='Size of the hidden layer of Receiver (default: 10)')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help="GS temperature for the sender (default: 1.0)")
    parser.add_argument('--scaler', type=int, default=100)
    parser.add_argument('--mode', type=str, choices=['gs', 'rf'], default='rf')
    parser.add_argument('--sender_entropy_coeff', type=float, default=5e-2)

    args = core.init(arg_parser=parser, params=params)
    return args

def compute_Entropy(variables):
    dico = {}
    for variable in variables:
        m = variable.argmax().detach().item()
        if m in dico.keys():
            dico[m] += 1.
        else:
            dico[m] = 1.
    #freq_dico = {}
    entropy = 0.
    for key, value in dico.items():
        tmp = value/sum([*dico.values()])
        #freq_dico[key] = tmp
        entropy += -tmp*np.log(tmp)
    return entropy

def compute_MI(variables1, variables2):
    return mutual_info_score(variables1, variables2)


class CylinderL1Loss(nn.Module):
    def __init__(self, distance_matrix):
        super().__init__()
        self.distance_matrix = distance_matrix

    def forward(self, sender_input, _message, _receiver_input, receiver_output, _labels):
        receiver_output = receiver_output.softmax(dim=-1)
        color_ids = sender_input[:, 0]
        distances = self.distance_matrix[color_ids].unsqueeze(1)
        expectation = torch.bmm(distances, receiver_output.unsqueeze(-1)).squeeze()
        acc = (receiver_output.argmax(dim=1)== color_ids).float()
        # Compute H(messages) and I(messages, inputs) and I(messages, outputs)
        H_m = compute_Entropy(_message)
        messages = []
        for m in _message:
            messages.append(m.argmax().detach().item())
        I_m_inp = compute_MI(messages, color_ids.detach().tolist())
        I_m_out = compute_MI(messages, receiver_output.detach().argmax(dim=1).tolist())

        return expectation, {'acc': acc, 'Hm': H_m, 'I_m_inp': I_m_inp, 'I_m_out': I_m_out}

def cross_entropy(sender_input, _message, _receiver_input, receiver_output, _labels):
    receiver_output = receiver_output.squeeze(1)
    sender_input = sender_input[:, 0]
    loss = F.cross_entropy(receiver_output, sender_input, reduction='none')
    acc = (receiver_output.argmax(dim=-1) == sender_input).float()
    return loss, {'acc': acc}

def dump(game, loader, device, gs_mode='gs'):
    #game.eval()
    with torch.no_grad():
        dataset = [x for x in loader][0][0].to(device)
        messages = game.sender(dataset)
        receiver_outputs = game.receiver(messages, None)

        unif_acc = 0.0
        for (input_sender,message, output) in zip(dataset, messages, receiver_outputs):
            inp = input_sender[0]
            out = output.argmax(dim=0)
            acc = inp==out
            unif_acc+=acc.item()
            print(f'input: {inp.item()} -> message: {message.argmax().item()} -> output: {out.item()}', flush=True)
        print(f'acc={unif_acc/dataset.size(0)}')

def main(params):
    opts = get_params(params)
    device = opts.device

    train_loader = torch.utils.data.DataLoader(
        ColorData(scaler=opts.scaler), batch_size=opts.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        ColorData(), batch_size=N_COLOR_IDS, shuffle=False)

    distance_matrix = build_distance_matrix(test_loader.dataset)
    distance_matrix = torch.from_numpy(distance_matrix).cuda().float()

    # initialize the agents and the game
    sender = Sender(N_COLOR_IDS, opts.vocab_size)  # the "data" transform part of an agent
    receiver = Receiver()
    receiver = core.SymbolReceiverWrapper(receiver, vocab_size=opts.vocab_size, agent_input_size=N_COLOR_IDS)

    loss = CylinderL1Loss(distance_matrix)

    if opts.mode == 'gs':
        sender = core.GumbelSoftmaxWrapper(sender, temperature=opts.temperature)
        game = core.SymbolGameGS(sender, receiver, loss)
    else:
        sender = core.ReinforceWrapper(sender)
        receiver = core.ReinforceDeterministicWrapper(receiver)
        game = core.SymbolGameReinforce(sender, receiver, loss, opts.sender_entropy_coeff)


    optimizer = core.build_optimizer(game.parameters())

    callbacks = [
        core.ConsoleLogger(print_train_loss=False, as_json=True),
    ]

    # initialize and launch the trainer
    trainer = core.Trainer(game=game, optimizer=optimizer, train_data=train_loader, validation_data=test_loader, callbacks=callbacks)
    trainer.train(n_epochs=opts.n_epochs)

    dump(game, test_loader, device)

    core.close()


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
