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
from egg.core import EarlyStopperLoss,EarlyStopperAccuracy
from egg.zoo.color_signaling.data import ColorData, ColorIterator, ClusterIterator, build_distance_matrix, Color_2D_Data, ColorData_converted
from colormath.color_objects import LabColor, sRGBColor, HSLColor

from egg.zoo.color_signaling.archs import Sender, Receiver

import numpy as np
import matplotlib.pyplot as plt
import ib_naming_model
from tools import *
from figures import mode_map

N_COLOR_IDS = 330

def get_params(params):
    parser = argparse.ArgumentParser()
    parser.add_argument('--temperature', type=float, default=1.0,
                        help="GS temperature for the sender (default: 1.0)")
    parser.add_argument('--mode', type=str, choices=['gs', 'rf'], default='gs')
    parser.add_argument('--sender_entropy_coeff', type=float, default=5e-2)
    parser.add_argument('--early_stopping_delta', type=float, default=1e-10,
                        help="Early stopping delta on loss (default: 1e-5)")
    parser.add_argument('--early_stopping_patience', type=int, default=100,
                        help="Early stopping patience on loss (default: 10)")
    parser.add_argument('--batches_per_epoch', type=int, default=100,
                        help='Number of batches per epoch (default: 100)')
    parser.add_argument('--n_distractor', type=int, default=1)

    parser.add_argument('--receiver_hidden', type=int, default=5,
                        help='Size of the hidden layer of Receiver (default: 5)')
    parser.add_argument('--receiver_num_hidden', type=int, default=1,
                        help='Number of the hidden layer of Receiver (default: 1)')

    parser.add_argument('--sender_hidden', type=int, default=1000,
                        help='Size of the hidden layer of Receiver (default: 1000)')
    parser.add_argument('--sender_num_hidden', type=int, default=3,
                        help='Number of the hidden layer of Receiver (default: 3)')

    parser.add_argument('--early_stopping_thr', type=float, default=0.9999,
                        help="Early stopping threshold on accuracy (default: 0.9999)")
    parser.add_argument('--input_id', type=int, default=3,
                        help="Give IDs as input, if 3 inputs are the CILAB coordinate, if 2 we are in the 2D space otherwise (if 1) we have only indexes (default: 3)")
    parser.add_argument('--input_space', type=str, default='cielab',
                        help="could be cielab or rgb", choices=['cielab', 'rgb', 'hsl', 'index'])

    parser.add_argument('--target_dst', type=str, choices=['uniform', 'SW'], default='uniform')
    parser.add_argument('--distractor_dst', type=str, choices=['uniform', 'SW'], default='uniform')
    parser.add_argument('--percentile', type=float, default=50.0,
                        help="If 0 there is no minimum target/distractor distance applied.")





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

def findposition(value, alist):
    if value>=alist[-1]:
        return len(alist)

    for i, element in enumerate(alist):
        if value >= element and value < alist[i+1]:
            return i

def cross_entropy(_sender_input,  _message, _receiver_input, receiver_output, _labels):
    acc = (receiver_output.argmax(dim=1) == _labels).detach().float()
    loss = F.cross_entropy(receiver_output, _labels, reduction="none")

    # The following works only with GS
    H_m = compute_Entropy(_message)
    messages = []
    for m in _message:
        messages.append(m.argmax().detach().item())

    color_ids = _sender_input[:,0].long()
    I_m_inp = compute_MI(messages, color_ids.detach().tolist())
    I_m_out = compute_MI(messages, receiver_output.detach().argmax(dim=1).tolist())

    return loss, {'acc': acc, 'Hm': H_m, 'I_m_inp': I_m_inp, 'I_m_out': I_m_out}


def dump(game, test_data, device, gs):
    game.eval()
    dataset = [x for x in test_data][0]
    sender_inp = dataset[0].to(device)
    labels = dataset[1].to(device)
    receiver_inp = dataset[2].to(device)
    

    with torch.no_grad():
        messages = game.sender(sender_inp, debug=True)
        if not gs: messages = messages[0]
        receiver_outputs = game.receiver(messages, receiver_inp)
        if not gs: receiver_outputs = receiver_outputs[0]

    unif_acc = 0.0
    for (input_sender, message, output, label, input_receiver) in zip(sender_inp, messages, receiver_outputs, labels, receiver_inp):
        out = output.argmax(dim=0)
        acc = label==out
        unif_acc+=acc.item()
        s_inp = input_sender[0].long() # # ID
        r_inp = input_receiver[:,0].long() # # ID
        if gs: message = message.argmax()
        print(f'input sender: {s_inp.item():3d} | input receiver: {[r.item() for r in r_inp]} -> message: {message.item()} -> output: {out.item()}', flush=True)
    print(f'acc={unif_acc/sender_inp.size(0)}')


def main(params):
    opts = get_params(params)
    device = opts.device
    if opts.input_id not in [1,2,3]:
        print('wrong parameter input_id')
        exit()

    data = ColorData()
    distance_matrix = build_distance_matrix(data)

    if opts.input_id == 2:
        data = Color_2D_Data()

    if opts.input_space=='rgb':
        data = ColorData_converted(new_space=sRGBColor)
    elif opts.input_space=='hsl':
        data = ColorData_converted(new_space=HSLColor)

    if opts.percentile>0:
        min_value = np.percentile(distance_matrix, opts.percentile)
    else:
        min_value = None

    train_loader = ColorIterator(n_distractor=opts.n_distractor, n_batches_per_epoch=opts.batches_per_epoch, seed=opts.random_seed, \
                                    batch_size=opts.batch_size, distance_matrix=distance_matrix, min_value=min_value, \
                                    data=data, proba_target=opts.target_dst, proba_distractor=opts.distractor_dst)

    #train_loader = ClusterIterator(n_distractor=opts.n_distractor, n_batches_per_epoch=opts.batches_per_epoch, seed=opts.random_seed, \
    #                                batch_size=opts.batch_size, \
    #                                cluster='/private/home/rchaabouni/EGG_public/egg/zoo/color_signaling/data/languages/darklight_argmaxing.txt', data=data)

    # Same validation across runs by fixing the seed
    val_loader = ColorIterator(n_distractor=opts.n_distractor, n_batches_per_epoch=1, train=False, seed=1, \
                                    batch_size=len(data), distance_matrix=distance_matrix, min_value=min_value if min_value else np.percentile(distance_matrix, 50), \
                                    data=data)

    #val_loader = ClusterIterator(n_distractor=opts.n_distractor, n_batches_per_epoch=1, train=False, seed=1, \
    #                                batch_size=len(data), \
    #                                cluster='/private/home/rchaabouni/EGG_public/egg/zoo/color_signaling/data/languages/darklight_argmaxing.txt', data=data)



    # initialize the agents and the game
    #sender = Sender(opts.vocab_size, n_colors=N_COLOR_IDS, ids=opts.input_id)  # the "data" transform part of an agent
    #receiver = Receiver(opts.receiver_hidden, n_colors=N_COLOR_IDS, ids=opts.input_id)
    sender1 = Sender(opts.vocab_size, num_layers=opts.sender_num_hidden, hidden_size=opts.sender_hidden, n_colors=N_COLOR_IDS, ids=opts.input_id)  # the "data" transform part of an agent
    receiver = Receiver(opts.receiver_hidden, num_layers=opts.receiver_num_hidden, n_colors=N_COLOR_IDS, ids=opts.input_id)


    receiver = core.SymbolReceiverWrapper(receiver, vocab_size=opts.vocab_size, agent_input_size=opts.receiver_hidden)


    if opts.mode == 'gs':
        sender = core.GumbelSoftmaxWrapper(sender1, temperature=opts.temperature)
        game = core.SymbolGameGS(sender, receiver, cross_entropy)
    else:
        sender = core.ReinforceWrapper(sender1)
        receiver = core.ReinforceDeterministicWrapper(receiver)
        game = core.SymbolGameReinforce(sender, receiver, cross_entropy, opts.sender_entropy_coeff)


    optimizer = core.build_optimizer(game.parameters())

    callbacks = [
        EarlyStopperAccuracy(opts.early_stopping_thr),
        EarlyStopperLoss(delta=opts.early_stopping_delta, patience=opts.early_stopping_patience),
        core.ConsoleLogger(print_train_loss=True, as_json=True),
    ]


    # initialize and launch the trainer
    trainer = core.Trainer(game=game, optimizer=optimizer, train_data=train_loader, validation_data=val_loader, callbacks=callbacks)
    trainer.train(n_epochs=opts.n_epochs)

    mydataset = [x for x in val_loader][0]
    sender_inp = mydataset[0].to(device)
    print("[DBG] Sender input tensor has shape", sender_inp.shape)
    with torch.no_grad():
        messages = sender1(sender_inp, debug=False)
        print('In main() method...')
        print("[DBG] Messages tensor has shape", messages.shape)
        print(messages)
        probs = torch.exp(messages)
        # psum = probs.sum(dim=-1)
        # print(psum.shape, psum)
    
    dump(game, val_loader, device, gs=(opts.mode=='gs'))
    
    core.close()

    # IB curve...

    # load model
    model = ib_naming_model.load_model()
    curve = model.IB_curve

    print(model.fit(probs.numpy())[:-1])

    epsilon, gnid, bl, qW_M_fit = model.fit(probs.numpy())

    print(f'epsilon = {epsilon}')
    print(f'gnid = {gnid}')
    print(f'bl = {bl}')

    complexity = model.complexity(qW_M_fit)
    accuracy = model.accuracy(qW_M_fit)

    print(f'accuracy = {accuracy}')
    print(f'complexity = {complexity}')

    complexity1 = model.complexity(probs)
    accuracy1 = model.accuracy(probs)

    print(f'accuracy1 = {accuracy1}')
    print(f'complexity1 = {complexity1}')

    # indx = 500
    # qW_M = model.qW_M[indx]
    # print(type(qW_M), qW_M.shape)

    # theoretical bound
    plt.figure()
    plt.plot(curve[0], curve[1])

    plt.plot(complexity, accuracy, 'o', markersize=5)
    plt.plot(complexity1, accuracy1, 'o', markersize=5)
    
    plt.xlabel('complexity, $I(M;W)$')
    plt.ylabel('accuracy, $I(W;U)$')
    
    plt.xlim([0, H(model.pM)])
    plt.ylim([0, model.I_MU + 0.1])

    # generate fake (random) data and fit an optimal IB system
    # pW_M_fake = np.random.rand(330, 10)
    # pW_M_fake /= pW_M_fake.sum(axis=1)[:, None]
    

    # let's plot a mode map!
    
    plt.figure(figsize=(6.4, 2.5))
    model.mode_map(qW_M_fit)
    plt.title('Optimal IB system for $\\beta = %.3f$' % bl)
    plt.tight_layout()
    plt.show()
    
    with open("Training_params.txt", "a") as o:
        o.write("\n")
        o.write("complexity: ")
        o.write(model.complexity(qW_M).__str__())
        o.write("\n")
        o.write("accuracy: ")
        o.write(model.accuracy(qW_M).__str__())
        o.write("\n")


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
