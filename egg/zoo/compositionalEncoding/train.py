# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import argparse
import numpy as np
import random
import itertools
import torch.utils.data
import torch.nn.functional as F

import egg.core as core
from egg.core import EarlyStopperAccuracy
from egg.zoo.compositionalEncoding.features import SimpleLoader, CompositionalLoader, \
                                                    ConcatLoader, UniformLoader, Split_Train_Test
from egg.zoo.compositionalEncoding.archs import Sender, Receiver


def get_params(params):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dimensions', type=str, default='[8]',
                        help='Dimensionality of the "concept" space (default: 10)')
    parser.add_argument('--batches_per_epoch', type=int, default=1000,
                        help='Number of batches per epoch (default: 1000)')
    parser.add_argument('--dim_dataset', type=int, default=10240,
                        help='Dim of constructing the data (default: 10240)')
    parser.add_argument('--force_eos', type=int, default=0,
                        help='Force EOS at the end of the messages (default: 0)')

    parser.add_argument('--sender_hidden', type=int, default=10,
                        help='Size of the hidden layer of Sender (default: 10)')
    parser.add_argument('--receiver_hidden', type=int, default=10,
                        help='Size of the hidden layer of Receiver (default: 10)')
    parser.add_argument('--receiver_num_layers', type=int, default=1,
                        help='Number hidden layers of receiver. Only in reinforce (default: 1)')
    parser.add_argument('--sender_num_layers', type=int, default=1,
                        help='Number hidden layers of receiver. Only in reinforce (default: 1)')
    parser.add_argument('--receiver_num_heads', type=int, default=8,
                        help='Number of attention heads for Transformer Receiver (default: 8)')
    parser.add_argument('--sender_num_heads', type=int, default=8,
                        help='Number of self-attention heads for Transformer Sender (default: 8)')
    parser.add_argument('--sender_embedding', type=int, default=10,
                        help='Dimensionality of the embedding hidden layer for Sender (default: 10)')
    parser.add_argument('--receiver_embedding', type=int, default=10,
                        help='Dimensionality of the embedding hidden layer for Receiver (default: 10)')

    parser.add_argument('--causal_sender', default=False, action='store_true')
    parser.add_argument('--causal_receiver', default=False, action='store_true')

    parser.add_argument('--sender_generate_style', type=str, default='in-place', choices=['standard', 'in-place'],
                        help='How the next symbol is generated within the TransformerDecoder (default: in-place)')

    parser.add_argument('--sender_cell', type=str, default='rnn',
                        help='Type of the cell used for Sender {rnn, gru, lstm, transformer} (default: rnn)')
    parser.add_argument('--receiver_cell', type=str, default='rnn',
                        help='Type of the model used for Receiver {rnn, gru, lstm, transformer} (default: rnn)')

    parser.add_argument('--sender_entropy_coeff', type=float, default=1e-1,
                        help='The entropy regularisation coefficient for Sender (default: 1e-1)')
    parser.add_argument('--receiver_entropy_coeff', type=float, default=1e-1,
                        help='The entropy regularisation coefficient for Receiver (default: 1e-1)')

    parser.add_argument('--probs', type=str, default='uniform',
                        help="Prior distribution over the concepts (default: uniform)")
    parser.add_argument('--complex_gram', type=str, default='surface',
                        help="how to build distribution over the complex concepts (default: surface)")
    parser.add_argument('--length_cost', type=float, default=0.0,
                        help="Penalty for the message length, each symbol would before <EOS> would be "
                             "penalized by this cost (default: 0.0)")
    parser.add_argument('--name', type=str, default='model',
                        help="Name for your checkpoint (default: model)")
    parser.add_argument('--early_stopping_thr', type=float, default=0.9999,
                        help="Early stopping threshold on accuracy (default: 0.9999)")

    args = core.init(parser, params)

    return args


def loss(sender_input, _message, _receiver_input, receiver_output, _labels):
    acc = (receiver_output.argmax(dim=1) == sender_input.argmax(dim=1)).detach().float()
    loss = F.cross_entropy(receiver_output, sender_input.argmax(dim=1), reduction="none")
    return loss, {'acc': acc}


def dump(game, n_features, device, gs_mode):
    # tiny "dataset"
    dataset = [[torch.eye(n_features).to(device), None]]

    sender_inputs, messages, receiver_inputs, receiver_outputs, _ = \
        core.dump_sender_receiver(game, dataset, gs=gs_mode, device=device, variable_length=True)

    unif_acc = 0.
    powerlaw_acc = 0.
    powerlaw_probs = 1 / np.arange(1, n_features+1, dtype=np.float32)
    powerlaw_probs /= powerlaw_probs.sum()

    for sender_input, message, receiver_output in zip(sender_inputs, messages, receiver_outputs):
        input_symbol = sender_input.argmax()
        output_symbol = receiver_output.argmax()
        acc = (input_symbol == output_symbol).float().item()

        unif_acc += acc
        powerlaw_acc += powerlaw_probs[input_symbol] * acc
        print(f'input: {input_symbol.item()} -> message: {",".join([str(x.item()) for x in message])} -> output: {output_symbol.item()}', flush=True)

    unif_acc /= n_features

    print(f'Mean accuracy wrt uniform distribution is {unif_acc}')
    print(f'Mean accuracy wrt powerlaw distribution is {powerlaw_acc}')
    print(json.dumps({'powerlaw': powerlaw_acc, 'unif': unif_acc}))


def main(params):
    opts = get_params(params)
    print(opts, flush=True)
    device = opts.device

    force_eos = opts.force_eos == 1
    dimensions = eval(opts.dimensions)

    probs = []
    if opts.probs == 'uniform':
        for dim in dimensions:
            probs_nonnorm = np.ones(dim)
            probs.append(probs_nonnorm/probs_nonnorm.sum())
    elif opts.probs == 'powerlaw':
        for dim in dimensions:
            probs_nonnorm = 1 / np.arange(1, dim+1, dtype=np.float32)
            probs.append(probs_nonnorm/probs_nonnorm.sum())
    else:
        print('Not supported feature')
        sys.exit("Error message")

    #print('the probs are: ', probs, flush=True)
    if opts.complex_gram=='surface':
        nbr_complex = 1
        for dim in dimensions:
            nbr_complex *= dim
        if opts.probs == 'uniform':
            probs_nonnorm = np.ones(nbr_complex)
            complex_probs = probs_nonnorm/probs_nonnorm.sum()
        else:
            probs_nonnorm = 1 / np.arange(1, nbr_complex+1, dtype=np.float32)
            complex_probs = probs_nonnorm/probs_nonnorm.sum()

    elif opts.complex_gram=='base':
        complex_probs_nonnorm = []
        combinations_prob = itertools.product(*probs)
        for p in combinations_prob:
            mult = 1
            for t_p in p:
                mult*=t_p
            complex_probs_nonnorm.append(mult)
        complex_probs = complex_probs_nonnorm/sum(complex_probs_nonnorm)

    else:
        print('Not supported complex_gram')
        sys.exit("Error message")

    (train, new_comp_proba), test = Split_Train_Test(dimensions, probs, complex_probs, ratio=0.9)
    comp_proba = new_comp_proba/sum(new_comp_proba)

    # Simple referents
    sl_loader = SimpleLoader(dimensions, opts.batches_per_epoch, int(round(opts.batch_size/(2.*len(dimensions)))), probs)
    # Complex referent
    cl_loader = CompositionalLoader(train, opts.batches_per_epoch, int(round(opts.batch_size/2.)), comp_proba)

    train_loader = ConcatLoader(sl_loader, cl_loader)
    validation_loader = UniformLoader(dimensions, torch.FloatTensor(train))

    if opts.sender_cell == 'transformer':
        sender = Sender(n_features=sum(dimensions), n_hidden=opts.sender_embedding)
        sender = core.TransformerSenderReinforce(agent=sender, vocab_size=opts.vocab_size,
                                                 embed_dim=opts.sender_embedding, max_len=opts.max_len,
                                                 num_layers=opts.sender_num_layers, num_heads=opts.sender_num_heads,
                                                 hidden_size=opts.sender_hidden,
                                                 force_eos=opts.force_eos,
                                                 generate_style=opts.sender_generate_style,
                                                 causal=opts.causal_sender)
    else:
        sender = Sender(n_features=sum(dimensions), n_hidden=opts.sender_hidden)

        sender = core.RnnSenderReinforce(sender,
                                   opts.vocab_size, opts.sender_embedding, opts.sender_hidden,
                                   cell=opts.sender_cell, max_len=opts.max_len, num_layers=opts.sender_num_layers,
                                   force_eos=force_eos)
    if opts.receiver_cell == 'transformer':
        receiver = Receiver(n_features=sum(dimensions), n_hidden=opts.receiver_embedding)
        receiver = core.TransformerReceiverDeterministic(receiver, opts.vocab_size, opts.max_len,
                                                         opts.receiver_embedding, opts.receiver_num_heads, opts.receiver_hidden,
                                                         opts.receiver_num_layers, causal=opts.causal_receiver)
    else:
        receiver = Receiver(n_features=sum(dimensions), n_hidden=opts.receiver_hidden)
        receiver = core.RnnReceiverDeterministic(receiver, opts.vocab_size, opts.receiver_embedding,
                                             opts.receiver_hidden, cell=opts.receiver_cell,
                                             num_layers=opts.receiver_num_layers)

    game = core.SenderReceiverRnnReinforce(sender, receiver, loss, sender_entropy_coeff=opts.sender_entropy_coeff,
                                           receiver_entropy_coeff=opts.receiver_entropy_coeff,
                                           length_cost=opts.length_cost)

    optimizer = core.build_optimizer(game.parameters())

    trainer = core.Trainer(game=game, optimizer=optimizer, train_data=train_loader,
                           validation_data=validation_loader, callbacks=[EarlyStopperAccuracy(opts.early_stopping_thr)], distribution=probs)

    #trainer.train(n_epochs=opts.n_epochs)
    if opts.checkpoint_dir:
        trainer.save_checkpoint(name=f'{opts.name}_vocab{opts.vocab_size}_rs{opts.random_seed}_lr{opts.lr}_shid{opts.sender_hidden}_rhid{opts.receiver_hidden}_sentr{opts.sender_entropy_coeff}_reg{opts.length_cost}_max_len{opts.max_len}')

    #dump(trainer.game, opts.n_features, device, False) TODO: redo it
    core.close()


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
