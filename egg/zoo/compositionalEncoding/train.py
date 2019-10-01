# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import argparse
import numpy as np
import random
import itertools
import os
import sys
import torch.utils.data
import torch.nn.functional as F

import egg.core as core
from egg.core import EarlyStopperAccuracy
from egg.zoo.compositionalEncoding.features import SimpleLoader, CompositionalLoader, \
                                                    ConcatLoader, UniformLoader
from egg.zoo.compositionalEncoding.archs import Sender, Receiver, CompoReceiver
from egg.core.callbacks import Callback, ConsoleLogger, CheckpointSaver


def get_params(params):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dimensions', type=str, default='[10, 10, 10]',
                        help='Dimensionality of the "concept" space (default: [10,10,10])')
    parser.add_argument('--dataset_path', type=str, default='/private/home/rchaabouni/EGG_public/egg/zoo/compositionalEncoding/datasets/',
                        help='Path to find the train/test dataset')
    parser.add_argument('--batches_per_epoch', type=int, default=1000,
                        help='Number of batches per epoch (default: 1000)')
    parser.add_argument('--force_eos', type=int, default=0,
                        help='Force EOS at the end of the messages (default: 0)')
    parser.add_argument('--exist_eos', type=int, default=1,
                        help='Is there a notion of eos in training (default: 0)')

    parser.add_argument('--sender_hidden', type=int, default=10,
                        help='Size of the hidden layer of Sender (default: 10)')
    parser.add_argument('--receiver_hidden', type=int, default=10,
                        help='Size of the hidden layer of Receiver (default: 10)')
    parser.add_argument('--receiver_num_layers', type=int, default=1,
                        help='Number hidden layers of receiver. Only in reinforce (default: 1)')
    parser.add_argument('--sender_num_layers', type=int, default=1,
                        help='Number hidden layers of receiver. Only in reinforce (default: 1)')
    parser.add_argument('--receiver_num_heads', type=int, default=1,
                        help='Number of attention heads for Transformer Receiver (default: 8)')
    parser.add_argument('--sender_num_heads', type=int, default=1,
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


def loss(sender_input, _message, _receiver_input, receiver_output, _labels, partition):
    acc = ((receiver_output > 0.5).long() == sender_input.long()).detach().all(dim=1).float().mean()
    loss = F.binary_cross_entropy(receiver_output, sender_input.float(), reduction="none").mean(dim=1)
    return loss, {'acc': acc}

def compoloss(sender_input, _message, _receiver_input, receiver_output, _labels, partition):
    accs = []
    losses = []
    start = 0

    for i, p in enumerate(partition):
        p_input = sender_input[:, start:(start+p)]
        p_output = receiver_output[:, start:(start+p)]
        accs.append((p_input.argmax(dim=1) == p_output.argmax(dim=1)).detach().float().unsqueeze(1))
        losses.append(F.cross_entropy(p_output, p_input.argmax(dim=1), reduction="none").unsqueeze(0))
        start += p

    acc = (torch.sum(torch.cat(accs,1),1)==len(partition)).detach().float().mean()
    loss = torch.cat(losses,0).mean(0)
    return loss, {'acc': acc}

def dump(game, partition, test, device, gs_mode):
    # tiny "dataset"
    if len(test)>0:
        dataset = [[torch.FloatTensor(test).to(device), None]]
        #toto = [[0,1,0,1,0,0], [0,0,1,1,0,0], [1,0,0,0,1,0], [1,0,0,0,0,1]]
        #train.extend(toto)
        #dataset = [[torch.FloatTensor(train).to(device), None]]
        sender_inputs, messages, receiver_inputs, receiver_outputs, _ = \
            core.dump_sender_receiver(game, dataset, gs=gs_mode, device=device, variable_length=True)

        unif_acc = 0.

        for sender_input, message, receiver_output in zip(sender_inputs, messages, receiver_outputs):
            start = 0
            boolean = []
            output_symbols = []
            input_symbols = []
            for p in partition:
                input_symbol = sender_input[start:(start+p)].argmax()
                input_symbols.append(input_symbol.item())
                output_symbol = receiver_output[start:(start+p)].argmax()
                output_symbols.append(output_symbol.item())
                boolean.append( (input_symbol==output_symbol).float().item() )
                start += p
            unif_acc += int(sum(boolean)==len(partition))

            print(f'input: {input_symbols} -> message: {",".join([str(x.item()) for x in message])} -> output: {output_symbols}', flush=True)
        unif_acc /= len(test)

        print(f'Mean accuracy wrt uniform distribution on test set is {unif_acc}')
        print(json.dumps({'unif': unif_acc}))

def main(params):
    opts = get_params(params)
    print(opts, flush=True)
    device = opts.device

    force_eos = opts.force_eos == 1
    exist_eos = opts.exist_eos == 1
    dimensions = eval(opts.dimensions)

    chars = ''
    for dim in dimensions:
        chars+=str(dim)+'_'
    if opts.probs == 'uniform':
        path = opts.dataset_path
        if not(os.path.exists(path)):
            print('create the right dataset or give the correct path')
            sys.exit("Error message")
    else:
        print('Not supported probs')
        sys.exit("Error message")

    dataset = torch.load(path)
    train = dataset['train']
    test = dataset['test']

    probs_nonnorm = np.ones(len(train))
    prob_uni = probs_nonnorm/probs_nonnorm.sum()

    # Complex referent
    train_loader = CompositionalLoader(train, opts.batches_per_epoch, opts.batch_size, prob_uni)
    validation_loader = UniformLoader(torch.FloatTensor(train))

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
        receiver = CompoReceiver(n_features=sum(dimensions), n_hidden=opts.receiver_embedding)
        receiver = core.TransformerReceiverDeterministic(receiver, opts.vocab_size, opts.max_len,
                                                         opts.receiver_embedding, opts.receiver_num_heads, opts.receiver_hidden,
                                                         opts.receiver_num_layers, causal=opts.causal_receiver)
    else:
        receiver = CompoReceiver(n_features=sum(dimensions), n_hidden=opts.receiver_hidden)
        receiver = core.RnnReceiverDeterministic(receiver, opts.vocab_size, opts.receiver_embedding,
                                             opts.receiver_hidden, cell=opts.receiver_cell,
                                             num_layers=opts.receiver_num_layers)

    game = core.SenderReceiverRnnReinforce(sender, receiver, compoloss, sender_entropy_coeff=opts.sender_entropy_coeff,
                                           receiver_entropy_coeff=opts.receiver_entropy_coeff,
                                           length_cost=opts.length_cost, dimensions=dimensions, exist_eos=exist_eos)

    optimizer = core.build_optimizer(game.parameters())

    trainer = core.Trainer(game=game, optimizer=optimizer, train_data=train_loader,
                           validation_data=validation_loader, callbacks=[EarlyStopperAccuracy(opts.early_stopping_thr), ConsoleLogger(print_train_loss=False, as_json=True)])

    trainer.train(n_epochs=opts.n_epochs)

    if opts.checkpoint_dir:
        checkpointer = CheckpointSaver(checkpoint_path=opts.checkpoint_dir)
        checkpointer.on_train_begin(trainer)
        checkpointer.save_checkpoint(filename=f'{opts.name}_dim{chars}vocab{opts.vocab_size}_probs{opts.probs}_type{opts.complex_gram}_rs{opts.random_seed}_lr{opts.lr}_shid{opts.sender_hidden}_rhid{opts.receiver_hidden}_sentr{opts.sender_entropy_coeff}_reg{opts.length_cost}_max_len{opts.max_len}')

    dump(trainer.game, dimensions, test, device, False)
    core.close()


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
