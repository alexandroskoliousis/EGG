# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.utils.data as data
import torch.nn.parallel
import torch
import numpy as np
import itertools
import pandas as pd
import random

def concatLoaders(loader1, loader2):
    loader = [x for x in loader1]
    loader.extend([x for x in loader2])


class _SimpleIterator:
    def __init__(self, all_dim, probs, n_batches_per_epoch, batch_size, seed=None):
        self.n_batches_per_epoch = n_batches_per_epoch
        self.all_dim = all_dim
        self.batch_size = batch_size
        self.probs = probs
        self.batches_generated = 0
        self.random_state = np.random.RandomState(seed)

    def __iter__(self):
        return self

    def __next__(self):
        if self.batches_generated >= self.n_batches_per_epoch:
            raise StopIteration()

        simple_batch = torch.Tensor()
        for pos in range(len(self.all_dim)):
            batch_data = self.random_state.multinomial(1, self.probs[pos], size=self.batch_size)
            torch_batch_data = torch.cat((torch.zeros((batch_data.shape[0], 1)), torch.from_numpy(batch_data).float()), 1)

            new_batch_data = torch.Tensor()
            for i, dim in enumerate(self.all_dim):
                if i==pos:
                    new_batch_data = torch.cat((new_batch_data,torch_batch_data),1)
                else:
                    other_dim = torch.eye(dim+1)[:,0].repeat(torch_batch_data.size(0), 1)
                    new_batch_data = torch.cat((new_batch_data, other_dim),1)

            simple_batch = torch.cat((simple_batch, new_batch_data))

        simple_batch=simple_batch[torch.randperm(simple_batch.size()[0])] # Shuffle the batch
        self.batches_generated += 1
        return simple_batch#, torch.zeros(1)


class SimpleLoader(torch.utils.data.DataLoader):
    def __init__(self, all_dim, batches_per_epoch, batch_size, probs, seed=None):
        self.seed = seed
        self.batches_per_epoch = batches_per_epoch
        self.all_dim = all_dim
        self.batch_size = batch_size
        self.probs = probs

    def __iter__(self):
        if self.seed is None:
            seed = np.random.randint(0, 2 ** 32)
        else:
            seed = self.seed
        return _SimpleIterator(n_batches_per_epoch=self.batches_per_epoch, batch_size=self.batch_size,
                                probs=self.probs, seed=seed, all_dim=self.all_dim)


class _CompoHotIterator:
    """
    An iterator where the probability of one complex referent correspond to the product
    of the probability of its simple component (n-gram model)
    """
    def __init__(self, possible_referents, n_batches_per_epoch, batch_size, probs, seed=None):
        self.referents = possible_referents
        self.n_batches_per_epoch = n_batches_per_epoch
        self.batch_size = batch_size
        self.probs = probs
        self.batches_generated = 0
        random.seed(seed)

    def __next__(self):
        if self.batches_generated >= self.n_batches_per_epoch:
            raise StopIteration()
        # Sample with respect to a given proba
        all_batch = random.choices(self.referents, weights=self.probs, k=self.batch_size)
        self.batches_generated += 1
        return torch.FloatTensor(all_batch)#, torch.zeros(1)

class CompositionalLoader(torch.utils.data.DataLoader):
    def __init__(self, possible_referents, batches_per_epoch, batch_size, probs, seed=None):
        self.possible_referents = possible_referents
        self.seed = seed
        self.batches_per_epoch = batches_per_epoch
        self.batch_size = batch_size
        self.probs = probs

    def __iter__(self):
        if self.seed is None:
            seed = np.random.randint(0, 2 ** 32)
        else:
            seed = self.seed
        return _CompoHotIterator(possible_referents=self.possible_referents, n_batches_per_epoch=self.batches_per_epoch,
                               batch_size=self.batch_size, probs=self.probs, seed=seed)


class _ConcatIterator:
    def __init__(self, loader1, loader2, seed=None):
        self.loader1 = [x for x in loader1]
        self.loader2 = [x for x in loader2]
        assert len(self.loader1)==len(self.loader2)
        self.batches_generated = 0
        self.gen = torch.manual_seed(seed)

    def __next__(self):
        if self.batches_generated >= len(self.loader2):
            raise StopIteration()

        # Sample with respect to a given proba
        batch = torch.cat((self.loader1[self.batches_generated], self.loader2[self.batches_generated]))
        shuffeled_batch = batch[torch.randperm(batch.size()[0], generator=self.gen)]
        self.batches_generated += 1
        return shuffeled_batch, torch.zeros(1)


class ConcatLoader(torch.utils.data.DataLoader):
    def __init__(self, loader1, loader2, seed=None):
        self.loader1 = loader1
        self.loader2 = loader2
        self.seed = seed

    def __iter__(self):
        if self.seed is None:
            seed = np.random.randint(0, 2 ** 32)
        else:
            seed = self.seed
        return _ConcatIterator(self.loader1, self.loader2, seed=seed)

class UniformLoader(torch.utils.data.DataLoader):
    def __init__(self, dimensions, complex_train):

        simple = torch.Tensor()
        for pos, a_dim in enumerate(dimensions):
            data = torch.eye(a_dim)
            torch_data = torch.cat((torch.zeros((data.shape[0], 1)), data.float()), 1)

            new_data = torch.Tensor()
            for i, dim in enumerate(dimensions):
                if i==pos:
                    new_data = torch.cat((new_data,torch_data),1)
                else:
                    other_dim = torch.eye(dim+1)[:,0].repeat(torch_data.size(0), 1)
                    new_data = torch.cat((new_data, other_dim),1)

            simple = torch.cat((simple, new_data))

        self.batch = torch.cat((simple, complex_train)), torch.zeros(1)

    def __iter__(self):
        return iter([self.batch])
