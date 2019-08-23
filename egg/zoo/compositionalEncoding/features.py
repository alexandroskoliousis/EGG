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

def splitting(df, ratio, seed):
    train = df.sample(frac=ratio, random_state=seed)
    test = df.drop(train.index)
    return train, test

def DF_split(df, ratio=0.9, seed=1):
    df1 = df[df['Base'] >= df['Surface']]
    df2 = df[df['Base'] < df['Surface']]

    train1, test1 = splitting(df1, ratio, seed)
    train2, test2 = splitting(df2, ratio, seed)

    train = pd.concat([train1, train2])
    test = pd.concat([test1, test2])

    return (list(train['referent']), np.array(train['Surface'])), list(test['referent'])

def dataFrameConstruction(dimensions, simple_prob, complex_prob):
    list_poss = []
    for dim in dimensions:
        all_poss = torch.eye(dim)
        list_poss.append(torch.cat((torch.zeros((all_poss.shape[0], 1)), all_poss), 1))

    combinations_inputs = itertools.product(*list_poss)
    combinations_freqs = itertools.product(*simple_prob)

    diff_freq = {'referent':[], 'Base':[], 'Surface':[]}
    for count, (inp, freq) in enumerate(zip(combinations_inputs, combinations_freqs)):
        base_freq = 1
        for i in freq:
            base_freq*=i
        diff_freq['referent'].append(torch.cat(inp).numpy())
        diff_freq['Base'].append(base_freq)
        diff_freq['Surface'].append(complex_prob[count])
    return pd.DataFrame(diff_freq)

def Split_Train_Test(dimensions, simple_prob, complex_prob, ratio=0.9, seed=1):
    df = dataFrameConstruction(dimensions, simple_prob, complex_prob)
    train, test = DF_split(df, ratio, seed)
    return train, test

def concatLoaders(loader1, loader2):
    loader = [x for x in loader1]
    loader.extend([x for x in loader2])

class _OneHotIterator:
    """
    >>> it_1 = _OneHotIterator(n_features=128, n_batches_per_epoch=2, batch_size=64, probs=np.ones(128)/128, seed=1)
    >>> it_2 = _OneHotIterator(n_features=128, n_batches_per_epoch=2, batch_size=64, probs=np.ones(128)/128, seed=1)
    >>> list(it_1)[0][0].allclose(list(it_2)[0][0])
    True
    >>> it = _OneHotIterator(n_features=8, n_batches_per_epoch=1, batch_size=4, probs=np.ones(8)/8)
    >>> data = list(it)
    >>> len(data)
    1
    >>> batch = data[0]
    >>> x, y = batch
    >>> x.size()
    torch.Size([4, 8])
    >>> x.sum(dim=1)
    tensor([1., 1., 1., 1.])
    >>> probs = np.zeros(128)
    >>> probs[0] = probs[1] = 0.5
    >>> it = _OneHotIterator(n_features=128, n_batches_per_epoch=1, batch_size=256, probs=probs, seed=1)
    >>> batch = list(it)[0][0]
    >>> batch[:, 0:2].sum().item()
    256.0
    >>> batch[:, 2:].sum().item()
    0.0
    """
    def __init__(self, n_features, n_batches_per_epoch, batch_size, probs, seed=None):
        self.n_batches_per_epoch = n_batches_per_epoch
        self.n_features = n_features
        self.batch_size = batch_size

        self.probs = probs
        self.batches_generated = 0
        self.random_state = np.random.RandomState(seed)

    def __iter__(self):
        return self

    def __next__(self):
        if self.batches_generated >= self.n_batches_per_epoch:
            raise StopIteration()

        batch_data = self.random_state.multinomial(1, self.probs, size=self.batch_size)
        self.batches_generated += 1
        return torch.from_numpy(batch_data).float(), torch.zeros(1)


class OneHotLoader(torch.utils.data.DataLoader):
    """
    >>> probs = np.ones(8) / 8
    >>> data_loader = OneHotLoader(n_features=8, batches_per_epoch=3, batch_size=2, probs=probs, seed=1)
    >>> epoch_1 = []
    >>> for batch in data_loader:
    ...     epoch_1.append(batch)
    >>> [b[0].size() for b in epoch_1]
    [torch.Size([2, 8]), torch.Size([2, 8]), torch.Size([2, 8])]
    >>> data_loader_other = OneHotLoader(n_features=8, batches_per_epoch=3, batch_size=2, probs=probs)
    >>> all_equal = True
    >>> for a, b in zip(data_loader, data_loader_other):
    ...     all_equal = all_equal and (a[0] == b[0]).all()
    >>> all_equal.item()
    0
    """
    def __init__(self, n_features, batches_per_epoch, batch_size, probs, seed=None):
        self.seed = seed
        self.batches_per_epoch = batches_per_epoch
        self.n_features = n_features
        self.batch_size = batch_size
        self.probs = probs

    def __iter__(self):
        if self.seed is None:
            seed = np.random.randint(0, 2 ** 32)
        else:
            seed = self.seed

        return _OneHotIterator(n_features=self.n_features, n_batches_per_epoch=self.batches_per_epoch,
                               batch_size=self.batch_size, probs=self.probs, seed=seed)


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


"""
class _BaseCompoHotIterator:
    ""
    An iterator where the probability of one complex referent correspond to the product
    of the probability of its simple component (n-gram model)
    ""
    def __init__(self, dimensions, n_batches_per_epoch, batch_size, probs, seed=None):
        self.dimensions = dimensions
        self.n_batches_per_epoch = n_batches_per_epoch
        self.batch_size = batch_size
        self.probs = probs
        self.batches_generated = 0
        self.random_state = np.random.RandomState(seed)

    def __next__(self):
        if self.batches_generated >= self.n_batches_per_epoch:
            raise StopIteration()

        all_poss = np.eye(sum(self.dimensions))
        list_poss = []
        start = 0
        for dim in dimensions:
            list_poss.append(all_poss[start:(start+dim)])
            start+=dim
        combinations_inputs = itertools.product(*list_poss)
        combinations_prob = itertools.product(*self.probs)

        proba = []
        for p in combinations_prob:
            mult = 1
            for t_p in p:
                mult*=t_p
            proba.append(mult)
        referents = [sum(r) for r in combinations_inputs]

        # Sample with respect to a given proba
        all_batch = random.choices(referents, weights=proba, k=self.batch_size)

        self.batches_generated += 1
        return torch.FloatTensor(all_batch), torch.zeros(1)

class BaseCompositionalLoader(torch.utils.data.DataLoader):
    def __init__(self, dimensions, batches_per_epoch, batch_size, probs, seed=None):
        if type(dimensions)==list:
            self.dimensions = dimensions
        else:
            self.dimensions = [dimensions]
        self.seed = seed
        self.batches_per_epoch = batches_per_epoch
        self.batch_size = batch_size
        self.probs = probs
        assert len(self.dimensions) == len(self.probs)

    def __iter__(self):
        if self.seed is None:
            seed = np.random.randint(0, 2 ** 32)
        else:
            seed = self.seed
        return _BaseCompoHotIterator(dimensions=self.dimensions, n_batches_per_epoch=self.batches_per_epoch,
                               batch_size=self.batch_size, probs=self.probs, seed=seed)


class _SurfaceCompoHotIterator:
    def __init__(self, dimensions, n_batches_per_epoch, batch_size, probs, seed=None):
        self.dimensions = dimensions
        self.n_batches_per_epoch = n_batches_per_epoch
        self.batch_size = batch_size
        self.probs = probs
        self.batches_generated = 0
        self.random_state = np.random.RandomState(seed)

    def __next__(self):
        if self.batches_generated >= self.n_batches_per_epoch:
            raise StopIteration()

        # Create all possible compositional items
        all_poss = torch.eye(sum(dimensions))
        list_poss = []
        start = 0
        for dim in dimensions:
            list_poss.append(all_poss[start:(start+dim)])
            start+=dim

        combinations = itertools.product(*list_poss)
        new_batch_data = []
        for combi in combinations:
            new_batch_data.append(sum(combi).unsqueeze(0))

        # Sample wrt a given probability
        tmp = random.choices(new_batch_data, weights=self.probs, k=self.batch_size)
        #list of tensor to one tensor
        all_batch = torch.cat(tmp,0)

        self.batches_generated += 1
        return all_batch, torch.zeros(1)

class SurfaceCompositionalLoader(torch.utils.data.DataLoader):
    ""
    Each (complex) referent is sampled wrt a given probability 'probs'
    ""
    def __init__(self, dimensions, batches_per_epoch, batch_size, probs, seed=None):
        if type(dimensions)==list:
            self.dimensions = dimensions
        else:
            self.dimensions = [dimensions]
        self.seed = seed
        self.batches_per_epoch = batches_per_epoch
        self.batch_size = batch_size
        self.probs = probs
        verif = 1
        for i in self.dimensions:
            verif*=i
        assert verif == len(self.probs)

    def __iter__(self):
        if self.seed is None:
            seed = np.random.randint(0, 2 ** 32)
        else:
            seed = self.seed
        return _SurfaceCompoHotIterator(dimensions=self.dimensions, n_batches_per_epoch=self.batches_per_epoch,
                               batch_size=self.batch_size, probs=self.probs, seed=seed)

class UniformLoader(torch.utils.data.DataLoader):
    def __init__(self, n_features):
        self.batch = torch.eye(n_features), torch.zeros(1)

    def __iter__(self):
        return iter([self.batch])
"""
