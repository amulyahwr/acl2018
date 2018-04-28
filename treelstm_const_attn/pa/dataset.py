import os
from tqdm import tqdm
from copy import deepcopy

import torch
import torch.utils.data as data

import Constants
from tree import Tree
from nltk.tree import Tree as CTree
from vocab import Vocab

# Dataset class for SICK dataset
class SICKDataset(data.Dataset):
    def __init__(self, path, vocab, num_classes):
        super(SICKDataset, self).__init__()
        self.vocab = vocab
        self.num_classes = num_classes

        self.lsentences = self.read_sentences(os.path.join(path,'a.toks'))
        self.rsentences = self.read_sentences(os.path.join(path,'b.toks'))

        self.lctrees = self.read_ctrees(os.path.join(path, 'a.cparents'))
        self.rctrees = self.read_ctrees(os.path.join(path, 'b.cparents'))

        self.labels = self.read_labels(os.path.join(path,'par.txt'))

        self.size = self.labels.size(0)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        lctree = deepcopy(self.lctrees[index])
        rctree = deepcopy(self.rctrees[index])
        lsent = deepcopy(self.lsentences[index])
        rsent = deepcopy(self.rsentences[index])
        label = deepcopy(self.labels[index])
        return (lctree,lsent,rctree,rsent,label)

    def read_sentences(self, filename):
        with open(filename,'r') as f:
            sentences = [self.read_sentence(line) for line in tqdm(f.readlines())]
        return sentences

    def read_sentence(self, line):
        indices = self.vocab.convertToIdx(line.split(), Constants.UNK_WORD)
        return torch.LongTensor(indices)

    def read_ctrees(self, filename):
        with open(filename,'r') as f:
            trees = [self.read_ctree(line) for line in tqdm(f.readlines())]
        return trees

    def read_ctree(self, line):
        t = CTree.fromstring(line)
        count = 0
        for s in t.subtrees(lambda t: t.height() == 2):
            s.__setattr__('idx',count)
            count = count + 1

        return t

    def read_labels(self, filename):
        with open(filename,'r') as f:
            labels = list(map(lambda x: float(x), f.readlines()))
            labels = torch.Tensor(labels)
        return labels
