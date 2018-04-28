from __future__ import division
from __future__ import print_function

import os
import scipy.stats
import torch
import gensim
from vocab import Vocab
import numpy as np

def load_word_vectors(path):
    if os.path.isfile(path+'.pth') and os.path.isfile(path+'.vocab'):
        print('==> File found, loading to memory')
        vectors = torch.load(path+'.pth')
        vocab = Vocab(filename=path+'.vocab')
        return vocab, vectors
    # saved file not found, read from txt file
    # and create tensors for word vectors
    print('==> File not found, preparing, be patient')
    count = sum(1 for line in open(path+'.txt'))
    with open(path+'.txt','r') as f:
        contents = f.readline().rstrip('\n').split(' ')
        dim = len(contents[1:])
    words = [None]*(count)
    vectors = torch.zeros(count,dim)
    with open(path+'.txt','r') as f:
        idx = 0
        for line in f:
            contents = line.rstrip('\n').split(' ')
            words[idx] = contents[0]
            vectors[idx] = torch.Tensor(list(map(float, contents[1:])))
            idx += 1
    with open(path+'.vocab','w') as f:
        for word in words:
            f.write(word+'\n')
    vocab = Vocab(filename=path+'.vocab')
    torch.save(vectors, path+'.pth')
    return vocab, vectors

def load_dep_vectors(path):
    if os.path.isfile(path+'.pth') and os.path.isfile(path+'.vocab'):
        print('==> File found, loading to memory')
        vectors = torch.load(path+'.pth')
        vocab = Vocab(filename=path+'.vocab')
        return vocab, vectors
    # saved file not found, read from txt file
    # and create tensors for word vectors
    print('==> File not found, preparing, be patient')
    model = gensim.models.Word2Vec.load(path)

    vectors = torch.zeros(len(model.wv.vocab), 22500)

    with open(path + '.vocab', 'w') as f,\
        open(path + '.pth', 'w') as d:
        idx = 0
        for word in list(model.wv.vocab):
            f.write(word + '\n')
            vectors[idx] = torch.Tensor(np.asarray(model[word]))
            idx += 1
    vocab = Vocab(filename=path + '.vocab')
    torch.save(vectors, path + '.pth')
    return vocab, vectors

# write unique words from a set of files to a new file
def build_vocab(filenames, vocabfile):
    vocab = set()
    for filename in filenames:
        with open(filename,'r') as f:
            for line in f:
                tokens = line.rstrip('\n').split(' ')
                vocab |= set(tokens)
    with open(vocabfile,'w') as f:
        for token in sorted(vocab):
            f.write(token+'\n')

# mapping from scalar to vector
def map_label_to_target(label,num_classes):
    target = torch.zeros(1, num_classes)

    mean = label
    std_dev = 0.4251

    for idx in range(num_classes):
        target[0][idx] = scipy.stats.norm(mean, std_dev).pdf(idx + 1)

    _, indices = torch.sort(target)

    if torch.sum(target) > 1:
        target[0][indices[0][4]] = target[0][indices[0][4]] - (torch.sum(target) - 1)

    elif torch.sum(target) < 1:
        target[0][indices[0][4]] = target[0][indices[0][4]] + 1 - (torch.sum(target))

    return target
