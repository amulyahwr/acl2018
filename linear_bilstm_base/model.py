import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as Var
import numpy as np

import Constants

# module for childsumtreelstm
class ChildSumTreeLSTM(nn.Module):
    def __init__(self, in_dim, mem_dim):
        super(ChildSumTreeLSTM, self).__init__()
        self.in_dim = in_dim
        self.mem_dim = mem_dim
        self.ioux = nn.Linear(self.in_dim, 3*self.mem_dim)
        self.iouh = nn.Linear(self.mem_dim, 3*self.mem_dim)
        self.fx = nn.Linear(self.in_dim,self.mem_dim)
        self.fh = nn.Linear(self.mem_dim,self.mem_dim)

    def node_forward(self, inputs, child_c, child_h):

        iou = self.ioux(inputs) + self.iouh(child_h)
        i, o, u = torch.split(iou, iou.size(1)//3, dim=1)
        i, o, u = F.sigmoid(i), F.sigmoid(o), F.tanh(u)

        f = F.sigmoid(
                self.fh(child_h) +
                self.fx(inputs)
            )
        fc = torch.mul(f, child_c)

        c = torch.mul(i, u) + torch.sum(fc, dim=0, keepdim=True)
        h = torch.mul(o, F.tanh(c))
        return c, h

    def forward(self, tree, inputs, hiddn_state_mat_all):

        child_c = Var(inputs[0].data.new(1, self.mem_dim).fill_(0.))
        child_h = Var(inputs[0].data.new(1, self.mem_dim).fill_(0.))

        for idx in range(len(inputs)):
            tree.state = self.node_forward(inputs[idx], child_c, child_h)
            child_c = tree.state[0]
            child_h = tree.state[1]

            hiddn_state_mat_all.append(child_h)

        return child_c, child_h, hiddn_state_mat_all

# module for distance-angle similarity
class Similarity(nn.Module):
    def __init__(self, mem_dim, hidden_dim, num_classes):
        super(Similarity, self).__init__()
        self.mem_dim = mem_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.wh = nn.Linear(4*self.mem_dim, self.hidden_dim)
        self.wp = nn.Linear(self.hidden_dim, self.num_classes)

    def forward(self, lvec, rvec):
        mult_dist = torch.mul(lvec, rvec)
        abs_dist = torch.abs(torch.add(lvec, -rvec))
        vec_dist = torch.cat((mult_dist, abs_dist), 1)

        out = F.sigmoid(self.wh(vec_dist))
        out = F.log_softmax(self.wp(out))
        return out

# putting the whole model together
class SimilarityTreeLSTM(nn.Module):
    def __init__(self, vocab_size, in_dim, mem_dim, hidden_dim, num_classes, sparsity):
        super(SimilarityTreeLSTM, self).__init__()
        self.emb = nn.Embedding(vocab_size, in_dim, padding_idx=Constants.PAD, sparse=sparsity)
        self.emb.weight.requires_grad = False

        self.childsumtreelstm = ChildSumTreeLSTM(in_dim, mem_dim)
        self.similarity = Similarity(mem_dim, hidden_dim, num_classes)

        self.bilstm = nn.LSTM(in_dim, mem_dim, num_layers=1, bidirectional=True)
        self.h0 = Var(torch.zeros(2, 1, mem_dim))
        self.c0 = Var(torch.zeros(2, 1, mem_dim))

    def forward(self, ltree, linputs, rtree, rinputs):
        linputs = self.emb(linputs)
        rinputs = self.emb(rinputs)

        #forward lstm on left sentence
        hiddn_state_mat_all = []
        _, lhidden_f, _ = self.childsumtreelstm(ltree, linputs, hiddn_state_mat_all)

        # backward lstm on left sentence
        hiddn_state_mat_all = []
        lflip = np.flip(linputs.data.numpy(), 0).copy()  # Reverse the input
        lflip = Var(torch.from_numpy(lflip))
        _, _, hiddn_state_l = self.childsumtreelstm(ltree, lflip, hiddn_state_mat_all)
        lhidden_b = hiddn_state_l[0]

        lhidden = torch.cat((lhidden_f,lhidden_b),dim=1)

        # forward lstm on right sentence
        hiddn_state_mat_all = []
        _, rhidden_f, _ = self.childsumtreelstm(rtree, rinputs, hiddn_state_mat_all)

        # backward lstm on right sentence
        hiddn_state_mat_all = []
        rflip = np.flip(rinputs.data.numpy(), 0).copy()  # Reverse the input
        rflip = Var(torch.from_numpy(rflip))
        _, _, hiddn_state_r = self.childsumtreelstm(rtree, rflip, hiddn_state_mat_all)
        rhidden_b = hiddn_state_r[0]

        rhidden = torch.cat((rhidden_f, rhidden_b), dim=1)

        output = self.similarity(lhidden, rhidden)
        return output
