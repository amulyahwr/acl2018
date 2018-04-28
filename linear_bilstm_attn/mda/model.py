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
        self.wh = nn.Linear(2*self.mem_dim, self.hidden_dim)
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
        
        self.softmax = nn.Softmax()

        self.attn = nn.Linear(2 * mem_dim, mem_dim)


    def forward(self, ltree, linputs, rtree, rinputs):
        linputs = self.emb(linputs)
        rinputs = self.emb(rinputs)

        lflip = np.flip(linputs.data.cpu().numpy(), 0).copy()  # Reverse the input
        lflip = Var(torch.from_numpy(lflip))

        rflip = np.flip(rinputs.data.cpu().numpy(), 0).copy()  # Reverse the input
        rflip = Var(torch.from_numpy(rflip))
        
        ################################left sentence##############################
        hiddn_state_l_f = []
        hiddn_state_l_b = []

        # forward lstm on left sentence
        hiddn_state_mat_all = []
        _, _, hiddn_state_mat_l_f = self.childsumtreelstm(ltree, linputs, hiddn_state_mat_all)

        # backward lstm on left sentence
        hiddn_state_mat_all = []
        _, _, hiddn_state_mat_l_b = self.childsumtreelstm(ltree, lflip, hiddn_state_mat_all)

        for each in range(len(hiddn_state_mat_l_f)):
            if hiddn_state_mat_l_f[each] is not None:
                hiddn_state_l_f.append(hiddn_state_mat_l_f[each])

        hiddn_state_l_f = torch.cat(hiddn_state_l_f, dim=0)

        for each in reversed(range(len(hiddn_state_mat_l_b))):
            if hiddn_state_mat_l_b[each] is not None:
                hiddn_state_l_b.append(hiddn_state_mat_l_b[each])

        hiddn_state_l_b = torch.cat(hiddn_state_l_b, dim=0)

        hiddn_state_l = F.tanh(hiddn_state_l_f + hiddn_state_l_b)


        ################################right sentence##############################
        hiddn_state_r_f = []
        hiddn_state_r_b = []

        # forward lstm on right sentence
        hiddn_state_mat_all = []
        _, _, hiddn_state_mat_r_f = self.childsumtreelstm(rtree, rinputs, hiddn_state_mat_all)

        # backward lstm on right sentence
        hiddn_state_mat_all = []
        _, _, hiddn_state_mat_r_b = self.childsumtreelstm(rtree, rflip, hiddn_state_mat_all)

        for each in range(len(hiddn_state_mat_r_f)):
            if hiddn_state_mat_r_f[each] is not None:
                hiddn_state_r_f.append(hiddn_state_mat_r_f[each])

        hiddn_state_r_f = torch.cat(hiddn_state_r_f, dim=0)

        for each in reversed(range(len(hiddn_state_mat_r_b))):
            if hiddn_state_mat_r_b[each] is not None:
                hiddn_state_r_b.append(hiddn_state_mat_r_b[each])

        hiddn_state_r_b = torch.cat(hiddn_state_r_b, dim=0)

        hiddn_state_r = F.tanh(hiddn_state_r_f + hiddn_state_r_b)

        # modified decomposible attention
        unnrmalized_scores = torch.matmul(hiddn_state_l, torch.t(hiddn_state_r))

        nrmalized_scores_r = torch.t(self.softmax(unnrmalized_scores))

        nrmalized_scores_l = torch.t(self.softmax(torch.t(unnrmalized_scores)))

        beta = torch.mul(torch.unsqueeze(hiddn_state_r, 1), torch.unsqueeze(nrmalized_scores_r, 2))
        beta_sum = torch.sum(beta, dim=0)

        alpha = torch.mul(torch.unsqueeze(hiddn_state_l, 1), torch.unsqueeze(nrmalized_scores_l, 2))
        alpha_sum = torch.sum(alpha, dim=0)

        v_l = self.attn(torch.cat((hiddn_state_l, beta_sum), dim=1))

        v_r = self.attn(torch.cat((hiddn_state_r, alpha_sum), dim=1))

        lhidden = torch.unsqueeze(v_l[len(hiddn_state_l)-1], 0)
        rhidden = torch.unsqueeze(v_r[len(hiddn_state_r)-1], 0)

        output = self.similarity(lhidden, rhidden)
        return output
