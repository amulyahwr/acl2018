import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as Var
import torch.nn.init as init

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

        child_h_sum = torch.sum(child_h, dim=0, keepdim=True)

        iou = self.ioux(inputs) + self.iouh(child_h_sum)
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i, o, u = F.sigmoid(i), F.sigmoid(o), F.tanh(u)

        f = F.sigmoid(
            self.fh(child_h) +
            self.fx(inputs).repeat(len(child_h), 1)
        )
        fc = torch.mul(f, child_c)

        c = torch.mul(i, u) + torch.sum(fc, dim=0, keepdim=True)
        h = torch.mul(o, F.tanh(c))

        return c, h

    def forward(self, tree, inputs,hiddn_state_mat_all):
        _ = [self.forward(tree.children[idx], inputs,hiddn_state_mat_all) for idx in range(tree.num_children)]

        if tree.num_children == 0:
            child_c = Var(inputs[0].data.new(1, self.mem_dim).fill_(0.))
            child_h = Var(inputs[0].data.new(1, self.mem_dim).fill_(0.))
        else:
            child_c, child_h = zip(*map(lambda x: x.state, tree.children))
            child_c, child_h = torch.cat(child_c, dim=0), torch.cat(child_h, dim=0)

        tree.state = self.node_forward(inputs[tree.idx], child_c, child_h)

        hiddn_state_mat_all[tree.idx] = tree.state[1]

        return tree.state[0],tree.state[1], hiddn_state_mat_all

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

        self.attn = nn.Linear(2*mem_dim, mem_dim)

    def forward(self, ltree, linputs, rtree, rinputs):

        linputs = self.emb(linputs)
        rinputs = self.emb(rinputs)

        # processing left tree
        hiddn_state_mat_all = {}
        hiddn_state_l = []

        _, lhidden_1, hiddn_state_mat_l = self.childsumtreelstm(ltree, linputs, hiddn_state_mat_all)
        for each in range(linputs.size()[0]):
            if hiddn_state_mat_l.get(each) is not None:
                hiddn_state_l.append(hiddn_state_mat_l.get(each))

        hiddn_state_l = torch.cat(hiddn_state_l, dim=0)

        # processing right tree
        hiddn_state_mat_all = {}
        hiddn_state_r = []

        _, rhidden_1,hiddn_state_mat_r = self.childsumtreelstm(rtree, rinputs, hiddn_state_mat_all)
        for each in range(rinputs.size()[0]):
            if hiddn_state_mat_r.get(each) is not None:
                hiddn_state_r.append(hiddn_state_mat_r.get(each))

        hiddn_state_r = torch.cat(hiddn_state_r, dim=0)

        #modified decomposible attention
        unnrmalized_scores = torch.matmul(hiddn_state_l, torch.t(hiddn_state_r))

        nrmalized_scores_r = torch.t(self.softmax(unnrmalized_scores))

        nrmalized_scores_l = torch.t(self.softmax(torch.t(unnrmalized_scores)))

        beta = torch.mul(torch.unsqueeze(hiddn_state_r, 1), torch.unsqueeze(nrmalized_scores_r, 2))
        beta_sum = torch.sum(beta,dim=0)


        alpha = torch.mul(torch.unsqueeze(hiddn_state_l, 1), torch.unsqueeze(nrmalized_scores_l, 2))
        alpha_sum = torch.sum(alpha,dim=0)


        v_l = self.attn(torch.cat((hiddn_state_l, beta_sum), dim=1))
        v_r = self.attn(torch.cat((hiddn_state_r,alpha_sum),dim=1))

        lhidden = torch.unsqueeze(v_l[ltree.idx],0)
        rhidden = torch.unsqueeze(v_r[rtree.idx],0)

        output = self.similarity(lhidden, rhidden)
        return output
