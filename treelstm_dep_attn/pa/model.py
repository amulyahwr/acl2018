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

        self.softmax = nn.Softmax()

        self.Wa = nn.Parameter(torch.Tensor(1, self.mem_dim).normal_(-0.05, 0.05))
        self.Wa.requires_grad = True

        self.attnh = nn.Linear(2 * self.mem_dim, self.mem_dim)

    def node_forward(self, inputs, child_c, child_h,hiddn_state_mat):

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

        #progressive attention
        if hiddn_state_mat is not None:
            cat_vec = torch.cat((h.repeat(len(hiddn_state_mat),1), hiddn_state_mat), dim=1)
            unnrmalized_scores = torch.matmul(self.Wa, torch.t(F.tanh(self.attnh(cat_vec))))

            nrmalized_scores = self.softmax(unnrmalized_scores)

            h = torch.sum(torch.mul(torch.t((1 - nrmalized_scores)),hiddn_state_mat),dim=0,keepdim=True) + \
            torch.sum(torch.mul(torch.t(nrmalized_scores),h.repeat(len(hiddn_state_mat),1)),dim=0,keepdim=True)

        return c, h

    def forward(self, tree, inputs,hiddn_state_mat_all,hiddn_state_mat):
        _ = [self.forward(tree.children[idx], inputs,hiddn_state_mat_all,hiddn_state_mat) for idx in range(tree.num_children)]

        if tree.num_children == 0:
            child_c = Var(inputs[0].data.new(1, self.mem_dim).fill_(0.))
            child_h = Var(inputs[0].data.new(1, self.mem_dim).fill_(0.))
        else:
            child_c, child_h = zip(*map(lambda x: x.state, tree.children))
            child_c, child_h = torch.cat(child_c, dim=0), torch.cat(child_h, dim=0)

        tree.state = self.node_forward(inputs[tree.idx], child_c, child_h,hiddn_state_mat)

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

    def forward(self, ltree, linputs, rtree, rinputs):
        linputs = self.emb(linputs)
        rinputs = self.emb(rinputs)

        #source is left sentence and target is right sentence
        hiddn_state_mat_all = {}
        hiddn_state_l = []
        _, lhidden_1, hiddn_state_mat_l = self.childsumtreelstm(ltree, linputs, hiddn_state_mat_all,None)
        for each in range(linputs.size()[0]):
            if hiddn_state_mat_l.get(each) is not None:
                hiddn_state_l.append(hiddn_state_mat_l.get(each))

        hiddn_state_l = torch.cat(hiddn_state_l, dim=0)

        hiddn_state_mat_all = {}
        _, rhidden_1,_ = self.childsumtreelstm(rtree, rinputs, hiddn_state_mat_all, hiddn_state_l)

        # source is right sentence and target is left sentence
        hiddn_state_r = []
        hiddn_state_mat_all = {}
        _, rhidden_2, hiddn_state_mat_r = self.childsumtreelstm(rtree, rinputs, hiddn_state_mat_all, None)
        for each in range(rinputs.size()[0]):
            if hiddn_state_mat_r.get(each) is not None:
                hiddn_state_r.append(hiddn_state_mat_r.get(each))

        hiddn_state_r = torch.cat(hiddn_state_r, dim=0)

        hiddn_state_mat_all = {}
        _, lhidden_2, _ = self.childsumtreelstm(ltree, linputs, hiddn_state_mat_all, hiddn_state_r)


        #final vectors
        lhidden = F.tanh(lhidden_1+lhidden_2)
        rhidden = F.tanh(rhidden_1+rhidden_2)

        output = self.similarity(lhidden, rhidden)
        return output
