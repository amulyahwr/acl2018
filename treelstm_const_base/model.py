import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as Var
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

    def node_forward(self, inputs, child_c, child_h,label):

        child_h_sum = torch.sum(child_h, dim=0, keepdim=True)
        if label != 'ROOT':
            if inputs is not None:
                iou = self.ioux(inputs) + self.iouh(child_h_sum)
                i, o, u = torch.split(iou, iou.size(1)//3, dim=1)
                i, o, u = F.sigmoid(i), F.sigmoid(o), F.tanh(u)

                f = F.sigmoid(
                        self.fh(child_h) +
                        self.fx(inputs).repeat(len(child_h), 1)
                    )
            else:
                iou = self.iouh(child_h_sum)
                i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
                i, o, u = F.sigmoid(i), F.sigmoid(o), F.tanh(u)

                f = F.sigmoid(
                    self.fh(child_h)
                )

            fc = torch.mul(f, child_c)

            c = torch.mul(i, u) + torch.sum(fc, dim=0, keepdim=True)
            h = torch.mul(o, F.tanh(c))
        else:
            c = child_c
            h = child_h

        return c, h

    def forward(self, tree, inputs):

        _ = [self.forward(tree[idx], inputs) for idx in range(len(tree)) if tree.height() != 2]

        if tree.height() == 2:
            child_c = Var(inputs[0].data.new(1, self.mem_dim).fill_(0.))
            child_h = Var(inputs[0].data.new(1, self.mem_dim).fill_(0.))
            tree.state = self.node_forward(inputs[tree.__getattribute__('idx')], child_c, child_h,tree.label())
        else:
            child_c = []
            child_h = []
            for idx in range(len(tree)):
                child_c.append(tree[idx].state[0])
                child_h.append(tree[idx].state[1])
            child_c, child_h = torch.cat(child_c, dim=0), torch.cat(child_h, dim=0)

            tree.state = self.node_forward(None, child_c, child_h,tree.label())

        return tree.state

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

    def forward(self, lctree, linputs, rctree, rinputs):
        linputs = self.emb(linputs)
        rinputs = self.emb(rinputs)
        lstate, lhidden = self.childsumtreelstm(lctree, linputs)
        rstate, rhidden = self.childsumtreelstm(rctree, rinputs)
        output = self.similarity(lhidden, rhidden)
        return output
