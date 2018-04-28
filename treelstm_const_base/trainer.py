from tqdm import tqdm

import torch
from torch.autograd import Variable as Var

from utils import map_label_to_target
import numpy as np

class Trainer(object):
    def __init__(self, args, model, criterion, optimizer):
        super(Trainer, self).__init__()
        self.args       = args
        self.model      = model
        self.criterion  = criterion
        self.optimizer  = optimizer
        self.epoch      = 0

    # helper function for training
    def train(self, dataset):
        self.model.train()
        self.optimizer.zero_grad()
        loss, k = 0.0, 0
        indices = torch.randperm(len(dataset))
        for idx in tqdm(range(len(dataset)),desc='Training epoch '+str(self.epoch+1)+''):
            lctree, lsent, rctree, rsent, label = dataset[indices[idx]]
            if lctree is not None and rctree is not None:
                linput, rinput = Var(lsent), Var(rsent)
                target = Var(map_label_to_target(label,dataset.num_classes))
                if self.args.cuda:
                    linput, rinput = linput.cuda(), rinput.cuda()
                    target = target.cuda()
                output = self.model(lctree,linput,rctree,rinput)
                err = self.criterion(output, target)
                loss += err.data[0]
                (err/self.args.batchsize).backward()
                k += 1
                if k%self.args.batchsize==0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

        self.epoch += 1
        return loss/len(dataset)

    # helper function for testing
    def test(self, dataset):
        self.model.eval()
        loss = 0
        count = 0
        predictions = torch.zeros(len(dataset))
        idxs = []
        indices = torch.arange(1,dataset.num_classes+1)
        for idx in tqdm(range(len(dataset)),desc='Testing epoch  '+str(self.epoch)+''):
            lctree,lsent,rctree,rsent,label = dataset[idx]
            if lctree is not None and rctree is not None:
                count = count + 1
                linput, rinput = Var(lsent, volatile=True), Var(rsent, volatile=True)
                target = Var(map_label_to_target(label,dataset.num_classes), volatile=True)
                if self.args.cuda:
                    linput, rinput = linput.cuda(), rinput.cuda()
                    target = target.cuda()
                output = self.model(lctree,linput,rctree,rinput)
                err = self.criterion(output, target)
                loss += err.data[0]
                output = output.data.squeeze().cpu()
                idxs.append(idx)
                predictions[idx] = torch.dot(indices, torch.exp(output))
        print('Sentences processed: %d'%(count))
        return loss/len(dataset), predictions , torch.from_numpy(np.asarray(idxs))
