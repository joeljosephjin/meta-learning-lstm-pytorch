from __future__ import division, print_function, absolute_import

import pdb
import copy
from collections import OrderedDict

import torch
import torch.nn as nn
import numpy as np

class Learner(nn.Module):

    def __init__(self, image_size, bn_eps, bn_momentum, n_classes):
        super(Learner, self).__init__()
        self.model = nn.ModuleDict({'features': nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 32, 3, padding=1)),
            ('norm1', nn.BatchNorm2d(32, bn_eps, bn_momentum)),
            ('relu1', nn.ReLU(inplace=False)),
            ('pool1', nn.MaxPool2d(2)),

            ('conv2', nn.Conv2d(32, 32, 3, padding=1)),
            ('norm2', nn.BatchNorm2d(32, bn_eps, bn_momentum)),
            ('relu2', nn.ReLU(inplace=False)),
            ('pool2', nn.MaxPool2d(2)),

            ('conv3', nn.Conv2d(32, 32, 3, padding=1)),
            ('norm3', nn.BatchNorm2d(32, bn_eps, bn_momentum)),
            ('relu3', nn.ReLU(inplace=False)),
            ('pool3', nn.MaxPool2d(2)),

            ('conv4', nn.Conv2d(32, 32, 3, padding=1)),
            ('norm4', nn.BatchNorm2d(32, bn_eps, bn_momentum)),
            ('relu4', nn.ReLU(inplace=False)),
            ('pool4', nn.MaxPool2d(2))]))
        })

        clr_in = image_size // 2**4
        # ??
        self.model.update({'cls': nn.Linear(32 * clr_in * clr_in, n_classes)})
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.model.features(x)
        x = torch.reshape(x, [x.size(0), -1])
        outputs = self.model.cls(x)
        return outputs

    # gets the model parameters in a concatenated torch list
    def get_flat_params(self):
        return torch.cat([p.view(-1) for p in self.model.parameters()], 0)

    # copy from cell state to model.parameters
    def copy_flat_params(self, cI):
        idx = 0
        # for each parameter in the model
        for p in self.model.parameters():
            # get the number-length
            plen = p.view(-1).size(0)
            # convert the flat parameters from the cell state to p-shape and then copy the values into the parameter
            p.data.copy_(cI[idx: idx+plen].view_as(p))
            idx += plen

    # same as copy_flat_params; only diff = parameters of the model are not nn.Params anymore, they're just plain tensors now.
    def transfer_params(self, learner_w_grad, cI):
        # Use load_state_dict only to copy the running mean/var in batchnorm, the values of the parameters
        #  are going to be replaced by cI
        self.load_state_dict(learner_w_grad.state_dict())
        #  replace nn.Parameters with tensors from cI (NOT nn.Parameters anymore).
        idx = 0
        # model.modules ??
        for m in self.model.modules():
            # if the layers are our guys
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
                # get size integers of the parameter vectors
                wlen = m._parameters['weight'].view(-1).size(0)
                # reshape the flat params from cell state to p-shape, then clone it to send it to module._params
                m._parameters['weight'] = cI[idx: idx+wlen].view_as(m._parameters['weight']).clone()
                idx += wlen
                # bias seems to have a seperate assignment strategy
                if m._parameters['bias'] is not None:
                    # get the length of the bias vector
                    blen = m._parameters['bias'].view(-1).size(0)
                    # same as is done with the weights
                    m._parameters['bias'] = cI[idx: idx+blen].view_as(m._parameters['bias']).clone()
                    idx += blen

    def reset_batch_stats(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.reset_running_stats()

