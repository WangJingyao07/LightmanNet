from create_task import *

import numpy as np
import copy

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data.sampler as sampler

from auxiliary_branch import *
from learner import learner
from create_dataset import TaskData

psi = [5]*20

class LightmanNet(nn.Module):
    def __init__(self, multi_task_net, label_generator):
        self.multi_task_net = multi_task_net
        self.multi_task_net_ = copy.deepcopy(multi_task_net)
        self.label_generator = label_generator

    def unrolled_backward(self, train_x, train_y, alpha, model_optim):

        train_pred1, train_pred2 = self.multi_task_net(train_x)
        train_pred3 = self.label_generator(train_x, train_y)

        train_loss1 = self.multi_task_net.model_fit(train_pred1, train_y, pri=True, num_output=20)
        train_loss2 = self.multi_task_net.model_fit(train_pred2, train_pred3, pri=False, num_output=100)
        train_loss3 = self.multi_task_net.model_entropy(train_pred3)

        loss = torch.mean(train_loss1) + torch.mean(train_loss2)

        # compute gradient
        gradients = torch.autograd.grad(loss, self.multi_task_net.parameters())


        with torch.no_grad():
            for weight, weight_, grad in zip(self.multi_task_net.parameters(), self.multi_task_net_.parameters(),
                                             gradients):
                if 'momentum' in model_optim.param_groups[0].keys():  # used in SGD with momentum
                    if model_optim.param_groups[0]['momentum'] == 0:
                        m = 0
                    else:
                        m = model_optim.state[weight].get('momentum_buffer', 0.) * model_optim.param_groups[0]['momentum']
                else:
                    m = 0
                weight_.copy_(weight - alpha * (m + grad + model_optim.param_groups[0]['weight_decay'] * weight))

        train_pred1, train_pred2 = self.multi_task_net_(train_x)
        train_loss1 = self.multi_task_net.model_fit(train_pred1, train_y, pri=True, num_output=20)
        train_loss2 = F.mse_loss(train_pred2, torch.zeros_like(train_pred2, device=train_x.device))  # dummy loss function


        loss = torch.mean(train_loss1) + 0 * torch.mean(train_loss2) + 0.2 * torch.mean(train_loss3)


        model_weights_ = tuple(self.multi_task_net_.parameters())
        d_model = torch.autograd.grad(loss, model_weights_)
        hessian = self.compute_hessian(d_model, train_x, train_y)

        with torch.no_grad():
            for mw, h in zip(self.label_generator.parameters(), hessian):
                mw.grad = - alpha * h

    def compute_hessian(self, d_model, train_x, train_y):
        norm = torch.cat([w.view(-1) for w in d_model]).norm()
        eps = 0.01 / norm


        with torch.no_grad():
            for p, d in zip(self.multi_task_net.parameters(), d_model):
                p += eps * d

        train_pred1, train_pred2 = self.multi_task_net(train_x)
        train_pred3 = self.label_generator(train_x, train_y)

        train_loss1 = self.multi_task_net.model_fit(train_pred1, train_y, pri=True, num_output=20)
        train_loss2 = self.multi_task_net.model_fit(train_pred2, train_pred3, pri=False, num_output=100)
        loss = torch.mean(train_loss1) + torch.mean(train_loss2)
        d_weight_p = torch.autograd.grad(loss, self.label_generator.parameters())


        with torch.no_grad():
            for p, d in zip(self.multi_task_net.parameters(), d_model):
                p -= 2 * eps * d

        train_pred1, train_pred2 = self.multi_task_net(train_x)
        train_pred3 = self.label_generator(train_x, train_y)

        train_loss1 = self.multi_task_net.model_fit(train_pred1, train_y, pri=True, num_output=20)
        train_loss2 = self.multi_task_net.model_fit(train_pred2, train_pred3, pri=False, num_output=100)
        loss = torch.mean(train_loss1) + torch.mean(train_loss2)
        d_weight_n = torch.autograd.grad(loss, self.label_generator.parameters())

        with torch.no_grad():
            for p, d in zip(self.multi_task_net.parameters(), d_model):
                p += eps * d

        hessian = [(p - n) / (2. * eps) for p, n in zip(d_weight_p, d_weight_n)]
        return hessian


