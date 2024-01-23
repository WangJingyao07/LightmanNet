from create_task import *

import numpy as np
import copy

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data.sampler as sampler



class learner(nn.Module):
    def __init__(self, psi):
        super(learner, self).__init__()

        filter = [64, 128, 256, 512, 512]

        # define convolution block in VGG-16
        self.block1 = self.conv_layer(3, filter[0], 1)
        self.block2 = self.conv_layer(filter[0], filter[1], 2)
        self.block3 = self.conv_layer(filter[1], filter[2], 3)
        self.block4 = self.conv_layer(filter[2], filter[3], 4)
        self.block5 = self.conv_layer(filter[3], filter[4], 5)

        # primary task prediction
        self.classifier1 = nn.Sequential(
            nn.Linear(filter[-1], filter[-1]),
            nn.ReLU(inplace=True),
            nn.Linear(filter[-1], len(psi)),
            nn.Softmax(dim=1)
        )

        # auxiliary task prediction
        self.classifier2 = nn.Sequential(
            nn.Linear(filter[-1], filter[-1]),
            nn.ReLU(inplace=True),
            nn.Linear(filter[-1], int(np.sum(psi))),
            nn.Softmax(dim=1)
        )

        # apply weight initialisation
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def conv_layer(self, in_channel, out_channel, index):
        if index < 3:
            conv_block = nn.Sequential(
                nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
        else:
            conv_block = nn.Sequential(
                nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
        return conv_block

    def forward(self, x):
        g_block1 = self.block1(x)
        g_block2 = self.block2(g_block1)
        g_block3 = self.block3(g_block2)
        g_block4 = self.block4(g_block3)
        g_block5 = self.block5(g_block4)

        t1_pred = self.classifier1(g_block5.view(g_block5.size(0), -1))
        t2_pred = self.classifier2(g_block5.view(g_block5.size(0), -1))
        return t1_pred, t2_pred

    def model_fit(self, x_pred, x_output, pri=True, num_output=3):
        if not pri:
            # generated auxiliary label is a soft-assignment vector (no need to change into one-hot vector)
            x_output_onehot = x_output
        else:
            # convert a single label into a one-hot vector
            x_output_onehot = torch.zeros((len(x_output), num_output)).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
            x_output_onehot.scatter_(1, x_output.unsqueeze(1), 1)

        # apply focal loss
        loss = x_output_onehot * (1 - x_pred)**2 * torch.log(x_pred + 1e-20)
        return torch.sum(-loss, dim=1)

    def model_entropy(self, x_pred1):
        # compute entropy loss
        x_pred1 = torch.mean(x_pred1, dim=0)
        loss1 = x_pred1 * torch.log(x_pred1 + 1e-20)
        return torch.sum(loss1)


