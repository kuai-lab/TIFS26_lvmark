from __future__ import absolute_import, division, print_function
import torch
import torch.nn as nn
import copy
import torch.nn.functional as F
# import utils

def get_activation(act):
    activations = {
        "relu": nn.ReLU(inplace=True),
        "lrelu": nn.LeakyReLU(0.2, inplace=True),
        "elu": nn.ELU(alpha=1.0, inplace=True),
        "prelu": nn.PReLU(num_parameters=1, init=0.25),
        "selu": nn.SELU(inplace=True)
    }
    return activations[act]

def calc_gradient_penalty(netD, real_data, fake_data, LAMBDA, device):
    alpha = torch.rand(1, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.to(device)

    interpolates = (alpha * real_data + ((1 - alpha) * fake_data))
    interpolates = torch.autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                    grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]
    # LAMBDA = 1
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty


class ConvBlock3DSN(nn.Sequential):
    def __init__(self, in_channel, out_channel, ker_size, padding, stride, bn=True, act='lrelu'):
        super(ConvBlock3DSN, self).__init__()
        if bn:
            self.add_module('conv', nn.utils.spectral_norm(nn.Conv3d(in_channel, out_channel, kernel_size=ker_size,
                                                                     stride=stride, padding=padding)))
        else:
            self.add_module('conv',
                            nn.Conv3d(in_channel, out_channel, kernel_size=ker_size, stride=stride, padding=padding,
                                      padding_mode='reflect'))
        if act is not None:
            self.add_module(act, get_activation(act))


class WDiscriminator3D(nn.Module):
    def __init__(self):
        super(WDiscriminator3D, self).__init__()

        N = int(64)

        self.head = ConvBlock3DSN(3, N, 3, 3 // 2, stride=1, bn=True, act='lrelu')
        self.body = nn.Sequential()
        for i in range(5):
            block = ConvBlock3DSN(N, N, 3, 3 // 2, stride=1, bn=True, act='lrelu')
            self.body.add_module('block%d' % (i), block)
        self.tail = nn.Conv3d(N, 1, kernel_size=3, padding=1, stride=1)
        # self.flatten = nn.Flatten()
        # self.fc = nn.Linear(1 * 8 * 256 * 256, 1)
    def forward(self, x):
        head = self.head(x)
        body = self.body(head)
        out = self.tail(body)
        # out = self.flatten(out)
        # out = self.fc(out)
        return out.squeeze()

class WDiscriminator3D_v2(nn.Module):
    def __init__(self):
        super(WDiscriminator3D_v2, self).__init__()

        N = int(64)

        self.head = ConvBlock3DSN(3, N, 3, 3 // 2, stride=1, bn=True, act='lrelu')
        self.body = nn.Sequential()
        for i in range(5):
            block = ConvBlock3DSN(N, N, 3, 3 // 2, stride=1, bn=True, act='lrelu')
            self.body.add_module('block%d' % (i), block)
        self.tail = nn.Conv3d(N, 1, kernel_size=3, padding=1, stride=1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(1 * 8 * 256 * 256, 1)
    def forward(self, x):
        head = self.head(x)
        body = self.body(head)
        out = self.tail(body)
        out = self.flatten(out)
        out = self.fc(out)
        return out.squeeze()
