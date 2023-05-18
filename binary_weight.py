import logging
import math
from enum import Enum

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class LearnableBias(nn.Module):
    def __init__(self, out_chn):
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(out_chn), requires_grad=True)

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        return out

class BwnQuantizer(torch.autograd.Function):
    """Binary Weight Network (BWN)
     Ref: https://arxiv.org/abs/1603.05279
     """

    @staticmethod
    def forward(ctx, input, clip_val, num_bits, layerwise):
        """
        :param input: tensor to be binarized
        :return: quantized tensor
        """
        ctx.save_for_backward(input)
        if layerwise:
            s = input.size()
            m = input.norm(p=1).div(input.nelement())#nelement: number of element
            e = input.mean()
            result = (input-e).sign().mul(m.expand(s))
        else:
            n = input[0].nelement()  # W of size axb, return a vector of  ax1
            s = input.size()
            m = input.norm(1, 1, keepdim=True).div(n)
            e = input.mean()
            result = (input-e).sign().mul(m.expand(s))

        return result

    @staticmethod
    def backward(ctx, grad_output):
        """
        :param ctx: saved non-clipped full-precision tensor and clip_val
        :param grad_output: gradient ert the quantized tensor
        :return: estimated gradient wrt the full-precision tensor
        """
        grad_input = grad_output.clone()
        return grad_input, None, None, None

def weight_quant_fn(weight,  clip_val,  num_bits,  symmetric, quant_method, layerwise):
    if num_bits == 32:
        return weight
    elif quant_method == "bwn" and num_bits == 1:
        quant_fn = BwnQuantizer
    else:
        raise ValueError("Unknown quant_method")

    weight = quant_fn.apply(weight, clip_val,  num_bits, layerwise)
    return weight


class QuantizeLinear(nn.Linear):

    def __init__(self, *kargs, clip_val=2.5, weight_bits=8, learnable=False, symmetric=True,
                 weight_layerwise=True, weight_quant_method="bwn", **kwargs):
        super(QuantizeLinear, self).__init__(*kargs, **kwargs)
        self.weight_bits = weight_bits
        self.learnable = learnable
        self.symmetric = symmetric
        self.weight_layerwise = weight_layerwise
        self.weight_quant_method = weight_quant_method
        self._build_weight_clip_val(weight_quant_method, learnable, init_val=clip_val)
        self.move = LearnableBias(self.weight.shape[1])

    def _build_weight_clip_val(self, quant_method, learnable, init_val):
        if quant_method == 'uniform':
            # init_val = self.weight.mean().item() + 3 * self.weight.std().item()
            self.register_buffer('weight_clip_val', torch.tensor([-init_val, init_val]))
            if learnable:
                self.weight_clip_val = nn.Parameter(self.weight_clip_val)
        else:
            self.register_buffer('weight_clip_val', torch.tensor([-init_val, init_val]))
        return self.weight_clip_val

    def forward(self, input):
        # quantize weight
        weight = weight_quant_fn(self.weight, self.weight_clip_val, num_bits=self.weight_bits, symmetric=self.symmetric,
                                 quant_method=self.weight_quant_method, layerwise=self.weight_layerwise)
        # quantize input
        input = self.move(input)
        out = nn.functional.linear(input, weight)
        if not self.bias is None:
            out += self.bias.view(1, -1).expand_as(out)

        return out
 

class QuantizeConv2dQ(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, 
                 **kwargs_q):
        super(QuantizeConv2dQ, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias,
            )
        
        self.alpha = Parameter(torch.Tensor(out_channels))
        
        self.nbits=kwargs_q['nbits']
        self.register_buffer('init_state', torch.zeros(1))

    def forward(self, x):
        if self.nbits == 32:
            return F.conv2d(x, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
        # w_reshape = self.weight.reshape([self.weight.shape[0], -1]).transpose(0, 1)
        Qn = -2 ** (self.nbits - 1)
        Qp = 2 ** (self.nbits - 1) - 1
        if self.training and self.init_state == 0:
            # self.alpha.data.copy_(self.weight.abs().max() / 2 ** (self.nbits - 1))
            self.alpha.data.copy_(2 * self.weight.abs().mean() / math.sqrt(Qp))
            # self.alpha.data.copy_(self.weight.abs().max() * 2)
            self.init_state.fill_(1)
        
        g = 1.0 / math.sqrt(self.weight.numel() * Qp)
        alpha = grad_scale(self.alpha, g)
        alpha = alpha.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        w_q = round_pass((self.weight / alpha).clamp(Qn, Qp)) * alpha
        return F.conv2d(x, w_q, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return y.detach() - y_grad.detach() + y_grad


def round_pass(x):
    y = x.round()
    y_grad = x
    return y.detach() - y_grad.detach() + y_grad
def sign_pass(x):
    y = x.sign()
    y_grad = x
    return y.detach() - y_grad.detach() + y_grad
 

        
        
    