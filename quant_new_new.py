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

################################################################
class ElasticQuantAttention(torch.autograd.Function):
    """
        Modified from Learned Step-size Quantization.
        https://arxiv.org/abs/1902.08153
    """
    @staticmethod
    def forward(ctx, input,alpha,alpha2,alpha3, num_bits, layerwise):
        """
        :param input: input to be quantized
        :param alpha: the step size
        :param num_bits: quantization bits
        :param layerwise: rowwise quant
        :return: quantized output
        """
        if not layerwise:
            # TODO
            raise NotImplementedError
        ctx.num_bits = num_bits
        if num_bits == 32:
            return input
        if num_bits == 1:
            Qn = 0
            Qp = 1
        else:
            Qn = 0
            Qp = 2 ** (num_bits - 1) -1
        eps = torch.tensor(0.00001).float().to(alpha.device)            
        T1_mask=(input> 0.7*input.max()).float()
        T2_mask=(input> 0.9*input.max()).float()
        input_T1=input*T1_mask
        input_T2=input*T2_mask
        if alpha.item() == 1.0 and (not alpha.initialized):
            alpha.initialize_wrapper(input, num_bits, symmetric=False, init_method='default')#'default' # input
        if alpha2.item() == 1.0 and (not alpha2.initialized):
            alpha2.initialize_wrapper([input_T1,input_T2], num_bits, symmetric=False, init_method='attention1')#'attention' # input_T1
            #alpha2=alpha2-((alpha*T1_mask).sum()/T1_mask.sum())
            alpha2=alpha2-alpha
        if alpha3.item() == 1.0 and (not alpha3.initialized):
            alpha3.initialize_wrapper(input_T2, num_bits, symmetric=False, init_method='attention2')#'attention' # input_T2
            #alpha3=alpha3-(((alpha2+alpha)*T2_mask).sum()/T2_mask.sum())
            alpha3=alpha3-(alpha2+alpha)
        alpha = torch.where(alpha > eps, alpha, eps)
        alpha2 = torch.where(alpha2 > eps, alpha2, eps)
        alpha3 = torch.where(alpha3 > eps, alpha3, eps)
        assert alpha > 0, 'alpha = {:.6f} becomes non-positive'.format(alpha)
        assert alpha2 > 0, 'alpha2 = {:.6f} becomes non-positive'.format(alpha2)
        assert alpha3 > 0, 'alpha3 = {:.6f} becomes non-positive'.format(alpha3)   
        grad_scale = 1.0 / math.sqrt(input.numel()) if not Qp else 1.0 / math.sqrt(input.numel() * Qp)
        grad_scale2 = 1.0 / math.sqrt(T1_mask.sum()) if not Qp else 1.0 / math.sqrt(T1_mask.sum() * Qp)
        grad_scale3 = 1.0 / math.sqrt(T2_mask.sum()) if not Qp else 1.0 / math.sqrt(T2_mask.sum() * Qp)
        ctx.save_for_backward(input,alpha,alpha2,alpha3)
        ctx.other = grad_scale, grad_scale2, grad_scale3, Qn, Qp,T1_mask,T2_mask
        q_w = (input / alpha).round().clamp(Qn, Qp)
        w_q = q_w* alpha  + T1_mask*alpha2 + T2_mask*alpha3 
        return w_q
    @staticmethod
    def backward(ctx, grad_output):
        if ctx.num_bits == 32:
            return grad_output, None,None, None, None, None

        input,alpha,alpha2,alpha3 = ctx.saved_tensors
        grad_scale,grad_scale2, grad_scale3, Qn, Qp,T1_mask,T2_mask = ctx.other
        q_w = input /alpha
        #####
        indicate_small = (q_w < Qn).float()
        indicate_big = (q_w > Qp).float()
        indicate_middle = 1.0 - indicate_small - indicate_big # this is more cpu-friendly than torch.ones(input_.shape)
        #####        
        grad_alpha = ((indicate_small * Qn + indicate_big * Qp + indicate_middle * (
                -q_w + q_w.round())) * grad_output * grad_scale).sum().unsqueeze(dim=0)
        grad_alpha2= (T1_mask * grad_output * grad_scale2).sum().unsqueeze(dim=0)
        grad_alpha3= (T2_mask * grad_output * grad_scale3).sum().unsqueeze(dim=0)
        
        
        grad_input = indicate_middle * grad_output + T1_mask * grad_output* alpha2  + T2_mask * grad_output*alpha3
        
        return grad_input,grad_alpha, grad_alpha2,grad_alpha3, None, None    

################################################################
class ElasticQuantQKV(torch.autograd.Function):
    """
        Modified from Learned Step-size Quantization.
        https://arxiv.org/abs/1902.08153
    """
    @staticmethod
    def forward(ctx, input,alpha,alpha2,alpha3, num_bits, layerwise):
        """
        :param input: input to be quantized
        :param alpha: the step size
        :param num_bits: quantization bits
        :param layerwise: rowwise quant
        :return: quantized output
        """
        if not layerwise:
            # TODO
            raise NotImplementedError
        ctx.num_bits = num_bits
        if num_bits == 32:
            return input

        if num_bits == 1:
            Qn = -1
            Qp = 1
        else:
            Qn = -2 ** (num_bits - 1)
            Qp = 2 ** (num_bits - 1) 

        eps = torch.tensor(0.00001).float().to(alpha.device)
        T1_mask=(input> 0.7*input.max()).float()+(input< 0.7*input.min()).float()
        T2_mask=(input> 0.9*input.max()).float()+(input< 0.9*input.min()).float()
        
        input_T1=input*T1_mask
        input_T2=input*T2_mask    
        if alpha.item() == 1.0 and (not alpha.initialized):
            alpha.initialize_wrapper([input,input_T1], num_bits, symmetric=True, init_method='qkv1')
        if alpha2.item() == 1.0 and (not alpha2.initialized):
            alpha2.initialize_wrapper([input_T1,input_T2], num_bits, symmetric=True, init_method='qkv1')
            alpha2=alpha2-((alpha*T1_mask).sum()/T1_mask.sum())
        if alpha3.item() == 1.0 and (not alpha3.initialized):
            alpha3.initialize_wrapper(input_T2, num_bits, symmetric=True, init_method='qkv2')#qkv
            alpha3=alpha3-(((alpha2+alpha)*T2_mask).sum()/T2_mask.sum())
        
        alpha = torch.where(alpha > eps, alpha, eps)
        alpha2 = torch.where(alpha2 > eps, alpha2, eps)
        alpha3 = torch.where(alpha3 > eps, alpha3, eps)
        
        assert alpha > 0, 'alpha = {:.6f} becomes non-positive'.format(alpha)
        assert alpha2 > 0, 'alpha2 = {:.6f} becomes non-positive'.format(alpha2)
        assert alpha3 > 0, 'alpha3 = {:.6f} becomes non-positive'.format(alpha3)
        
        
        grad_scale = 1.0 / math.sqrt(input.numel()) if not Qp else 1.0 / math.sqrt(input.numel() * Qp)
        grad_scale2 = 1.0 / math.sqrt(T1_mask.sum()) if not Qp else 1.0 / math.sqrt(T1_mask.sum() * Qp)
        grad_scale3 = 1.0 / math.sqrt(T2_mask.sum()) if not Qp else 1.0 / math.sqrt(T2_mask.sum()  * Qp)
        
        
        
          
        ctx.save_for_backward(input,alpha,alpha2,alpha3,input_T1,input_T2)
        ctx.other = grad_scale, grad_scale2, grad_scale3, Qn, Qp
        q_w_3=(input_T2).sign()
        q_w_2=(input_T1).sign()
        q_w = (input).sign()#sign(0)=0
        w_q = q_w* alpha  + q_w_2*alpha2 + q_w_3*alpha3
        return w_q
        
    @staticmethod
    def backward(ctx, grad_output):
        if ctx.num_bits == 32:
            return grad_output, None,None, None, None, None

        input_,alpha,alpha2,alpha3,input_T1,input_T2 = ctx.saved_tensors
        grad_scale,grad_scale2, grad_scale3, Qn, Qp = ctx.other
        q_w = input_ /alpha
        q_w_2 = input_T1 /alpha2
        q_w_3 = input_T2 /alpha3
        #####
        indicate_small = (q_w < Qn).float()
        indicate_big = (q_w > Qp).float()
        indicate_middle = 1.0 - indicate_small - indicate_big # this is more cpu-friendly than torch.ones(input_.shape)
        #####
        indicate_small2 = (q_w_2 < Qn).float()
        indicate_big2 = (q_w_2 >Qp).float()
        indicate_middle2 = 1.0 - indicate_small2 - indicate_big2 # this is more cpu-friendly than torch.ones(input_.shape)
        #####
        indicate_small3 = (q_w_3 < Qn).float()
        indicate_big3 = (q_w_3 > Qp).float()
        indicate_middle3 = 1.0 - indicate_small3 - indicate_big3 # this is more cpu-friendly than torch.ones(input_.shape)
        
        grad_alpha = ((input_.sign()+indicate_middle *(-(q_w))) * grad_output * grad_scale).sum().unsqueeze(dim=0)
        grad_alpha2 = (((input_T1).sign()+indicate_middle2 *(-(q_w_2))) * grad_output * grad_scale2).sum().unsqueeze(dim=0)
        grad_alpha3 = (((input_T2).sign()+indicate_middle3 *(-(q_w_3))) * grad_output * grad_scale3).sum().unsqueeze(dim=0)
        
        
        grad_input = indicate_middle * grad_output + (indicate_middle2) * grad_output  + (indicate_middle3) * grad_output
        #grad_input =  grad_output + (indicate_middle2) * grad_output  + (indicate_middle3) * grad_output
        return grad_input,grad_alpha, grad_alpha2,grad_alpha3, None, None
    
################################################################


class ElasticQuantBinarizerSigned(torch.autograd.Function):
    """
        Modified from Learned Step-size Quantization.
        https://arxiv.org/abs/1902.08153
    """
    @staticmethod
    def forward(ctx, input, alpha, num_bits, layerwise):
        """
        :param input: input to be quantized
        :param alpha: the step size
        :param num_bits: quantization bits
        :param layerwise: rowwise quant
        :return: quantized output
        """
        if not layerwise:
            # TODO
            raise NotImplementedError
        ctx.num_bits = num_bits
        if num_bits == 32:
            return input

        if num_bits == 1:
            Qn = -1
            Qp = 1
        else:
            Qn = -2 ** (num_bits - 1)
            Qp = 2 ** (num_bits - 1) - 1

        eps = torch.tensor(0.00001).float().to(alpha.device)
        if alpha.item() == 1.0 and (not alpha.initialized):
            alpha.initialize_wrapper(input, num_bits, symmetric=True, init_method='default')
        alpha = torch.where(alpha > eps, alpha, eps)
        assert alpha > 0, 'alpha = {:.6f} becomes non-positive'.format(alpha)
        
        grad_scale = 1.0 / math.sqrt(input.numel()) if not Qp else 1.0 / math.sqrt(input.numel() * Qp)
        
        ctx.save_for_backward(input, alpha)
        ctx.other = grad_scale, Qn, Qp
        if num_bits == 1:
            q_w = (input ).sign()
        else:
            q_w = (input / alpha).round().clamp(Qn, Qp)
        w_q = q_w * alpha
        return w_q

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.num_bits == 32:
            return grad_output, None, None, None

        input_, alpha = ctx.saved_tensors
        grad_scale, Qn, Qp = ctx.other
        q_w = input_ / alpha
        indicate_small = (q_w < Qn).float()
        indicate_big = (q_w > Qp).float()
        indicate_middle = 1.0 - indicate_small - indicate_big # this is more cpu-friendly than torch.ones(input_.shape)
        if ctx.num_bits == 1:
            grad_alpha = ((input_.sign()+indicate_middle *(-(q_w))) * grad_output * grad_scale).sum().unsqueeze(dim=0)
        else:
            grad_alpha = ((indicate_small * Qn + indicate_big * Qp + indicate_middle * (
                    -q_w + q_w.round())) * grad_output * grad_scale).sum().unsqueeze(dim=0)
        grad_input = indicate_middle * grad_output
        #grad_input =  grad_output
        return grad_input, grad_alpha, None, None


class ElasticQuantBinarizerUnsigned(torch.autograd.Function):
    """
        Modified from Learned Step-size Quantization.
        https://arxiv.org/abs/1902.08153
    """
    @staticmethod
    def forward(ctx, input, alpha, num_bits, layerwise):
        """
        :param input: input to be quantized
        :param alpha: the step size
        :param num_bits: quantization bits
        :param layerwise: rowwise quant
        :return: quantized output
        """
        if not layerwise:
            # TODO
            raise NotImplementedError
        ctx.num_bits = num_bits
        if num_bits == 32:
            return input

        Qn = 0
        Qp = 2 ** (num_bits) - 1
        if num_bits == 1:
            input_ = input
        else:
            min_val = input.min().item()
            input_ = input - min_val

        eps = torch.tensor(0.00001).float().to(alpha.device)
        if alpha.item() == 1.0 and (not alpha.initialized):
            alpha.initialize_wrapper(input, num_bits, symmetric=False, init_method='default')
        alpha = torch.where(alpha > eps, alpha, eps)
        assert alpha > 0, 'alpha = {:.6f} becomes non-positive'.format(alpha)
        
        grad_scale = 1.0 / math.sqrt(input.numel() * Qp)
        ctx.save_for_backward(input_, alpha)
        ctx.other = grad_scale, Qn, Qp
        q_w = (input_ / alpha).round().clamp(Qn, Qp)
        w_q = q_w * alpha
        if num_bits != 1:
            w_q = w_q + min_val
        return w_q

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.num_bits == 32:
            return grad_output, None, None, None

        input_, alpha = ctx.saved_tensors
        grad_scale, Qn, Qp = ctx.other
        q_w = input_ / alpha
        indicate_small = (q_w < Qn).float()
        indicate_big = (q_w > Qp).float()
        indicate_smallx = (q_w < 3*Qn).float()
        indicate_bigx = (q_w > 3*Qp).float()
        indicate_middle = 1.0 - indicate_small - indicate_big 
        indicate_middlex = 1.0 - indicate_smallx - indicate_bigx  # this is more cpu-friendly than torch.ones(input_.shape)
        grad_alpha = ((indicate_small * Qn + indicate_big * Qp + indicate_middle * (
                -q_w + q_w.round())) * grad_output * grad_scale).sum().unsqueeze(dim=0)
        grad_input = indicate_middle * grad_output
        
        return grad_input, grad_alpha, None, None

class AlphaInit(nn.Parameter):
    def __init__(self, tensor):
        super(AlphaInit, self).__new__(nn.Parameter, data=tensor)
        self.initialized = False

    def _initialize(self, init_tensor):
        assert not self.initialized, 'already initialized.'
        self.data.copy_(init_tensor)
        self.initialized = True

    def initialize_wrapper(self, tensor, num_bits, symmetric, init_method='default'):
        Qp = 2 ** (num_bits - 1) - 1 if symmetric else 2 ** (num_bits) - 1
        if Qp == 0:
            Qp = 1.0
        if init_method == 'default':
            init_val = 2* tensor.abs().mean() / math.sqrt(Qp) if symmetric \
                else 4* tensor.abs().mean() / math.sqrt(Qp)
        elif init_method == 'uniform':
            init_val = 1./(2*Qp+1) if symmetric else 1./Qp
        elif init_method == 'attention':
            init_val =2* (tensor.sum()/tensor.sign().sum() )/ math.sqrt(Qp) if symmetric \
                else 4*(tensor.sum()/tensor.sign().sum() ) / math.sqrt(Qp)
        elif init_method == 'attention1':
            init_val =2* ((tensor[0].sum()-tensor[1].sum())/(tensor[0].sign().sum()-tensor[1].sign().sum()) )/ math.sqrt(Qp) if symmetric \
                else 4*((tensor[0].sum()-tensor[1].sum())/(tensor[0].sign().sum()-tensor[1].sign().sum()) ) / math.sqrt(Qp)
        elif init_method == 'attention2':
            init_val =2* (tensor.sum()/tensor.sign().sum() )/ math.sqrt(Qp) if symmetric \
                else 4*(tensor.sum()/tensor.sign().sum() ) / math.sqrt(Qp)
        elif init_method == 'qkv1':
            a=torch.count_nonzero(tensor[0]).item()
            b=torch.count_nonzero(tensor[1]).item()
            init_val = 2*((tensor[0].abs().sum()-tensor[1].abs().sum())/(a-b))/ math.sqrt(Qp) if symmetric \
                else 4*((tensor[0].abs().sum()-tensor[1].abs().sum())/(a-b)) / math.sqrt(Qp)
        elif init_method == 'qkv2':
            init_val = 2*(tensor.abs().sum()/tensor.sign().abs().sum())/ math.sqrt(Qp) if symmetric \
                else 4*(tensor.abs().sum()/tensor.sign().abs().sum() ) / math.sqrt(Qp)
        else:
            init_val = 2*(tensor.abs().sum()/tensor.sign().abs().sum())/ math.sqrt(Qp) if symmetric \
                else 4*(tensor.abs().sum()/tensor.sign().abs().sum() ) / math.sqrt(Qp)
       
        self._initialize(init_val)

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


def act_quant_fn(input, clip_val, num_bits, symmetric, quant_method, layerwise):
    if num_bits == 32:
        return input
    elif quant_method == "bwn" and num_bits == 1:
        quant_fn = BwnQuantizer
    elif quant_method == "elastic" and num_bits >= 1 and symmetric:
        quant_fn = ElasticQuantBinarizerSigned
    elif quant_method == "elastic" and num_bits >= 1 and not symmetric:
        quant_fn = ElasticQuantBinarizerUnsigned
    elif quant_method == "uniform" and num_bits >= 1 and not symmetric:
        quant_fn = ElasticQuantBinarizerUnsigned
    elif quant_method == "uniform" and num_bits >= 1 and symmetric:
        quant_fn = ElasticQuantBinarizerSigned
    else:
        raise ValueError("Unknownquant_method")

    input = quant_fn.apply(input, clip_val, num_bits, layerwise)

    return input

def attention_quant_fn(input,  clip_val, clip_valx, clip_vald,  num_bits, symmetric, quant_method, layerwise):
    if num_bits == 32:
        return input
    elif quant_method == "attention" :
        quant_fn = ElasticQuantAttention
    else:
        raise ValueError("Unknown quant_method")

    input = quant_fn.apply(input, clip_val, clip_valx, clip_vald,  num_bits, layerwise)
    return input

def qkv_quant_fn(input,  clip_val, clip_valx, clip_vald,  num_bits, symmetric, quant_method, layerwise):
    if num_bits == 32:
        return input
    elif quant_method == "qkv" :
        quant_fn = ElasticQuantQKV
    else:
        raise ValueError("Unknown quant_method")

    input = quant_fn.apply(input, clip_val, clip_valx, clip_vald,  num_bits, layerwise)
    return input




def weight_quant_fn(weight,  clip_val,  num_bits,  symmetric, quant_method, layerwise):
    if num_bits == 32:
        return weight
    elif quant_method == "bwn" and num_bits == 1:
        quant_fn = BwnQuantizer
    elif quant_method == "elastic" and num_bits >= 1 and symmetric:
        quant_fn = ElasticQuantBinarizerSigned
    elif quant_method == "elastic" and num_bits >= 1 and not symmetric:
        quant_fn = ElasticQuantBinarizerUnsigned
    else:
        raise ValueError("Unknown quant_method")

    weight = quant_fn.apply(weight, clip_val,  num_bits, layerwise)
    return weight


class QuantizeLinear(nn.Linear):

    def __init__(self, *kargs, clip_val=2.5, weight_bits=8, input_bits=8, learnable=False, symmetric=True,
                 weight_layerwise=True, input_layerwise=True, weight_quant_method="bwn", input_quant_method="uniform",
                 **kwargs):
        super(QuantizeLinear, self).__init__(*kargs, **kwargs)
        self.weight_bits = weight_bits
        self.input_bits = input_bits
        self.learnable = learnable
        self.symmetric = symmetric
        self.weight_layerwise = weight_layerwise
        self.input_layerwise = input_layerwise
        self.weight_quant_method = weight_quant_method
        self.input_quant_method = input_quant_method
        self._build_weight_clip_val(weight_quant_method, learnable, init_val=clip_val)
        self._build_input_clip_val(input_quant_method, learnable, init_val=clip_val)
        self.move = LearnableBias(self.weight.shape[1])

    def _build_weight_clip_val(self, quant_method, learnable, init_val):
        if quant_method == 'uniform':
            # init_val = self.weight.mean().item() + 3 * self.weight.std().item()
            self.register_buffer('weight_clip_val', torch.tensor([-init_val, init_val]))
            if learnable:
                self.weight_clip_val = nn.Parameter(self.weight_clip_val)
        elif quant_method == 'elastic':
            assert learnable, 'Elastic method must use leranable step size!'
            self.weight_clip_val = AlphaInit(torch.tensor(1.0)) # stepsize will be initialized in the first quantization
        else:
            self.register_buffer('weight_clip_val', None)

    def _build_input_clip_val(self, quant_method, learnable, init_val):
        if quant_method == 'uniform':
            self.register_buffer('input_clip_val', torch.tensor([-init_val, init_val]))
            if learnable:
                self.input_clip_val = nn.Parameter(self.input_clip_val)
        elif quant_method == 'elastic' or quant_method == 'bwn':
            assert learnable, 'Elastic method must use leranable step size!'
            self.input_clip_val = AlphaInit(torch.tensor(1.0))  # stepsize will be initialized in the first quantization
        else:
            self.register_buffer('input_clip_val', None)

    def forward(self, input):
        # quantize weight
        weight = weight_quant_fn(self.weight, self.weight_clip_val, num_bits=self.weight_bits, symmetric=self.symmetric,
                                 quant_method=self.weight_quant_method, layerwise=self.weight_layerwise)
        # quantize input
        input = self.move(input)
        input = act_quant_fn(input, self.input_clip_val, num_bits=self.input_bits, symmetric=self.symmetric,
                             quant_method=self.input_quant_method, layerwise=self.input_layerwise)
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






def get_default_kwargs_q(kwargs_q, layer_type):
    default = {
        'nbits': 4
    }
    if isinstance(layer_type, _Conv2dQ):
        default.update({
            'mode': Qmodes.layer_wise})
    else:
        assert NotImplementedError
        return
    for k, v in default.items():
        if k not in kwargs_q:
            kwargs_q[k] = v
    return kwargs_q 

class Qmodes(Enum):
    layer_wise = 1
    kernel_wise = 2
 
class _Conv2dQ(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, **kwargs_q):
        super(_Conv2dQ, self).__init__(in_channels, out_channels, kernel_size, stride=stride,
                                       padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.kwargs_q = get_default_kwargs_q(kwargs_q, layer_type=self)
        self.nbits = kwargs_q['nbits']
        if self.nbits < 0:
            self.register_parameter('alpha', None)
            return
        self.q_mode = kwargs_q['mode']
        if self.q_mode == Qmodes.kernel_wise:
            self.alpha = Parameter(torch.Tensor(out_channels))
        else:  # layer-wise quantization
            self.alpha = Parameter(torch.Tensor(1))
        self.register_buffer('init_state', torch.zeros(1))

    def add_param(self, param_k, param_v):
        self.kwargs_q[param_k] = param_v

    def set_bit(self, nbits):
        self.kwargs_q['nbits'] = nbits

    def extra_repr(self):
        s_prefix = super(_Conv2dQ, self).extra_repr()
        if self.alpha is None:
            return '{}, fake'.format(s_prefix)
        return '{}, {}'.format(s_prefix, self.kwargs_q) 
 



class _ActQ(nn.Module):
    def __init__(self, in_features, **kwargs_q):
        super(_ActQ, self).__init__()
        self.kwargs_q = get_default_kwargs_q(kwargs_q, layer_type=self)
        self.nbits = kwargs_q['nbits']
        if self.nbits < 0:
            self.register_parameter('alpha', None)
            self.register_parameter('zero_point', None)
            return
        # self.signed = kwargs_q['signed']
        self.q_mode = kwargs_q['mode']
        self.alpha = Parameter(torch.Tensor(1))
        self.zero_point = Parameter(torch.Tensor([0]))
        if self.q_mode == Qmodes.kernel_wise:
            self.alpha = Parameter(torch.Tensor(in_features))
            self.zero_point = Parameter(torch.Tensor(in_features))
            torch.nn.init.zeros_(self.zero_point)
        # self.zero_point = Parameter(torch.Tensor([0]))
        self.register_buffer('init_state', torch.zeros(1))
        self.register_buffer('signed', torch.zeros(1))

    def add_param(self, param_k, param_v):
        self.kwargs_q[param_k] = param_v

    def set_bit(self, nbits):
        self.kwargs_q['nbits'] = nbits

    def extra_repr(self):
        # s_prefix = super(_ActQ, self).extra_repr()
        if self.alpha is None:
            return 'fake'
        return '{}'.format(self.kwargs_q)


def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return y.detach() - y_grad.detach() + y_grad


def round_pass(x):
    y = x.round()
    y_grad = x
    return y.detach() - y_grad.detach() + y_grad


class ActQ(_ActQ):
    def __init__(self, in_features, nbits_a=4, mode=Qmodes.kernel_wise, **kwargs):
        super(ActQ, self).__init__(in_features=in_features, nbits=nbits_a, mode=mode)
        # print(self.alpha.shape, self.zero_point.shape)
    def forward(self, x):
        if self.alpha is None:
            return x

        if self.training and self.init_state == 0:
            # The init alpha for activation is very very important as the experimental results shows.
            # Please select a init_rate for activation.
            # self.alpha.data.copy_(x.max() / 2 ** (self.nbits - 1) * self.init_rate)
            if x.min() < -1e-5:
                self.signed.data.fill_(1)
            
            Qn = -2 ** (self.nbits - 1)
            Qp = 2 ** (self.nbits - 1) - 1
            
            self.alpha.data.copy_(2 * x.abs().mean() / math.sqrt(Qp))
            self.zero_point.data.copy_(self.zero_point.data * 0.9 + 0.1 * (torch.min(x.detach()) - self.alpha.data * Qn))
            self.init_state.fill_(1)

        
        Qn = -2 ** (self.nbits - 1)
        Qp = 2 ** (self.nbits - 1) - 1
        

        g = 1.0 / math.sqrt(x.numel() * Qp)

        # Method1:
        zero_point = (self.zero_point.round() - self.zero_point).detach() + self.zero_point
        alpha = grad_scale(self.alpha, g)
        zero_point = grad_scale(zero_point, g)
        # x = round_pass((x / alpha).clamp(Qn, Qp)) * alpha
        if len(x.shape)==2:
            alpha = alpha.unsqueeze(0)
            zero_point = zero_point.unsqueeze(0)
        elif len(x.shape)==4:
            alpha = alpha.unsqueeze(0).unsqueeze(2).unsqueeze(3)
            zero_point = zero_point.unsqueeze(0).unsqueeze(2).unsqueeze(3)

        x = round_pass((x / alpha + zero_point).clamp(Qn, Qp))
        x = (x - zero_point) * alpha

        return x




   
class Conv2dQ(_Conv2dQ):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, nbits_w=8, mode=Qmodes.kernel_wise, **kwargs):
        super(Conv2dQ, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias,
            nbits=nbits_w, mode=mode)
        self.act = ActQ(in_features=in_channels, nbits_a=nbits_w)

    def forward(self, x):
        if self.alpha is None:
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
        """  
        Implementation according to paper. 
        Feels wrong ...
        When we initialize the alpha as a big number (e.g., self.weight.abs().max() * 2), 
        the clamp function can be skipped.
        Then we get w_q = w / alpha * alpha = w, and $\frac{\partial w_q}{\partial \alpha} = 0$
        As a result, I don't think the pseudo-code in the paper echoes the formula.
       
        Please see jupyter/STE_LSQ.ipynb fo detailed comparison.
        """
        g = 1.0 / math.sqrt(self.weight.numel() * Qp)

        # Method1: 31GB GPU memory (AlexNet w4a4 bs 2048) 17min/epoch
        alpha = grad_scale(self.alpha, g)
        # print(alpha.shape)
        # print(self.weight.shape)
        alpha = alpha.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        w_q = round_pass((self.weight / alpha).clamp(Qn, Qp)) * alpha

        x = self.act(x)
        # w = w.clamp(Qn, Qp)
        # q_w = round_pass(w)
        # w_q = q_w * alpha

        # Method2: 25GB GPU memory (AlexNet w4a4 bs 2048) 32min/epoch
        # w_q = FunLSQ.apply(self.weight, self.alpha, g, Qn, Qp)
        # wq = y.transpose(0, 1).reshape(self.weight.shape).detach() + self.weight - self.weight.detach()
        return F.conv2d(x, w_q, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
def sign_pass(x):
    y = x.sign()
    y_grad = x
    return y.detach() - y_grad.detach() + y_grad

class _LinearQ(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, **kwargs_q):
        super(_LinearQ, self).__init__(in_features=in_features, out_features=out_features, bias=bias)
        self.kwargs_q = get_default_kwargs_q(kwargs_q, layer_type=self)
        self.nbits = kwargs_q['nbits']
        if self.nbits < 0:
            self.register_parameter('alpha', None)
            return
        self.q_mode = kwargs_q['mode']
        self.alpha = Parameter(torch.Tensor(1))
        if self.q_mode == Qmodes.kernel_wise:
            self.alpha = Parameter(torch.Tensor(out_features))
        self.register_buffer('init_state', torch.zeros(1))

    def add_param(self, param_k, param_v):
        self.kwargs_q[param_k] = param_v

    def extra_repr(self):
        s_prefix = super(_LinearQ, self).extra_repr()
        if self.alpha is None:
            return '{}, fake'.format(s_prefix)
        return '{}, {}'.format(s_prefix, self.kwargs_q) 
class LinearQ(_LinearQ):
    def __init__(self, in_features, out_features, bias=True, nbits_w=4, **kwargs):
        super(LinearQ, self).__init__(in_features=in_features,
                                        out_features=out_features, bias=bias, nbits=nbits_w, mode=Qmodes.kernel_wise)
        self.act = ActQ(in_features=in_features, nbits_a=nbits_w)

    def forward(self, x):
        if self.alpha is None:
            return F.linear(x, self.weight, self.bias)
        if self.nbits == 1:
            Qn = -1
            Qp = 1        
        else:
            Qn = -2 ** (self.nbits - 1)
            Qp = 2 ** (self.nbits - 1) - 1
        if self.training and self.init_state == 0:
            self.alpha.data.copy_(2 * self.weight.abs().mean() / math.sqrt(Qp))
            # self.alpha.data.copy_(self.weight.abs().max() / 2 ** (self.nbits - 1))
            self.init_state.fill_(1)
        g = 1.0 / math.sqrt(self.weight.numel() * Qp)

        # Method1:
        alpha = grad_scale(self.alpha, g)
        alpha = alpha.unsqueeze(1)
        if self.nbits == 1:
            w_q = sign_pass((self.weight / alpha).clamp(Qn, Qp)) * alpha
        else:
            w_q = round_pass((self.weight / alpha).clamp(Qn, Qp)) * alpha
        x = self.act(x)
        # w = self.weight / alpha
        # w = w.clamp(Qn, Qp)
        # q_w = round_pass(w)
        # w_q = q_w * alpha

        # Method2:
        # w_q = FunLSQ.apply(self.weight, self.alpha, g, Qn, Qp)
        return F.linear(x, w_q, self.bias)
       
        
        
    