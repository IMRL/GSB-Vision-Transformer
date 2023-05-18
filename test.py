# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from pathlib import Path

from timm.models import create_model
from timm.data import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, ModelEma

from datasets import build_dataset
from engine import train_one_epoch, evaluate
from losses import DistillationLoss
from samplers import RASampler
from One_bit_vit_new import *
import utils

class OQ_VitConfig():
    r"""
        :class:`OQ_VitConfig` is the configuration class to store the configuration of a
        `OQ_VitModel`."""
    def __init__(self,
                 clip_init_val=2.5,
                 hidden_size=768//2,
                 layer_norm_eps=1e-12,
                 weight_bits=1,
                 input_bits=1,
                 num_attention_heads=6,
                 **kwargs):
        super(OQ_VitConfig, self).__init__(**kwargs)
        self.clip_init_val = clip_init_val
        self.hidden_size = hidden_size
        self.layer_norm_eps = layer_norm_eps
        self.weight_bits = weight_bits
        self.input_bits = input_bits
        self.num_attention_heads = num_attention_heads
        
def get_args_parser():
    parser = argparse.ArgumentParser('Onebits_DeiT training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    
    # Model parameters
    parser.add_argument('--model', default='onebits_deit_small_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')

    parser.add_argument("--weight_quant_method", default='bwn', type=str,
                        choices=['bwn', 'uniform'],
                        help="weight quantization methods")
    parser.add_argument("--input_quant_method", default='elastic', type=str,
                        help="weight quantization methods")
    parser.add_argument('--not_quantize_attention', action='store_true', help="Keep attention calculations in 32-bit.")

    parser.add_argument('--clip_init_val', default=2.5, type=float, help='init value of clip_vals, default to (-2.5, +2.5).')

    parser.add_argument('--learnable_scaling', action='store_true', default=True)

    parser.add_argument('--sym_quant_qkvo', action='store_true', default=True,
                        help='whether use asym quant for Q/K/V and others.')  # default sym
   
    


    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--model_ema', action='store_true')
    #parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=False)
    #parser.add_argument('--model_ema_decay', type=float, default=0.99996, help='')
    #parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')


    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
    
    # * Finetuning params /opt/data/private/ob_vit/eval/best_checkpoint1_32.pth
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')

    # Dataset parameters /opt/data/common/ImageNet/ILSVRC2012/
    parser.add_argument('--data-path', default='/opt/data/private/ob_vit/dataset', type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='CIFAR', choices=['CIFAR', 'IMNET', 'INAT', 'INAT19','IMNET100'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')

    parser.add_argument('--output_dir', default='/opt/data/private/ob_vit/result_stage2',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    #D:/SEG/ob-vit-imagnet/eval/best_checkpoint69.32zhen.pth
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.add_argument('--dist-eval', action='store_true', default=True, help='Enabling distributed evaluation')
    parser.add_argument('--weight_layerwise', default=True, type=lambda x: bool(int(x)))
    parser.add_argument('--input_layerwise', default=True, type=lambda x: bool(int(x)))

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')

    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--root_dir_train', default='', type=str, help='dataset path')
    parser.add_argument('--meta_file_train', default='', type=str, help='dataset path')
    parser.add_argument('--root_dir_val', default='', type=str, help='dataset path')
    parser.add_argument('--meta_file_val', default='', type=str, help='dataset path')
    return parser


def main(args):
    
    utils.init_distributed_mode(args)
    # utils.setup_distributed()
    
   # print(args)
    config = OQ_VitConfig()
    config.input_size = args.input_size
    config.weight_quant_method = args.weight_quant_method
    config.input_quant_method = args.input_quant_method
    config.clip_init_val = args.clip_init_val
    config.learnable_scaling = args.learnable_scaling
    config.sym_quant_qkvo = args.sym_quant_qkvo
    config.weight_layerwise = args.weight_layerwise
    config.input_layerwise = args.input_layerwise
    config.not_quantize_attention = args.not_quantize_attention
    # if args.distillation_type != 'none' and args.finetune and not args.eval:
    #     raise NotImplementedError("Finetuning with distillation not yet supported")
  
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True
    dataset_val, args.nb_classes = build_dataset(is_train=False, args=args)

    if True: # args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )
    print(f"Creating model: {args.model}")
    model = onebits_deit_small_patch16_224(
        config,
        num_classes=args.nb_classes
    )

    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')

        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        state_dict.update(checkpoint_model)
        # interpolate position embedding
        model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
    
    test_stats = evaluate(data_loader_val, model, device)
    print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
