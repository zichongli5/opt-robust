import random
import os
import operator as op
import warnings
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from models import PreActResNet18, vit_cifar
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torch.autograd import grad


hyp = {
    'opt': {
        'train_epochs': 9.9,
        'batch_size': 1024,
        'lr': 11.5,                 # learning rate per 1024 examples
        'momentum': 0.85,
        'weight_decay': 0.0153,     # weight decay per 1024 examples (decoupled from learning rate)
        'bias_scaler': 64.0,        # scales up learning rate (but not weight decay) for BatchNorm biases
        'label_smoothing': 0.2,
        'whiten_bias_epochs': 3,    # how many epochs to train the whitening layer bias before freezing
    },
    'aug': {
        'flip': True,
        'translate': 2,
    },
    'net': {
        'widths': {
            'block1': 64,
            'block2': 256,
            'block3': 256,
        },
        'batchnorm_momentum': 0.6,
        'scaling_factor': 1/9,
        'tta_level': 2,         # the level of test-time augmentation: 0=none, 1=mirror, 2=mirror+translate
    },
}

#############################################
#            Network Components             #
#############################################

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Mul(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale
    def forward(self, x):
        return x * self.scale

class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, momentum, eps=1e-12,
                 weight=False, bias=True):
        super().__init__(num_features, eps=eps, momentum=1-momentum)
        self.weight.requires_grad = weight
        self.bias.requires_grad = bias
        # Note that PyTorch already initializes the weights to one and bias to zero

class Conv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding='same', bias=False):
        super().__init__(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias)

    def reset_parameters(self):
        super().reset_parameters()
        if self.bias is not None:
            self.bias.data.zero_()
        w = self.weight.data
        torch.nn.init.dirac_(w[:w.size(1)])

class ConvGroup(nn.Module):
    def __init__(self, channels_in, channels_out, batchnorm_momentum):
        super().__init__()
        self.conv1 = Conv(channels_in,  channels_out)
        self.pool = nn.MaxPool2d(2)
        self.norm1 = BatchNorm(channels_out, batchnorm_momentum)
        self.conv2 = Conv(channels_out, channels_out)
        self.norm2 = BatchNorm(channels_out, batchnorm_momentum)
        self.activ = nn.GELU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.norm1(x)
        x = self.activ(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activ(x)
        return x

#############################################
#            Network Definition             #
#############################################

def make_net94(widths=hyp['net']['widths'], batchnorm_momentum=hyp['net']['batchnorm_momentum']):
    whiten_kernel_size = 2
    whiten_width = 2 * 3 * whiten_kernel_size**2
    net = nn.Sequential(
        Conv(3, whiten_width, whiten_kernel_size, padding=0, bias=True),
        nn.GELU(),
        ConvGroup(whiten_width,     widths['block1'], batchnorm_momentum),
        ConvGroup(widths['block1'], widths['block2'], batchnorm_momentum),
        ConvGroup(widths['block2'], widths['block3'], batchnorm_momentum),
        nn.MaxPool2d(3),
        Flatten(),
        nn.Linear(widths['block3'], 10, bias=False),
        Mul(hyp['net']['scaling_factor']),
    )
    net[0].weight.requires_grad = False
    net = net.cuda()
    net = net.to(memory_format=torch.channels_last)
    for mod in net.modules():
        if isinstance(mod, BatchNorm):
            mod.float()
    return net

def seed_everything(manual_seed):
    # set benchmark to False for EXACT reproducibility
    # when benchmark is true, cudnn will run some tests at
    # the beginning which determine which cudnn kernels are
    # optimal for opertions
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed(manual_seed)
    np.random.seed(manual_seed)
    os.environ['PYTHONHASHSEED'] = str(manual_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def get_model(args):
    if args.dataset == 'cifar100':
        num_classes = 100
        img_size = 32
    elif args.dataset == 'cifar10':
        num_classes = 10
        img_size = 32
    else:
        raise ValueError(f"Unsupported dataset '{args.dataset}' for model creation.")

    if args.speedrun_arch:
        model = make_net94()
        return model

    arch = getattr(args, "arch", "")
    arch = arch.lower() if isinstance(arch, str) else arch

    if arch in {"vit", "vit_cifar", "vision_transformer"}:
        model = vit_cifar(
            num_classes=num_classes,
            img_size=img_size,
            patch_size=getattr(args, "vit_patch_size", 4),
            embed_dim=getattr(args, "vit_embed_dim", 256),
            depth=getattr(args, "vit_depth", 6),
            num_heads=getattr(args, "vit_num_heads", 8),
            mlp_ratio=getattr(args, "vit_mlp_ratio", 4.0),
            dropout=getattr(args, "vit_dropout", 0.0),
            attention_dropout=getattr(args, "vit_attention_dropout", 0.0),
            input_normalization=args.input_normalization,
        )
    else:
        model = PreActResNet18(args.dataset, num_classes, args.input_normalization, args.enable_batchnorm)

    for n, p in model.named_parameters():
        print(n, p.size())
    if args.pretrain:
        model.load_state_dict(torch.load(args.pretrain))
        print('Checkpoint Loaded at {}.'.format(args.pretrain))

    return model

def get_optim(parameters, args, np):

    if "sgd" in args.optim:
        opt = optim.SGD(parameters, lr=args.lr, momentum=0.85, weight_decay=args.weight_decay, nesterov=True)
    elif "adam" in args.optim:
        opt = optim.Adam(parameters, lr=args.lr, betas=(args.adam_beta1, args.adam_beta2))
    elif "rmsprop" in args.optim:
        opt = optim.RMSprop(parameters, lr=args.lr, alpha=args.rmsp_alpha, weight_decay=args.weight_decay, momentum=args.momentum, centered=False)
    elif "muon" in args.optim:
        from src.muon import SingleDeviceMuonWithAuxAdam
        hidden_matrix_params = [p for n, p in np if p.ndim == 4 and p.requires_grad]
        other_params = [p for n, p in np if p.ndim != 4 and p.requires_grad]
        adam_groups = dict(params=other_params, lr=0.005, betas=(args.adam_beta1, args.adam_beta2), use_muon=False)
        muon_group = dict(params=hidden_matrix_params, lr=args.lr, momentum=args.adam_beta1, use_muon=True)
        param_groups = [adam_groups, muon_group]
        opt = SingleDeviceMuonWithAuxAdam(param_groups)


    if args.lr_scheduler_type == "multistep":
        _milestones = [args.epoch/ 2, args.epoch * 3 / 4]
        print(_milestones)
        main_lr_scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=_milestones, gamma=0.1)
    elif args.lr_scheduler_type == 'cosine':
        main_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epoch - args.lr_warmup_epoch, eta_min=0.)
    elif args.lr_scheduler_type == "fixed":
        main_lr_scheduler = None
    else:
        raise ValueError('invalid lr_schduler=%s' % args.lr_scheduler_type)

    if args.lr_warmup_epoch > 0:
        if args.lr_warmup_type == 'linear':
            warmup_lr_scheduler = optim.lr_scheduler.LinearLR(
                    opt, start_factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epoch)
        elif args.lr_warmup_type == 'speedrun':
            def get_lr(step):
                total_train_steps = args.epoch * (50000 // args.batch_size)
                warmup_steps = int(total_train_steps * 0.2)
                warmdown_steps = total_train_steps - warmup_steps
                if step < warmup_steps:
                    frac = step / warmup_steps
                    return 0.2 * (1 - frac) + 1.0 * frac
                else:
                    frac = (step - warmup_steps) / warmdown_steps
                    return 1.0 * (1 - frac) + 0.1 * frac
            warmup_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(opt, get_lr)

        elif args.lr_warmup_type == 'constant':
            warmup_lr_scheduler = optim.lr_scheduler.ConstantLR(
                    opt, factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epoch)
        else:
            raise RuntimeError(
                    f"Invalid warmup lr method '{args.lr_warmup_method}'. Only linear and constant are supported."
            )
        lr_scheduler = optim.lr_scheduler.SequentialLR(
                opt, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[args.lr_warmup_epoch]
                )
        # lr_scheduler = warmup_lr_scheduler
    else:
        lr_scheduler = main_lr_scheduler

    return opt, lr_scheduler

class DictWrapper(object):
    def __init__(self, d):
        self.d = d

    def __getattr__(self, key):
        return self.d[key]
