import yaml
import argparse
import os
from src.utils_general import DictWrapper
import distutils.util
    
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--method",
                        default=argparse.SUPPRESS)
    parser.add_argument("--dataset",
                        default=argparse.SUPPRESS)
    parser.add_argument("--arch",
                        default=argparse.SUPPRESS)
    parser.add_argument("--pretrain",
                        default=None, type=str)

    # vision transformer specific hyperparameters
    parser.add_argument("--vit_patch_size",
                        default=argparse.SUPPRESS, type=int)
    parser.add_argument("--vit_embed_dim",
                        default=argparse.SUPPRESS, type=int)
    parser.add_argument("--vit_depth",
                        default=argparse.SUPPRESS, type=int)
    parser.add_argument("--vit_num_heads",
                        default=argparse.SUPPRESS, type=int)
    parser.add_argument("--vit_mlp_ratio",
                        default=argparse.SUPPRESS, type=float)
    parser.add_argument("--vit_dropout",
                        default=argparse.SUPPRESS, type=float)
    parser.add_argument("--vit_use_cls_token",
                        default=argparse.SUPPRESS, type=lambda x: bool(distutils.util.strtobool(x)))

    # hyper-param for optimization
    parser.add_argument("--optim",
    			default=argparse.SUPPRESS)
    parser.add_argument("--lr",
    			default=argparse.SUPPRESS, type=float)
    parser.add_argument("--lr_scheduler_type",
    			default=argparse.SUPPRESS)
    parser.add_argument("--momentum",
    			default=argparse.SUPPRESS, type=float)
    parser.add_argument("--weight_decay",
    			default=argparse.SUPPRESS, type=float)
    parser.add_argument("--adam_beta1",
    			default=argparse.SUPPRESS, type=float)
    parser.add_argument("--adam_beta2",
    			default=argparse.SUPPRESS, type=float)
    parser.add_argument("--rmsp_alpha",
    			default=argparse.SUPPRESS, type=float)
    parser.add_argument("--batch_size",
    			default=argparse.SUPPRESS, type=int)
    parser.add_argument("--seed",
    			default=argparse.SUPPRESS, type=int)
    parser.add_argument("--epoch",
    			default=argparse.SUPPRESS, type=int)

    parser.add_argument("--j_dir",
    			default='./exp')

    parser.add_argument("--lr_warmup_type",
    			default=argparse.SUPPRESS)
    parser.add_argument("--lr_warmup_epoch",
    			default=argparse.SUPPRESS, type=int)
    parser.add_argument("--lr_warmup_decay",
    			default=argparse.SUPPRESS, type=float)
    parser.add_argument("--lr_end_factor",
    			default=argparse.SUPPRESS, type=float)

    parser.add_argument("--threshold",
    			default=90, type=float)

    parser.add_argument('--input_normalization',
                        default=argparse.SUPPRESS, type=distutils.util.strtobool)
    parser.add_argument('--enable_batchnorm',
                        default=argparse.SUPPRESS, type=distutils.util.strtobool)
    
    parser.add_argument('--eval_only',
                        action="store_true")

    parser.add_argument('--speedrun_arch',
                        action="store_true")

    args = parser.parse_args()

    return args

def get_default(yaml_path):
    default = {}
    with open(yaml_path, 'r') as handle:
        default = yaml.load(handle, Loader=yaml.FullLoader)
    return default 

def get_args():
    args = parse_args()
    cli_args = vars(args)
    default = get_default('options/default.yaml')

    arch_value = cli_args.get('arch', default.get('arch', ''))
    arch_normalized = arch_value.lower() if isinstance(arch_value, str) else arch_value

    if arch_normalized in {"vit", "vit_cifar", "vision_transformer"}:
        vit_overrides = {
            "optim": "adamw",
            "lr": 5e-4,
            "weight_decay": 0.05,
            "adam_beta1": 0.9,
            "adam_beta2": 0.999,
            "lr_scheduler_type": "cosine",
            "lr_warmup_epoch": 5,
            "lr_warmup_type": "linear",
            "lr_warmup_decay": 0.0,
            "lr_end_factor": 0.0,
        }
        for key, value in vit_overrides.items():
            if key not in cli_args:
                default[key] = value

        if "batch_size" not in cli_args:
            default["batch_size"] = 512
        if "epoch" not in cli_args:
            default["epoch"] = 100
        if "input_normalization" not in cli_args:
            default["input_normalization"] = True
        if "vit_use_cls_token" not in cli_args:
            default.setdefault("vit_use_cls_token", True)

    default.update(cli_args)
    args_dict = DictWrapper(default)

    return args_dict
