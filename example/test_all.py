from timm.models.layers import config
from torch.nn.modules import module
from test_vit import *
from quant_layers.conv import MinMaxQuantConv2d
from quant_layers.linear import MinMaxQuantLinear, PTQSLQuantLinear
from quant_layers.matmul import MinMaxQuantMatMul, PTQSLQuantMatMul
import matplotlib.pyplot as plt
from utils.net_wrap import wrap_certain_modules_in_net
from tqdm import tqdm
import torch.nn.functional as F
import pickle as pkl
from itertools import product
import types
from utils.quant_calib import HessianQuantCalibrator, QuantCalibrator
from utils.models import get_net
import time
import wandb


def test_all(orig_ckpt_dir, ckpt_path, no_eval, base_name, name, cfg_modifier=lambda x: x, calib_size=32, config_name="PTQ4ViT"):
    quant_cfg = init_config(config_name)
    quant_cfg = cfg_modifier(quant_cfg)

    if orig_ckpt_dir != None:
        net = get_net(base_name)
        state_dict_path = os.path.join(orig_ckpt_dir, f'{name}.pth')
        state_dict = torch.load(state_dict_path)
        net.load_state_dict(state_dict)
        print(f'load state dict from {state_dict_path}')
    else:
        net = get_net(name)

    wrapped_modules = net_wrap.wrap_modules_in_net(net, quant_cfg)

    g = datasets.ViTImageNetLoaderGenerator(
        'datasets/imagenet', 'imagenet', 32, 32, 16, kwargs={"model": net})
    test_loader = g.test_loader()
    calib_loader = g.calib_loader(num=calib_size)

    # add timing
    calib_start_time = time.time()
    quant_calibrator = HessianQuantCalibrator(
        net, wrapped_modules, calib_loader, sequential=False, batch_size=4)  # 16 is too big for ViT-L-16
    quant_calibrator.batching_quant_calib()
    calib_end_time = time.time()
    # TODO: save ckpt of calib models
    ckpt_full_name = os.path.join(
        ckpt_path, f'{name}_{config_name}_{str(quant_cfg.bit[0])}_{str(quant_cfg.bit[1])}_{calib_size}.pth')
    torch.save(net.state_dict(), ckpt_full_name)

    if no_eval:
        return

    acc = test_classification(
        net, test_loader, description=quant_cfg.ptqsl_linear_kwargs["metric"])

    wandb.init(project='PTQ4ViT',
               name=f'{name}_{config_name}_{str(quant_cfg.bit)}_{calib_size}', reinit=True, entity="ther")
    print(f"model: {name} \n")
    print(f"calibration size: {calib_size} \n")
    print(f"bit settings: {quant_cfg.bit} \n")
    print(f"config: {config_name} \n")
    print(f"ptqsl_conv2d_kwargs: {quant_cfg.ptqsl_conv2d_kwargs} \n")
    print(f"ptqsl_linear_kwargs: {quant_cfg.ptqsl_linear_kwargs} \n")
    print(f"ptqsl_matmul_kwargs: {quant_cfg.ptqsl_matmul_kwargs} \n")
    print(f"calibration time: {(calib_end_time-calib_start_time)/60}min \n")
    print(f"accuracy: {acc} \n\n")
    wandb.log({'acc': acc, 'time': (calib_end_time-calib_start_time)/60})


class cfg_modifier():
    def __init__(self, **kwargs):
        for name, value in kwargs.items():
            setattr(self, name, value)

    def __call__(self, cfg):
        # bit setting
        cfg.bit = self.bit_setting
        cfg.w_bit = {name: self.bit_setting[0]
                     for name in cfg.conv_fc_name_list}
        cfg.a_bit = {name: self.bit_setting[1]
                     for name in cfg.conv_fc_name_list}
        cfg.A_bit = {name: self.bit_setting[1]
                     for name in cfg.matmul_name_list}
        cfg.B_bit = {name: self.bit_setting[1]
                     for name in cfg.matmul_name_list}

        # conv2d configs
        cfg.ptqsl_conv2d_kwargs["n_V"] = self.linear_ptq_setting[0]
        cfg.ptqsl_conv2d_kwargs["n_H"] = self.linear_ptq_setting[1]
        cfg.ptqsl_conv2d_kwargs["metric"] = self.metric
        cfg.ptqsl_conv2d_kwargs["init_layerwise"] = False

        # linear configs
        cfg.ptqsl_linear_kwargs["n_V"] = self.linear_ptq_setting[0]
        cfg.ptqsl_linear_kwargs["n_H"] = self.linear_ptq_setting[1]
        cfg.ptqsl_linear_kwargs["n_a"] = self.linear_ptq_setting[2]
        cfg.ptqsl_linear_kwargs["metric"] = self.metric
        cfg.ptqsl_linear_kwargs["init_layerwise"] = False

        # matmul configs
        cfg.ptqsl_matmul_kwargs["metric"] = self.metric
        cfg.ptqsl_matmul_kwargs["init_layerwise"] = False

        return cfg


if __name__ == '__main__':
    args = parse_args()

    names = [
        # "vit_tiny_patch16_224",
        # "vit_small_patch32_224",
        # "vit_small_patch16_224",
        # "vit_base_patch16_224",
        # "vit_base_patch16_384",

        # "deit_tiny_patch16_224",
        # "deit_small_patch16_224",
        # "deit_base_patch16_224",
        # "deit_base_patch16_384",

        # "swin_tiny_patch4_window7_224",
        # "swin_small_patch4_window7_224",
        # "swin_base_patch4_window7_224",
        # "swin_base_patch4_window12_384",
        # "a_vit_small_patch16_224"
    ]

    names.append(args.name)

    metrics = ["hessian"]
    linear_ptq_settings = [(1, 1, 1)]  # n_V, n_H, n_a
    calib_sizes = [32]
    bit_settings = [(8, 8), (6, 6)]  # weight, activation
    config_names = ["PTQ4ViT", "BasePTQ"]

    cfg_list = []
    for name, metric, linear_ptq_setting, calib_size, bit_setting, config_name in product(names, metrics, linear_ptq_settings, calib_sizes, bit_settings, config_names):
        cfg_list.append({
            "name": name,
            "cfg_modifier": cfg_modifier(linear_ptq_setting=linear_ptq_setting, metric=metric, bit_setting=bit_setting),
            "calib_size": calib_size,
            "config_name": config_name
        })

    if args.multiprocess:
        multiprocess(test_all, cfg_list, n_gpu=args.n_gpu)
    else:
        for cfg in cfg_list:
            if hasattr(args, 'original_checkpoint_dir'):
                original_checkpoint_dir = args.original_checkpoint_dir
            else:
                original_checkpoint_dir = None
            test_all(original_checkpoint_dir, './checkpoints', args.no_eval, args.base_name, **cfg)
