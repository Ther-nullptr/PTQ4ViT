import quant_layers
import utils
from quant_layers import *
from utils import *
import torch

if __name__ == '__main__':
    net = torch.load('/root/kyzhang/yjwang/PTQ4ViT/checkpoints/a_vit_small_patch16_224_PTQ4ViT_8_8_32.pth')
    print(net.keys())