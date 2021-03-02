import typing as tp

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_activation(activation_name: str='relu'):
    act_dict = {
        'relu': nn.ReLU(inplace=True),
        'tanh': nn.Tanh(),
        'sigmoid': nn.Sigmoid()
    }
    if activation_name in act_dict:
        return act_dict[activation_name]
    else:
        raise NotImplementedError


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super().__init__()

        self.avgpool2d = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction, bias=False)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.siz()  # x: [B, C, H, W]

        s = self.avgpool2d(x)  # s: [B, C, 1, 1]
        s = torch.flatten(s, 1)  # s: [B, C, 1]
        s = self.fc1(s)  # s: [B, C // reduction, 1]
        s = F.relu(s, inplace=True)
        s = self.fc2(s)  # s: [B, C, 1]
        s = self.sigmoid(s)
        s = s.view(b, c, 1, 1)  # S:[B, C, 1, 1]

        x = x * s
        return x


class Conv2dBNActive(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
            kernel_size: int, stride: int = 1, padding: int = 0,
            bias: bool = False, use_bn: bool = True, active: str = "relu"):
        super().__init__()
        self.use_bn = use_bn

        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(active)

    def forward(self, x):
        x = self.conv2d(x)
        if self.use_bn:
            x = self.bn(x)
        x = self.activation(x)

        return x


class SpatialAttentionBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels_list: tp.List[int]):
        super().__init__()

        self.n_layers = len(out_channels_list)
        channels_list = [in_channels] + out_channels_list

        assert self.n_layers > 0
        assert channels_list[-1] == 1  # mask is 1ch

        for i in range(self.n_layers - 1):
            in_chs, out_chs = channels_list[i: i + 2]
            layer = Conv2dBNActive(in_chs, out_chs, 3, 1, 1, active='relu')
            setattr(self, f'conv{i + 1}', layer)

        in_chs, out_chs = channels_list[-2:]
        layer = Conv2dBNActive(in_chs, out_chs, 3, 1, 1, active='sigmoid')
        setattr(self, f'conv{self.n_layers}', layer)

    def forward(self, x):
        h = x
        for i in range(self.n_layers):
            h = getattr(self, f'conv{i + 1}')(h)
        h = h * x
        return h


class AttentionBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()

        self.conv2d = nn.Conv2d(in_channels, 1, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        mask = self.conv2d(x)
        mask = self.activation(mask)

        return mask

