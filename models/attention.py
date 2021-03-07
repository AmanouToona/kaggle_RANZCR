import gc
import typing as tp
from parts import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import timm
from bottleneck_transformer_pytorch import BottleStack


class SingleHeadModel(nn.Module):
    def __init__(self, base_name: str='resnext50_32x4d', out_dim: int=11, pretrained: bool=False):
        super().__init__()
        self.base_name = base_name

        base_model = timm.create_model(base_name, pretrained=pretrained)
        fc_in_features = base_model.fc.in_features

        base_model.reset_classifier(0)  # base_model.fc = nn.Identity() に等しい

        self.backbone = base_model

        self.head_fc = nn.Sequential(
            nn.Linear(fc_in_features, fc_in_features),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(fc_in_features, out_dim)
        )

    def forward(self, x):
        h = self.backbone(x)
        h = self.head_fc(h)
        return h


# class MultiHeadAttention(nn.Module):
#     def __init__(self, base_name: str='resnext50_32x4d', out_dims: tp.List[int] = [3, 4, 3, 1], pretrained: bool=False):
#         super().__init__()
#         self.base_name = base_name
#         self.n_heads = len(out_dims)
#
#         base_model = timm.create_model(base_name, pretrained=pretrained)
#         fc_in_features = base_model.fc.in_features
#
#         base_model.fc = nn.Identity()
#         base_model.global_pool = nn.Identity()
#
#         self.backbone = base_model
#
#         for i, out_dim in enumerate(out_dims):
#             layer = nn.Sequential(
#                 SpatialAttentionBlock(in_channels=fc_in_features, out_channels_list=[64, 32, 16, 1]),
#                 nn.AdaptiveAvgPool2d(output_size=1),
#                 nn.Flatten(start_dim=1),
#                 nn.Linear(fc_in_features, fc_in_features),
#                 nn.ReLU(inplace=True),
#                 nn.Dropout(0.5),
#                 nn.Linear(fc_in_features, fc_in_features)
#             )
#             setattr(self, f'head_{i}', layer)
#
#     def forward(self, x):
#         x = self.backbone(x)
#         x = [getattr(self, f'head_{i}')(x) for i in range(self.n_heads)]
#         x = torch.cat(x, dim=1)
#         return x


class MultiHeadModel(nn.Module):

    def __init__(
            self, base_name: str = 'resnext50_32x4d',
            out_dims_head: tp.List[int] = [3, 4, 3, 1], pretrained=False):
        """"""
        self.base_name = base_name
        self.n_heads = len(out_dims_head)
        super(MultiHeadModel, self).__init__()

        # # load base model
        base_model = timm.create_model(base_name, pretrained=pretrained)
        in_features = base_model.num_features

        # # remove global pooling and head classifier
        base_model.reset_classifier(0, '')

        # # Shared CNN Bacbone
        self.backbone = base_model

        # # Multi Heads.
        for i, out_dim in enumerate(out_dims_head):
            layer_name = f"head_{i}"
            layer = nn.Sequential(
                SpatialAttentionBlock(in_features, [64, 32, 16, 1]),
                nn.AdaptiveAvgPool2d(output_size=1),
                nn.Flatten(start_dim=1),
                nn.Linear(in_features, in_features),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(in_features, out_dim))
            setattr(self, layer_name, layer)

    def forward(self, x):
        """"""
        h = self.backbone(x)
        hs = [
            getattr(self, f"head_{i}")(h) for i in range(self.n_heads)]
        y = torch.cat(hs, axis=1)
        return y


class MultiHeadAttention(nn.Module):
    def __init__(self, base_name: str = 'resnext50_32x4d', out_dims_head: tp.List[int] = [3, 4, 3, 1], pretrained=False):
        super().__init__()
        self.base_name = base_name
        self.n_heads = len(out_dims_head)

        # # load base model
        base_model = timm.create_model(base_name, pretrained=pretrained)
        in_features = base_model.num_features

        # # remove global pooling and head classifier
        base_model.fc = nn.Identity()
        base_model.global_pool = nn.Identity()

        # # Shared CNN Bacbone
        self.backbone = base_model

        # Multi Head Attention
        bot_layer_1 = BottleStack(
            dim=1024,  # channels in
            fmap_size=32,  # feature map size
            dim_out=2048,  # channels out
            proj_factor=4,  # projection factor
            downsample=True,  # downsample on first layer or not
            heads=4,  # number of heads
            dim_head=128,  # dimension per head, defaults to 128
            rel_pos_emb=True,  # use relative positional embedding - uses absolute if False
            activation=nn.SiLU()  # activation throughout the network
        )

        bot_layer_2 = BottleStack(
            dim=2048,  # channels in
            fmap_size=16,  # feature map size
            dim_out=2048,  # channels out
            proj_factor=4,  # projection factor
            downsample=True,  # downsample on first layer or not
            heads=4,  # number of heads
            dim_head=128,  # dimension per head, defaults to 128
            rel_pos_emb=True,  # use relative positional embedding - uses absolute if False
            activation=nn.SiLU()  # activation throughout the network
        )

        bot_layer_3 = BottleStack(
            dim=2048,  # channels in
            fmap_size=8,  # feature map size
            dim_out=2048,  # channels out
            proj_factor=4,  # projection factor
            downsample=True,  # downsample on first layer or not
            heads=4,  # number of heads
            dim_head=128,  # dimension per head, defaults to 128
            rel_pos_emb=True,  # use relative positional embedding - uses absolute if False
            activation=nn.SiLU()  # activation throughout the network
        )

        BotStackLayer = nn.Sequential(bot_layer_1, bot_layer_2, bot_layer_3)

        self.backbone.s4 = BotStackLayer

        # # Multi Heads.
        for i, out_dim in enumerate(out_dims_head):
            layer_name = f"head_{i}"
            layer = nn.Sequential(
                SpatialAttentionBlock(in_features, [64, 32, 16, 1]),
                nn.AdaptiveAvgPool2d(output_size=1),
                nn.Flatten(start_dim=1),
                nn.Linear(in_features, in_features),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(in_features, out_dim))
            setattr(self, layer_name, layer)

    def forward(self, x):
        """"""
        h = self.backbone(x)
        hs = [
            getattr(self, f"head_{i}")(h) for i in range(self.n_heads)]
        y = torch.cat(hs, axis=1)

        return h


class StMultiHeadModel(nn.Module):
    def __init__(self, base_name: str = 'resnext50_32x4d', out_dims_head: tp.List[int] = [3, 4, 3, 1],
                 pretrained=False, fmap_size: int = 256):
        super().__init__()

        self.multiheadmodel = MultiHeadModel(base_name, out_dims_head, pretrained)

        self.stn = SpatialTransformer(in_channels=3, fmap_size=fmap_size)

    def forward(self, x):
        x = self.stn(x)
        x = x.repeat(1, 3, 1, 1)
        x = self.multiheadmodel(x)
        return x


if __name__ == '__main__':
    size = (3, 256, 256)
    size2 = (1, 3, 256, 256)

    # x = torch.rand(size2)
    # layer = nn.Conv2d(3, 8, kernel_size=7)
    # y = layer(x)
    # print(y.size())

    # m = MultiHeadAttention(pretrained=True)
    # m = MultiHeadModel(pretrained=True)
    # m = StMultiHeadModel(pretrained=True, base_name='regnety_032')
    m = MultiHeadAttention( base_name='regnety_032')
    print(m)
    device = torch.device('cpu')
    m = m.to(device)
    m = m.eval()

    summary(m, size)

    x = torch.rand(size2)
    x = x.to(device)
    with torch.no_grad():
        y = m(x)

    # print('[forward test]')
    # print(f'input: {x.shape}\noutput: {y.shape}')

    del m
    del x
    del y
    _ = gc.collect()

