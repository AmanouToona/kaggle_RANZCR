# deep learning
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

import pandas as pd
import numpy as np
from tqdm import tqdm


# model = timm.create_model('resnet200d_320', num_classes=11)
# print(model)
# print(model.num_features)
from torchsummary import summary

import math
import torch
from torch import nn, einsum

from einops import rearrange


def pair(x):
    return (x, x) if not isinstance(x, tuple) else x

def expand_dim(t, dim, k):
    t = t.unsqueeze(dim = dim)
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)

def rel_to_abs(x):
    b, h, l, _, device, dtype = *x.shape, x.device, x.dtype
    dd = {'device': device, 'dtype': dtype}
    col_pad = torch.zeros((b, h, l, 1), **dd)
    x = torch.cat((x, col_pad), dim = 3)
    flat_x = rearrange(x, 'b h l c -> b h (l c)')
    flat_pad = torch.zeros((b, h, l - 1), **dd)
    flat_x_padded = torch.cat((flat_x, flat_pad), dim = 2)
    final_x = flat_x_padded.reshape(b, h, l + 1, 2 * l - 1)
    final_x = final_x[:, :, :l, (l-1):]
    return final_x

def relative_logits_1d(q, rel_k):
    b, heads, h, w, dim = q.shape
    logits = einsum('b h x y d, r d -> b h x y r', q, rel_k)
    print(f'logit shape : {logits.size()}')
    logits = rearrange(logits, 'b h x y r -> b (h x) y r')
    logits = rel_to_abs(logits)
    logits = logits.reshape(b, heads, h, w, w)
    logits = expand_dim(logits, dim = 3, k = h)
    print(f'logit shape: {logits.size()}')
    return logits


class AbsPosEmb(nn.Module):
    def __init__(
        self,
        fmap_size,
        dim_head
    ):
        super().__init__()
        height, width = pair(fmap_size)
        scale = dim_head ** -0.5
        self.height = nn.Parameter(torch.randn(height, dim_head) * scale)
        print(self.height.size())

        self.width = nn.Parameter(torch.randn(width, dim_head) * scale)

    def forward(self, q):
        emb = rearrange(self.height, 'h d -> h () d') + rearrange(self.width, 'w d -> () w d')
        emb = rearrange(emb, ' h w d -> (h w) d')
        logits = einsum('b h i d, j d -> b h i j', q, emb)
        return logits


class RelPosEmb(nn.Module):
    def __init__(
        self,
        fmap_size,
        dim_head
    ):
        super().__init__()
        height, width = pair(fmap_size)
        scale = dim_head ** -0.5
        self.fmap_size = fmap_size
        self.rel_height = nn.Parameter(torch.randn(height * 2 - 1, dim_head) * scale)
        print(f'rel_height size: {self.rel_height.size()}')
        self.rel_width = nn.Parameter(torch.randn(width * 2 - 1, dim_head) * scale)

    def forward(self, q):
        h, w = pair(self.fmap_size)

        q = rearrange(q, 'b h (x y) d -> b h x y d', x = h, y = w)
        rel_logits_w = relative_logits_1d(q, self.rel_width)
        rel_logits_w = rearrange(rel_logits_w, 'b h x i y j-> b h (x y) (i j)')

        q = rearrange(q, 'b h x y d -> b h y x d')
        rel_logits_h = relative_logits_1d(q, self.rel_height)
        rel_logits_h = rearrange(rel_logits_h, 'b h x i y j -> b h (y x) (j i)')
        return rel_logits_w + rel_logits_h

# classes


class Attention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        fmap_size,
        heads = 4,
        dim_head = 128,
        rel_pos_emb = False
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head

        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, 1, bias = False)

        rel_pos_class = AbsPosEmb if not rel_pos_emb else RelPosEmb
        self.pos_emb = rel_pos_class(fmap_size, dim_head)

    def forward(self, fmap):
        heads, b, c, h, w = self.heads, *fmap.shape

        q, k, v = self.to_qkv(fmap).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h d) x y -> b h (x y) d', h = heads), (q, k, v))
        print(f'q size: {q.size()}')

        q *= self.scale

        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        print(f'sim size: {sim.size()}')
        sim += self.pos_emb(q)

        attn = sim.softmax(dim = -1)
        print(f'attn size: {attn.size()}')

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        print(f'out size: {out.size()}')
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        print(f'out size: {out.size()}')
        return out


device = torch.device('cuda:0')
# m = Attention(dim=3, fmap_size=32, rel_pos_emb=True)
m = timm.create_model(model_name='regnety_032')
# m = timm.create_model(model_name='resnext50_32x4d')

m = m.to(device)

# print(m)
# print('*' * 100)
# print(m.s4.b1.conv1.conv)

# size = (10, 3, 32, 32)
size = (3, 256, 256)

summary(m, size)

# x = torch.rand(size)
# x = x.to(device)
# with torch.no_grad():
#     y = m(x)


