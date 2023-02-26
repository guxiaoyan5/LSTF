import math
from typing import Tuple, Any

import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as F


def drop_path(x, drop_prob=0., training=False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class TemporalSelfAttention(nn.Module):
    def __init__(self, input_size, output_size, num_heads=4, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert input_size % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = input_size // num_heads
        self.scale = self.head_dim ** -0.5

        self.t_q_conv = nn.Conv2d(input_size, input_size, kernel_size=1, bias=qkv_bias)
        self.t_k_conv = nn.Conv2d(input_size, input_size, kernel_size=1, bias=qkv_bias)
        self.t_v_conv = nn.Conv2d(input_size, input_size, kernel_size=1, bias=qkv_bias)
        self.t_attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(input_size, output_size)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor) -> Tensor:
        """
          @param x : [batch_size, seq_len, num_nodes, input_size]
          @return : [batch_size, seq_len, num_nodes, output_size]
          """
        B, T, N, D = x.shape
        t_q = self.t_q_conv(x.permute(0, 3, 1, 2)).permute(0, 3, 2, 1)
        t_k = self.t_k_conv(x.permute(0, 3, 1, 2)).permute(0, 3, 2, 1)
        t_v = self.t_v_conv(x.permute(0, 3, 1, 2)).permute(0, 3, 2, 1)
        t_q = t_q.reshape(B, N, T, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        t_k = t_k.reshape(B, N, T, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        t_v = t_v.reshape(B, N, T, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)

        t_attn = (t_q @ t_k.transpose(-2, -1)) * self.scale

        t_attn = t_attn.softmax(dim=-1)
        t_attn = self.t_attn_drop(t_attn)

        t_x = (t_attn @ t_v).transpose(2, 3).reshape(B, N, T, D).transpose(1, 2)

        x = self.proj(t_x)
        x = self.proj_drop(x)
        return x


class SpatioSelfAttention(nn.Module):
    def __init__(self, input_size, output_size, num_heads=4, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = input_size // num_heads
        self.scale = self.head_dim ** -0.5
        self.q_conv = nn.Conv2d(input_size, input_size, kernel_size=1, bias=qkv_bias)
        self.k_conv = nn.Conv2d(input_size, input_size, kernel_size=1, bias=qkv_bias)
        self.v_conv = nn.Conv2d(input_size, input_size, kernel_size=1, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(input_size, output_size)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor, mask=None, **kwargs) -> Tuple[Tensor, Tensor]:
        """
        @param x:  [batch_size, seq_len, num_nodes, input_size]
        @param mask: [batch_size, seq_len, num_nodes, output_size]
        @return:
        """
        B, T, N, D = x.shape
        q = self.q_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        k = self.k_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        v = self.v_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        q = q.reshape(B, T, N, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        k = k.reshape(B, T, N, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        v = v.reshape(B, T, N, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn.masked_fill_(mask, float("-inf"))

        # if self.training:
        #     random_noise = torch.empty_like(attn).uniform_(1e-10, 1 - 1e-10)
        #     random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
        #
        #     attn = ((attn + random_noise) / 1).sigmoid()
        #     r = self.get_r(**kwargs)
        #     structure_kl_loss = (attn * torch.log(attn / r + 1e-6) + (
        #             1 - attn) * torch.log(
        #         (1 - attn) / (1 - r + 1e-6) + 1e-6)).mean()
        # else:
        #     attn = attn.sigmoid()
        #     structure_kl_loss = 0
        attn = attn.softmax(dim=-1)
        structure_kl_loss = 0
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(2, 3).reshape(B, N, T, D).transpose(1, 2)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x, structure_kl_loss

    @staticmethod
    def get_r(decay_interval, decay_r, current_epoch, init_r=0.9, final_r=0.5):
        r = init_r - current_epoch // decay_interval * decay_r
        if r < final_r:
            r = final_r
        return r


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=100):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_dim).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        @param x : [batch_size, seq_len, num_nodes, embed_dim]
        @return : [batch_size, seq_len, num_nodes, embed_dim]
        """
        return self.pe[:, :x.size(1)].unsqueeze(2).expand_as(x).detach()


class TokenEmbedding(nn.Module):
    def __init__(self, input_size, embed_dim, norm_layer=None):
        super().__init__()
        self.token_embed = nn.Linear(input_size, embed_dim, bias=True)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """
        @param x : [batch_size, seq_len, num_nodes, input_size]
        @return : [batch_size, seq_len, num_nodes, input_size]
        """
        x = self.token_embed(x)
        x = self.norm(x)
        return x


class DataEmbedding(nn.Module):
    def __init__(self, input_size, embed_dim, max_len=100, dropout=0.):
        super().__init__()
        self.tokenEmbedding = TokenEmbedding(input_size, embed_dim)
        self.positionEncoding = PositionalEncoding(embed_dim, max_len=max_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """
        @param x : [batch_size, seq_len, num_nodes, input_size]
        @return : [batch_size, seq_len, num_nodes, embed_dim]
        """
        x = self.tokenEmbedding(x)
        x = x + self.positionEncoding(x)
        return self.dropout(x)


class Mlp(nn.Module):
    def __init__(self, in_feature, hidden_feature, out_feature, drop=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_feature, hidden_feature)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_feature, out_feature)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class STEncoderBlock(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, mlp_hidden_dim, t_num_heads=4, geo_num_heads=4,
                 sem_num_heads=4, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.temporalSelfAttention = TemporalSelfAttention(input_size, hidden_size, num_heads=t_num_heads,
                                                           qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop)
        self.geoSelfAttention = SpatioSelfAttention(hidden_size, hidden_size, num_heads=geo_num_heads,
                                                    qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop)
        self.semSelfAttention = SpatioSelfAttention(hidden_size, hidden_size, num_heads=sem_num_heads,
                                                    qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop)

        self.norm = nn.LayerNorm(output_size)
        self.residual = nn.Conv2d(input_size, output_size, 1)
        self.mlp = Mlp(in_feature=2 * hidden_size, hidden_feature=mlp_hidden_dim, out_feature=output_size)

    def forward(self, x, geo_mask=None, sem_mask=None, **kwargs):
        residual = x
        x = self.temporalSelfAttention(x)
        geo, structure_kl_loss1 = self.geoSelfAttention(x, geo_mask, **kwargs)
        sem, structure_kl_loss2 = self.semSelfAttention(x, sem_mask, **kwargs)
        x = torch.cat((geo, sem), dim=-1)
        x = self.mlp(x)
        residual = self.norm(self.residual(residual.permute(0, 3, 1, 2)).permute(0, 2, 3, 1) + x)
        return residual, structure_kl_loss1 + structure_kl_loss2


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, embed_dim, mlp_hidden_dim, t_num_heads=4, geo_num_heads=4,
                 sem_num_heads=4, blocks=3, dropout=0., seq_len=12, pred_len=12):
        super().__init__()
        self.dataEmbedding = DataEmbedding(input_size, embed_dim, dropout=dropout)
        self.blocks = nn.ModuleList()
        for i in range(blocks):
            input_ = embed_dim if i == 0 else hidden_size
            self.blocks.append(
                STEncoderBlock(input_, hidden_size, hidden_size, mlp_hidden_dim, t_num_heads, geo_num_heads,
                               sem_num_heads,
                               attn_drop=dropout, proj_drop=dropout))
        self.end_conv1 = nn.Conv2d(
            in_channels=seq_len, out_channels=pred_len, kernel_size=1, bias=True,
        )
        self.end_conv2 = nn.Conv2d(
            in_channels=hidden_size * blocks, out_channels=output_size, kernel_size=1, bias=True,
        )

    def forward(self, x: Tensor, geo_mask=None, sem_mask=None, **kwargs) -> Tuple[Any, int]:
        """
        @param sem_mask:
        @param geo_mask:
        @param x:[batch_size,seq_len,num_nodes,input_sizes]
        @return: [batch_size,seq_len,num_nodes,output_size]
        """
        x = self.dataEmbedding(x)
        x_list = []
        structure_kl_loss = 0
        for block in self.blocks:
            x, kl = block(x, geo_mask, sem_mask, **kwargs)
            x_list.append(x)
            structure_kl_loss += kl
        x = torch.cat(x_list, dim=-1)
        x = F.relu(self.end_conv1(x).permute(0, 3, 1, 2))
        x = self.end_conv2(x).permute(0, 2, 3, 1)
        return x, structure_kl_loss
