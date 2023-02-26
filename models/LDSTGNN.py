import math
from typing import List, Tuple, Any, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
import numpy as np


class SScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super().__init__()
        self.d_k = d_k

    def forward(self, Q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        """
        :param Q: batch_size,n_heads, num_nodes, self.d_k
        :param K: batch_size,n_heads, num_nodes, self.d_k
        :return: batch_size,n_heads, num_nodes, num_nodes
        """
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        return scores


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super().__init__()
        self.d_k = d_k

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, res_att: torch.Tensor) -> Tuple[Tensor, Any]:
        """
        :param Q: batch_size, feature_size,n_heads, seq_len,  d_k
        :param K: batch_size, feature_size,n_heads, seq_len,  d_k
        :param V: batch_size, feature_size,n_heads, seq_len,  d_V
        :param res_att: batch_size, feature_size,n_heads, seq_len,  d_k
        :return: [batch_size, feature_size,n_heads, seq_len,  d_V] [batch_size, feature_size,n_heads, seq_len, seq_len]
        """
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k) + res_att
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)
        return context, scores


class SMultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.sScaledDotProductAttention = SScaledDotProductAttention(self.d_k)

    def forward(self, input_Q: torch.Tensor, input_K: torch.Tensor) -> torch.Tensor:
        """
        :param input_Q:[batch_size,num_node,d_model]
        :param input_K:[batch_size,num_node,d_model]
        :return:
        """
        residual = input_Q
        batch_size, num_nodes = input_Q.shape[0], input_Q.shape[1]
        Q = self.W_Q(input_Q).view(batch_size, num_nodes, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(input_K).view(batch_size, num_nodes, self.n_heads, self.d_k).transpose(1, 2)
        attn = self.sScaledDotProductAttention(Q, K)
        return attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model)
        self.scaledDotProductAttention = ScaledDotProductAttention(self.d_k)
        # self.norm = nn.LayerNorm(self.d_model)
        self.tanh = nn.Tanh()

    def forward(self, input_Q: torch.Tensor, input_K: torch.Tensor, input_V: torch.Tensor, res_att: torch.Tensor) -> \
            Tuple[Any, Any]:
        """
        :param input_Q:  [batch_size,feature_size,seq_len,d_model=num_nodes]
        :param input_K:  [batch_size,feature_size,seq_len,d_model=num_nodes]
        :param input_V:  [batch_size,feature_size,seq_len,d_model=num_nodes]
        :param res_att: [batch_size,feature_size,n_heads,seq_len,seq_len]
        :return:
        """
        batch_size, feature_size, seq_len = input_Q.shape[0], input_Q.shape[1], input_Q.shape[2]
        residual = input_Q
        Q = self.W_Q(input_Q).view(batch_size, feature_size, seq_len, self.n_heads, self.d_k).transpose(2, 3)
        K = self.W_K(input_K).view(batch_size, feature_size, seq_len, self.n_heads, self.d_k).transpose(2, 3)
        V = self.W_V(input_V).view(batch_size, feature_size, seq_len, self.n_heads, self.d_v).transpose(2, 3)
        context, res_attn = self.scaledDotProductAttention(Q, K, V, res_att)
        context = context.transpose(-2, -3).reshape(batch_size, feature_size, seq_len, self.n_heads * self.d_v)
        output = self.tanh(self.fc(context)) + residual
        return output, res_attn


class GraphConvWithSAT(nn.Module):
    def __init__(self, K, input_size, output_size, num_nodes):
        super().__init__()
        self.K = K
        self.input_size = input_size
        self.output_size = output_size
        self.num_nodes = num_nodes
        self.relu = nn.Tanh()
        self.Theta = nn.ParameterList([nn.Parameter(torch.FloatTensor(input_size, output_size)) for _ in range(K)])
        self.linear = nn.Linear(self.K * self.output_size, self.output_size)

    def forward(self, x: torch.Tensor, spatial_attention: torch.Tensor, **kwargs) -> Tuple[Any, Union[int, Any]]:
        """
        :param x:[B,N,F,T]
        :param spatial_attention: [B,H,N,N]
        :return:[B,N,output_size,T]
        """
        batch_size, num_node, _, T = x.shape
        outputs = []
        structure_kl_loss_sum = 0

        for k in range(self.K):
            myspatial_attention = spatial_attention[:, k, :, :]
            if self.training:
                random_noise = torch.empty_like(myspatial_attention).uniform_(1e-10, 1 - 1e-10)
                random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
                myspatial_attention = ((myspatial_attention + random_noise) / 1).softmax(dim=-1)
                r = self.get_r(**kwargs)
                structure_kl_loss = (myspatial_attention * torch.log(myspatial_attention / r + 1e-6) + (
                        1 - myspatial_attention) * torch.log(
                    (1 - myspatial_attention) / (1 - r + 1e-6) + 1e-6)).mean()
            else:
                myspatial_attention = myspatial_attention.softmax(dim=-1)
                structure_kl_loss = 0
            structure_kl_loss_sum += structure_kl_loss
            theta_k = self.Theta[k]
            output = []
            for t in range(T):
                graph_signal = x[:, :, :, t]  # (b, N, F_in)
                rhs = myspatial_attention.permute(0, 2, 1).matmul(graph_signal)
                output.append(rhs.matmul(theta_k))
            output = torch.stack(output, dim=-1)
            outputs.append(output)
        outputs = torch.stack(outputs, dim=-1).permute(0, 1, 3, 2, 4).reshape(batch_size, num_node, T, -1)
        outputs = self.relu(self.linear(outputs).permute(0, 1, 3, 2))
        return outputs, structure_kl_loss_sum / self.K

    @staticmethod
    def get_r(decay_interval, decay_r, current_epoch, init_r=0.9, final_r=0.5):
        r = init_r - current_epoch // decay_interval * decay_r
        if r < final_r:
            r = final_r
        return r


class DSTAGNNBlock(nn.Module):
    def __init__(self, num_nodes, num_of_time, num_of_d, nb_filter, time_strides, input_size, K, d_model, d_k, d_v,
                 n_heads, dropout=0.3):
        super().__init__()
        self.TAT = MultiHeadAttention(num_nodes, d_k, d_v, n_heads)
        self.SAT = SMultiHeadAttention(d_model, d_k, d_v, K)
        self.graphConv = GraphConvWithSAT(K, input_size, nb_filter, num_nodes)
        self.pre_conv = nn.Conv2d(num_of_time, d_model, kernel_size=(1, num_of_d))
        self.dropout = nn.Dropout(dropout)
        self.pooling = torch.nn.MaxPool2d(kernel_size=(1, 2), stride=None, padding=0,
                                          return_indices=False, ceil_mode=False)
        self.residual_conv = nn.Conv2d(input_size, nb_filter, kernel_size=(1, 1), stride=(1, time_strides))
        self.fcmy = nn.Sequential(
            nn.Linear(num_of_time, num_of_time),
            nn.Dropout(0.05),
        )
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        # self.ln = nn.LayerNorm(nb_time_filter)

    def forward(self, x: torch.Tensor, res_att: torch.Tensor, **kwargs) -> \
            Tuple[Any, Any, Any]:
        batch_size, num_of_vertices, num_of_features, num_of_time = x.shape
        TEmx = x.permute(0, 2, 3, 1)
        TATout, re_At = self.TAT(TEmx, TEmx, TEmx, res_att)
        x_TAt = self.pre_conv(TATout.permute(0, 2, 3, 1))[:, :, :, -1].permute(0, 2, 1)
        SEmx_TAt = self.dropout(x_TAt)
        STAt = self.SAT(SEmx_TAt, SEmx_TAt)
        spatial_gcn, structure_kl_loss = self.graphConv(x, STAt, **kwargs)
        X = spatial_gcn.permute(0, 2, 1, 3)
        time_conv = self.fcmy(X)
        time_conv_output = self.tanh(time_conv)
        # time_conv_output = X
        if num_of_features == 1:
            x_residual = self.residual_conv(x.permute(0, 2, 1, 3))
        else:
            x_residual = x.permute(0, 2, 1, 3)
        x_residual = self.tanh(x_residual + time_conv_output).permute(0, 2, 1, 3)
        # x_residual = self.ln(self.tanh(x_residual + time_conv_output).permute(0, 3, 2, 1)).permute(0, 2, 3, 1)

        return x_residual, re_At, structure_kl_loss


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


class LDSTGNN(nn.Module):
    def __init__(self, num_of_d, nb_block, in_channels, K, nb_filter, time_strides,
                 num_for_predict, len_input, num_of_vertices, d_model, d_k, d_v, n_heads, dropout):
        super().__init__()
        self.dataEmbedding = DataEmbedding(in_channels, nb_filter)
        self.BlockList = nn.ModuleList(
            [DSTAGNNBlock(num_of_vertices, len_input, nb_filter, nb_filter, time_strides,
                          nb_filter, K, d_model, d_k, d_v, n_heads, dropout)])
        self.BlockList.extend([DSTAGNNBlock(num_of_vertices, len_input // time_strides, nb_filter,
                                            nb_filter, 1, nb_filter, K, d_model, d_k, d_v, n_heads, dropout) for _ in
                               range(nb_block - 1)])
        self.final_conv = nn.Conv2d(int((len_input / time_strides) * nb_block), 128, kernel_size=(1, nb_filter))
        self.final_fc = nn.Linear(128, num_for_predict)

    def forward(self, x: torch.Tensor, **kwargs):
        x = self.dataEmbedding(x).permute(0, 2, 3, 1)
        need_concat = []
        res_att = 0
        structure_kl_loss_sum = 0
        for block in self.BlockList:
            x, res_att, structure_kl_loss = block(x, res_att, **kwargs)
            need_concat.append(x)
            structure_kl_loss_sum += structure_kl_loss

        final_x = torch.cat(need_concat, dim=-1)
        output1 = torch.tanh(self.final_conv(final_x.permute(0, 3, 1, 2))[:, :, :, -1].permute(0, 2, 1))
        output = self.final_fc(output1)
        output = output.unsqueeze(-1).permute(0, 2, 1, 3)
        return output, structure_kl_loss_sum / len(self.BlockList)
