import mindspore
from mindspore import Tensor,nn,ops

import matplotlib.pyplot as plt

import numpy as np
import math
from math import sqrt
from utils.masking import TriangularCausalMask, ProbMask
import os


class FullAttention(nn.Cell):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(1 - attention_dropout)

    def construct(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = ops.BatchMatMul()(queries.transpose(0, 2, 1, 3), keys.transpose(0, 2, 3, 1)) * scale

        if self.mask_flag:
            if attn_mask is not None:
                scores = scores + attn_mask.mask * (-np.inf)

        A = self.dropout(ops.Softmax(axis=-1)(scores))
        V = ops.BatchMatMul()(A, values.transpose(0, 2, 1, 3)).transpose(0, 2, 1, 3)

        if self.output_attention:
            return V, A
        else:
            return V, None


class ProbAttention(nn.Cell):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(1 - attention_dropout)  # MindSpore 的 Dropout 参数是保留率

    def _prob_QK(self, Q, K, sample_k, n_top):
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # 计算采样 Q_K
        K_expand = ops.Tile()(K.expand_dims(-3), (1, 1, L_Q, 1, 1))
        index_sample = np.random.randint(0, L_K, (L_Q, sample_k))
        index_sample = Tensor(index_sample, mindspore.int32)
        K_sample = ops.GatherD()(K_expand, -2, index_sample)

        Q_K_sample = ops.BatchMatMul()(Q.expand_dims(-2), K_sample.swapaxes(-2, -1)).squeeze(-2)

        # 寻找稀疏度最高的 Top_k query
        M = ops.ReduceMax(keep_dims=False)(Q_K_sample, -1) - ops.ReduceMean(keep_dims=False)(Q_K_sample, -1)
        M_top = ops.TopK()(M, n_top)[1]

        # 使用选中的 Q 计算 Q_K
        batch_indices = Tensor(np.arange(B), mindspore.int32).view(-1, 1, 1)
        head_indices = Tensor(np.arange(H), mindspore.int32).view(1, -1, 1)
        Q_reduce = ops.GatherD()(Q, -2, M_top.expand_dims(-1))
        Q_K = ops.BatchMatMul()(Q_reduce, K.swapaxes(-2, -1))

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            V_mean = ops.ReduceMean(keep_dims=False)(V, -2)
            context = ops.Tile()(V_mean.expand_dims(-2), (1, 1, L_Q, 1))
        else:
            assert L_Q == L_V, "L_Q must equal L_V for self-attention with mask"
            context = ops.Cumsum()(V, axis=-2)
        return context

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores = ops.Select()(attn_mask.mask, Tensor(-np.inf, mindspore.float32), scores)

        attn = ops.Softmax(axis=-1)(scores)
        batch_indices = Tensor(np.arange(B), mindspore.int32).view(-1, 1, 1)
        head_indices = Tensor(np.arange(H), mindspore.int32).view(1, -1, 1)
        context_in = ops.tensor_scatter_update(context_in, index, ops.BatchMatMul()(attn, V))

        if self.output_attention:
            attns = Tensor(np.ones([B, H, L_V, L_V]) / L_V, mindspore.float32)
            attns = ops.tensor_scatter_update(attns, index, attn)
            return context_in, attns
        else:
            return context_in, None

    def construct(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = ops.Transpose()(queries, (0, 2, 1, 3))
        keys = ops.Transpose()(keys, (0, 2, 1, 3))
        values = ops.Transpose()(values, (0, 2, 1, 3))

        U_part = min(self.factor * int(np.ceil(np.log(L_K))), L_K)
        u = min(self.factor * int(np.ceil(np.log(L_Q))), L_Q)

        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale

        context = self._get_initial_context(values, L_Q)
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)

        return context, attn


class AttentionLayer(nn.Cell):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Dense(d_model, d_keys * n_heads)
        self.key_projection = nn.Dense(d_model, d_keys * n_heads)
        self.value_projection = nn.Dense(d_model, d_values * n_heads)
        self.out_projection = nn.Dense(d_values * n_heads, d_model)
        self.n_heads = n_heads

        self.reshape = ops.Reshape()
        self.transpose = ops.Transpose()

    def construct(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries)
        queries = self.reshape(queries, (B, L, H, -1))

        keys = self.key_projection(keys)
        keys = self.reshape(keys, (B, S, H, -1))

        values = self.value_projection(values)
        values = self.reshape(values, (B, S, H, -1))

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )

        out = self.reshape(out, (B, L, -1))
        out = self.out_projection(out)

        return out, attn
