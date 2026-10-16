"""BERT 模型及相关组件（替代 d2l 包中的 BERT 实现）。

提供《动手学深度学习》第 14 章 BERT 相关类与函数的纯 PyTorch 实现，
供 14.06 / 14.07 等 notebook 通过 ``from src.bert import ...`` 复用，
避免依赖 d2l 包。

实现忠实于 d2l 教材，模块与属性命名与 d2l 完全一致，因而可直接加载
d2l 提供的预训练参数（pretrained.params）。
"""

import math
import torch
from torch import nn


# ---------------------------------------------------------------------------
# 注意力机制
# ---------------------------------------------------------------------------

def sequence_mask(X, valid_len, value=0):
    """遮蔽超出有效长度的位置。"""
    maxlen = X.size(1)
    mask = torch.arange(maxlen, dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    return X.masked_fill(~mask, value)


def masked_softmax(X, valid_lens):
    """在最后一个轴上遮蔽无效元素后执行 softmax。"""
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    shape = X.shape
    if valid_lens.dim() == 1:
        valid_lens = valid_lens.reshape(
            -1, *([1] * (X.dim() - 2))).expand(shape[:-1])
    else:
        valid_lens = valid_lens.reshape(shape[:-1])
    mask = torch.arange(shape[-1], device=X.device) < valid_lens.unsqueeze(-1)
    return nn.functional.softmax(X.masked_fill(~mask, -1e6), dim=-1)


class DotProductAttention(nn.Module):
    """缩放点积注意力。"""

    def __init__(self, dropout, **kwargs):
        super().__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)


def transpose_qkv(X, num_heads):
    """为多头注意力重组张量。"""
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
    X = X.permute(0, 2, 1, 3)
    return X.reshape(-1, X.shape[2], X.shape[3])


def transpose_output(X, num_heads):
    """逆转 transpose_qqv 的操作。"""
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)


class MultiHeadAttention(nn.Module):
    """多头注意力。"""

    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 num_heads, dropout, bias=False, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens=None):
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)
        if valid_lens is not None:
            valid_lens = torch.repeat_interleave(
                valid_lens, repeats=self.num_heads, dim=0)
        output = self.attention(queries, keys, values, valid_lens)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)


# ---------------------------------------------------------------------------
# Transformer 编码块
# ---------------------------------------------------------------------------

class AddNorm(nn.Module):
    """残差连接后接层规范化。"""

    def __init__(self, normalized_shape, dropout, **kwargs):
        super().__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)


class PositionWiseFFN(nn.Module):
    """基于位置的前馈网络。"""

    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs,
                 **kwargs):
        super().__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))


class TransformerEncoderBlock(nn.Module):
    """Transformer 编码器块。"""

    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, use_bias=False, **kwargs):
        super().__init__(**kwargs)
        self.attention = MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads,
            dropout, use_bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def forward(self, X, valid_lens):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))


# ---------------------------------------------------------------------------
# BERT 模型
# ---------------------------------------------------------------------------

def get_tokens_and_segments(tokens_a, tokens_b=None):
    """获取输入序列的词元及其片段索引。"""
    tokens = ['<cls>'] + tokens_a + ['<sep>']
    segments = [0] * (len(tokens_a) + 2)
    if tokens_b is not None:
        tokens += tokens_b + ['<sep>']
        segments += [1] * (len(tokens_b) + 1)
    return tokens, segments


class BERTEncoder(nn.Module):
    """BERT 编码器。"""

    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout,
                 max_len=1000, key_size=768, query_size=768,
                 value_size=768, **kwargs):
        super().__init__(**kwargs)
        self.token_embedding = nn.Embedding(vocab_size, num_hiddens)
        self.segment_embedding = nn.Embedding(2, num_hiddens)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(f"{i}", TransformerEncoderBlock(
                key_size, query_size, value_size, num_hiddens,
                norm_shape, ffn_num_input, ffn_num_hiddens,
                num_heads, dropout, True))
        self.pos_embedding = nn.Parameter(
            torch.randn(1, max_len, num_hiddens))

    def forward(self, tokens, segments, valid_lens):
        X = self.token_embedding(tokens) + self.segment_embedding(segments)
        X = X + self.pos_embedding.data[:, :X.shape[1], :]
        for blk in self.blks:
            X = blk(X, valid_lens)
        return X


class MaskLM(nn.Module):
    """BERT 的掩蔽语言模型任务。"""

    def __init__(self, vocab_size, num_hiddens, num_inputs, **kwargs):
        super().__init__(**kwargs)
        self.mlp = nn.Sequential(nn.Linear(num_inputs, num_hiddens),
                                 nn.Tanh(),
                                 nn.LayerNorm(num_hiddens),
                                 nn.Linear(num_hiddens, vocab_size))

    def forward(self, X, pred_positions):
        num_pred_positions = pred_positions.shape[1]
        pred_positions = pred_positions.reshape(-1)
        batch_size = X.shape[0]
        batch_idx = torch.arange(0, batch_size)
        batch_idx = torch.repeat_interleave(batch_idx, num_pred_positions)
        masked_X = X[batch_idx, pred_positions]
        masked_X = masked_X.reshape((batch_size, num_pred_positions, -1))
        mlm_Y_hat = self.mlp(masked_X)
        return mlm_Y_hat


class NextSentencePred(nn.Module):
    """BERT 的下一句预测任务。"""

    def __init__(self, num_inputs, **kwargs):
        super().__init__(**kwargs)
        self.output = nn.Linear(num_inputs, 2)

    def forward(self, X):
        return self.output(X)


class BERTModel(nn.Module):
    """BERT 模型。"""

    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout,
                 max_len=1000, key_size=768, query_size=768,
                 value_size=768, hid_in_features=768,
                 mlm_in_features=768, nsp_in_features=768):
        super().__init__()
        self.encoder = BERTEncoder(
            vocab_size, num_hiddens, norm_shape, ffn_num_input,
            ffn_num_hiddens, num_heads, num_layers, dropout, max_len,
            key_size, query_size, value_size)
        self.hidden = nn.Sequential(
            nn.Linear(hid_in_features, num_hiddens), nn.Tanh())
        self.mlm = MaskLM(vocab_size, num_hiddens, mlm_in_features)
        self.nsp = NextSentencePred(nsp_in_features)

    def forward(self, tokens, segments, valid_lens=None,
                pred_positions=None):
        encoded_X = self.encoder(tokens, segments, valid_lens)
        if pred_positions is not None:
            mlm_Y_hat = self.mlm(encoded_X, pred_positions)
        else:
            mlm_Y_hat = None
        nsp_Y_hat = self.nsp(self.hidden(encoded_X[:, 0, :]))
        return encoded_X, mlm_Y_hat, nsp_Y_hat
