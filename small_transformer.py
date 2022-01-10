import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tools import attention_mask


class DotProductAttention(nn.Module):
    def __init__(self, dropout):
        super(DotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, mask, num_heads):
        # queries, keys, values = (batch_size * num_heads, seq_len, dim)
        # mask = (batch_size * num_heads, seq_len, seq_len)
        # self.attention_weights = (num_heads, seq_len, seq_len)
        _, seq_len, dim = queries.shape
        dot_prods = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(dim)
        attention_weights = self.dropout(
            F.softmax(dot_prods.masked_fill(mask, float("-inf")), dim=-1)
        )
        attention_output = torch.bmm(attention_weights, values)
        self.attention_weights = torch.mean(
            attention_weights.reshape(-1, num_heads, seq_len, seq_len), dim=0
        )
        return attention_output


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, model_dim, head_dim, dropout):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dp_attention = DotProductAttention(dropout)
        self.lin_keys = nn.Linear(model_dim, self.head_dim * self.num_heads)
        self.lin_queries = nn.Linear(model_dim, self.head_dim * self.num_heads)
        self.lin_values = nn.Linear(model_dim, self.head_dim * self.num_heads)
        self.lin_final = nn.Linear(self.head_dim * self.num_heads, model_dim)

    def forward(self, queries, keys, values, mask):
        # queries, keys, values = (batch_size, seq_len, model_dim)
        # mask = (1, seq_len, seq_len)
        # attention_output = (batch_size, seq_len, num_heads * head_dim)
        # proj_final = (batch_size, seq_len, model_dim)
        proj_queries = self.transpose_qkv(self.lin_queries(queries))
        proj_keys = self.transpose_qkv(self.lin_keys(keys))
        proj_values = self.transpose_qkv(self.lin_values(values))
        attention_output = self.dp_attention(
            proj_queries, proj_keys, proj_values, mask, self.num_heads
        )
        attention_output = self.transpose_attention_output(attention_output)
        proj_final = self.lin_final(attention_output)
        return proj_final

    def transpose_qkv(self, m):
        # Input m = (batch_size, seq_len, num_heads * head_dim)
        # Output m = (batch_size * num_heads, seq_len, head_dim)
        batch_size, seq_len, _ = m.shape
        m = m.reshape(batch_size, seq_len, self.num_heads, -1)
        m = m.permute(0, 2, 1, 3)
        _, _, seq_len, head_dim = m.shape
        m = m.reshape(-1, seq_len, head_dim)
        return m

    def transpose_attention_output(self, m):
        # Input m = (batch_size * num_heads, seq_len, head_dim)
        # Output m = (batch_size, seq_len, num_heads * head_dim)
        _, seq_len, head_dim = m.shape
        m = m.reshape(-1, self.num_heads, seq_len, head_dim)
        m = m.permute(0, 2, 1, 3)
        batch_size, seq_len, _, _ = m.shape
        m = m.reshape(batch_size, seq_len, -1)
        return m


class PositionwiseFeedForward(nn.Module):
    def __init__(self, model_dim, ff_middle_dim, dropout):
        super(PositionwiseFeedForward, self).__init__()
        self.lin1 = nn.Linear(model_dim, ff_middle_dim)
        self.lin2 = nn.Linear(ff_middle_dim, model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.lin2(self.dropout(F.relu(self.lin1(x))))


class DecoderBlock(nn.Module):
    def __init__(self, num_heads, model_dim, head_dim, ff_middle_dim, dropout=0.1):
        super(DecoderBlock, self).__init__()
        self.mh_attention = MultiHeadAttention(num_heads, model_dim, head_dim, dropout)
        self.layer_norm1 = nn.LayerNorm(model_dim)
        self.layer_norm2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.ff = PositionwiseFeedForward(model_dim, ff_middle_dim, dropout)

    def forward(self, x, mask):
        x_layer_norm1 = self.layer_norm1(x)
        mh_attention_output = self.mh_attention(
            x_layer_norm1, x_layer_norm1, x_layer_norm1, mask
        )
        x = x + self.dropout1(mh_attention_output)
        x_layer_norm2 = self.layer_norm2(x)
        ff_output = self.ff(x_layer_norm2)
        x = x + self.dropout2(ff_output)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, max_seq_len, model_dim, dropout):
        super(PositionalEncoding, self).__init__()
        self.model_dim = model_dim
        self.dropout = nn.Dropout(dropout)

        # Fixed positional encoding based on Vaswani et al 2017
        pos_enc = torch.zeros(max_seq_len, model_dim)
        pos_indices = torch.arange(0, max_seq_len).unsqueeze(1)
        div_term = 1 / torch.pow(10000, torch.arange(0, model_dim, 2) / model_dim)
        pos_enc[:, 0::2] = torch.sin(pos_indices * div_term)
        pos_enc[:, 1::2] = torch.cos(pos_indices * div_term)
        pos_enc = pos_enc.unsqueeze(0)
        self.register_buffer("pos_enc", pos_enc)

    def forward(self, x):
        _, seq_len, _ = x.shape
        x = x * math.sqrt(self.model_dim) + self.pos_enc[:, :seq_len]
        return self.dropout(x)


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        max_seq_len,
        num_heads,
        model_dim,
        head_dim,
        ff_middle_dim,
        num_blocks,
        output_dim,
        dropout=0.1,
    ):
        super(Transformer, self).__init__()
        self.model_dim = model_dim
        self.embedding = nn.Embedding(vocab_size, model_dim)
        self.pos_encoding = PositionalEncoding(max_seq_len, model_dim, dropout)
        self.transformer_stack = nn.ModuleList(
            [
                DecoderBlock(num_heads, model_dim, head_dim, ff_middle_dim, dropout)
                for _ in range(num_blocks)
            ]
        )
        self.layer_norm = nn.LayerNorm(model_dim)
        self.lin_output = nn.Linear(model_dim, output_dim)

    def forward(self, x, mask):
        # input x = (batch_size, seq_len)
        # output x = (batch_size, seq_len, output_dim)
        # mask = (1, seq_len, seq_len)
        # cache_attention_weights = list of len num_blocks
        cache_attention_weights = []
        x = self.embedding(x)
        for i in range(len(self.transformer_stack)):
            block = self.transformer_stack[i]
            x = block(x, mask)
            cache_attention_weights.append(
                block.mh_attention.dp_attention.attention_weights
            )
        x = self.layer_norm(x)
        x = self.lin_output(x)
        return x, cache_attention_weights


class TransformerModel(nn.Module):
    def __init__(self, model_config, vocab_size, output_dim, device):
        super(TransformerModel, self).__init__()
        self.transformer = Transformer(
            **model_config, vocab_size=vocab_size, output_dim=output_dim
        )
        self.device = device

    def forward(self, x):
        _, seq_len = x.shape
        mask = attention_mask(seq_len).unsqueeze(0).to(self.device)
        preds, cached_attention_weights = self.transformer(x, mask)
        return preds, cached_attention_weights

    def eval_loss(self, x, y):
        raw_preds, _ = self(x)
        preds = torch.argmax(raw_preds[:, -1, :], dim=-1)
        loss = F.cross_entropy(raw_preds[:, -1, :], y.flatten())
        accuracy = (preds == y.flatten()).float().mean()
        n = x.shape[0]
        cache_info = {"loss": (loss.item(), n), "accuracy": (accuracy.item(), n)}
        return loss, cache_info
