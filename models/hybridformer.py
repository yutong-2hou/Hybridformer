# models/hybridformer.py
import torch
import torch.nn as nn
from math import sqrt, ceil
from einops import rearrange, repeat
from config import TARGET_INDICES

# ====================
# Basic Layers
# ====================

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        super(RevIN, self).__init__()
        self.num_features, self.eps, self.affine = num_features, eps, affine
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))
    def forward(self, x, mode:str):
        if mode == 'norm':
            dim2reduce = tuple(range(1, x.ndim-1))
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
            self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()
            x = (x - self.mean) / self.stdev
            if self.affine: x = x * self.affine_weight + self.affine_bias
        elif mode == 'denorm':
            if self.affine: x = (x - self.affine_bias) / (self.affine_weight + self.eps*1e-5)
            x = x * self.stdev + self.mean
        return x

class FullAttention(nn.Module):
    def __init__(self, scale=None, dropout=0.1):
        super(FullAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)
    def forward(self, queries, keys, values):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = 1. / sqrt(E)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        return V.contiguous()

class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
    def forward(self, queries, keys, values):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        
        # 强制展平处理 Linear
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        
        out = self.inner_attention(queries, keys, values)
        out = out.view(B, L, -1)
        return self.out_projection(out)

class TwoStageAttentionLayer(nn.Module):
    def __init__(self, seg_num, factor, d_model, n_heads, d_ff=None, dropout=0.1):
        super(TwoStageAttentionLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.time_attention = AttentionLayer(FullAttention(dropout=dropout), d_model, n_heads)
        self.dim_attention = AttentionLayer(FullAttention(dropout=dropout), d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model))

    def forward(self, x):
        B, D, L, E = x.shape
        x = rearrange(x, 'b d l e -> (b d) l e').contiguous()
        x = x + self.dropout(self.time_attention(x, x, x))
        x = self.norm1(x)
        
        x = rearrange(x, '(b d) l e -> (b l) d e', b=B).contiguous()
        x = x + self.dropout(self.dim_attention(x, x, x))
        x = self.norm2(x)
        
        # MPS 安全修复: FeedForward 前确保形状简单
        # x is (B*L, D, E), Rank 3 is usually OK, but to be safe:
        x_ff = self.ff(x) 
        x = x + self.dropout(x_ff)
        x = self.norm3(x)
        
        x = rearrange(x, '(b l) d e -> b d l e', b=B).contiguous()
        return x

class DSW_embedding(nn.Module):
    def __init__(self, seg_len, d_model):
        super(DSW_embedding, self).__init__()
        self.seg_len = seg_len
        self.linear = nn.Linear(seg_len, d_model)
    def forward(self, x):
        x = rearrange(x, 'b l d -> b d l')
        x = rearrange(x, 'b d (l s) -> b d l s', s=self.seg_len).contiguous()
        
        # === MPS 4D Linear 修复 ===
        # 将 (Batch, Dim, SegNum, SegLen) 展平为 (Batch*Dim*SegNum, SegLen) 进行计算
        B, D, L, S = x.shape
        x = x.view(-1, S) 
        x = self.linear(x)
        x = x.view(B, D, L, -1) 
        # ==========================
        
        return x

class SegMerging(nn.Module):
    def __init__(self, d_model, win_size, norm_layer=nn.LayerNorm):
        super().__init__()
        self.d_model = d_model
        self.win_size = win_size
        self.linear_layers = nn.ModuleList([nn.Linear(i * d_model, d_model) for i in range(1, win_size + 1)])
        self.norm = norm_layer(d_model)
    def forward(self, x):
        batch_size, n_vars, seg_num, d_model = x.shape
        if seg_num % self.win_size != 0: return x 
        
        x = rearrange(x, 'b d (s w) e -> b d s (w e)', w=self.win_size).contiguous()
        
        # === MPS 4D Linear 修复 ===
        # Linear 输入是 4维 (B, D, S, W*E)，MPS 不喜欢，展平为 2维
        B, D, S, WE = x.shape
        x = x.view(-1, WE)
        x = self.linear_layers[self.win_size - 1](x)
        x = x.view(B, D, S, -1)
        # ==========================
        
        x = self.norm(x)
        return x

class Encoder(nn.Module):
    def __init__(self, e_layers, win_size, d_model, n_heads, d_ff, block_num, dropout, in_seg_num, factor):
        super().__init__()
        self.layers = nn.ModuleList([TwoStageAttentionLayer(in_seg_num // (win_size**i), factor, d_model, n_heads, d_ff, dropout) for i in range(e_layers)])
        self.merger = nn.ModuleList([SegMerging(d_model, win_size) for i in range(e_layers-1)])
    def forward(self, x):
        enc_out = [x]
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.merger[i](x)
                enc_out.append(x)
        return enc_out

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.self_attention = AttentionLayer(FullAttention(dropout=dropout), d_model, n_heads)
        self.cross_attention = AttentionLayer(FullAttention(dropout=dropout), d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model))
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, cross):
        x = x + self.dropout(self.self_attention(x, x, x))
        x = self.norm1(x)
        x = x + self.dropout(self.cross_attention(x, cross, cross))
        x = self.norm2(x)
        x = x + self.dropout(self.ff(x))
        x = self.norm3(x)
        return x

class Decoder(nn.Module):
    def __init__(self, seg_len, d_layers, d_model, n_heads, d_ff, dropout, out_seg_num, factor):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, dropout) for i in range(d_layers)])
        self.linear = nn.Linear(d_model, seg_len)
    def forward(self, x, cross_list):
        for i, layer in enumerate(self.layers):
            x = rearrange(x, 'b d l e -> (b d) l e').contiguous()
            cross = rearrange(cross_list[i], 'b d l e -> (b d) l e').contiguous()
            x = layer(x, cross)
            x = rearrange(x, '(b d) l e -> b d l e', d=cross_list[i].shape[1]).contiguous()
        
        # === MPS 4D Linear 修复 ===
        # 最后的 Linear 层输入是 (B, D, L, E)，Rank 4 -> 报错
        # 展平处理
        B, D, L, E = x.shape
        x = x.view(-1, E)
        x = self.linear(x)
        x = x.view(B, D, L, -1)
        # ==========================
        
        x = rearrange(x, 'b d l s -> b (l s) d').contiguous()
        return x

# ====================
# Main Model
# ====================

class Crossformer(nn.Module):
    def __init__(self, data_dim, in_len, out_len, seg_len, win_size=4,
                factor=10, d_model=512, d_ff=1024, n_heads=8, e_layers=3, 
                dropout=0.0, baseline=False, device=torch.device('cpu'), c_out=6):
        super(Crossformer, self).__init__()
        self.data_dim = data_dim
        self.in_len = in_len
        self.out_len = out_len
        self.seg_len = seg_len
        self.baseline = baseline
        self.device = device
        self.c_out = c_out
        
        self.pad_in_len = ceil(1.0 * in_len / seg_len) * seg_len
        self.pad_out_len = ceil(1.0 * out_len / seg_len) * seg_len
        self.in_len_add = self.pad_in_len - self.in_len
        
        self.revin = RevIN(data_dim, affine=True) 
        self.enc_value_embedding = DSW_embedding(seg_len, d_model)
        self.enc_pos_embedding = nn.Parameter(torch.randn(1, data_dim, (self.pad_in_len // seg_len), d_model))
        self.pre_norm = nn.LayerNorm(d_model)
        
        self.encoder = Encoder(e_layers, win_size, d_model, n_heads, d_ff, 1, dropout, (self.pad_in_len // seg_len), factor)
        
        self.dec_pos_embedding = nn.Parameter(torch.randn(1, c_out, (self.pad_out_len // seg_len), d_model))
        self.decoder = Decoder(seg_len, e_layers, d_model, n_heads, d_ff, dropout, (self.pad_out_len // seg_len), factor)
        
        self.target_indices = torch.tensor(TARGET_INDICES, dtype=torch.long)

    def forward(self, x_seq):
        x_seq = self.revin(x_seq, 'norm')
        
        if self.baseline:
            base = x_seq.mean(dim = 1, keepdim = True)[:, :, self.target_indices]
        else:
            base = 0
            
        if self.in_len_add != 0:
            x_seq = torch.cat((x_seq[:, :1, :].expand(-1, self.in_len_add, -1), x_seq), dim = 1)
        
        x_emb = self.enc_value_embedding(x_seq)
        x_emb = self.pre_norm(x_emb + self.enc_pos_embedding)
        
        enc_out_list = self.encoder(x_emb)
        
        enc_out_filtered = []
        for enc_x in enc_out_list:
            enc_out_filtered.append(torch.index_select(enc_x, 1, self.target_indices.to(x_seq.device)))
            
        dec_in = repeat(self.dec_pos_embedding, 'b ts_d l d -> (repeat b) ts_d l d', repeat = x_seq.shape[0])
        predict_y = self.decoder(dec_in, enc_out_filtered)
        
        predict_y = base + predict_y[:, :self.out_len, :]

        indices = self.target_indices.to(predict_y.device)
        target_mean = self.revin.mean[:, :, indices]
        target_stdev = self.revin.stdev[:, :, indices]
        
        if self.revin.affine:
            predict_y = (predict_y - self.revin.affine_bias[indices]) / (self.revin.affine_weight[indices] + 1e-5)
            
        return predict_y * target_stdev + target_mean