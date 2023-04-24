import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from math import sqrt
from utils.masking import TriangularCausalMask, ProbMask

class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.fullattn_mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        
    def forward(self, queries, keys, values, attn_mask):
        #注意力缩放机制
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1./sqrt(E)
        #注意力机制 用于计算查询和键之间的相似度得分
        Full_QKscores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.fullattn_mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            Full_QKscores.masked_fill_(attn_mask.mask, -np.inf)
        # 缩放点积注意力
        Attn = self.dropout(torch.softmax(scale * Full_QKscores, dim=-1))
        WeightSum_AttnVal = torch.einsum("bhls,bshd->blhd", Attn, values)
        if self.output_attention:
            return (WeightSum_AttnVal.contiguous(), Attn)
        else:
            return (WeightSum_AttnVal.contiguous(), None)

class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.Probattn_mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def Prob_QK_similar(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)

        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_dim_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)  # 先增加一个维度，相当于复制，再扩充
        random_index = torch.randint(L_K, (L_Q, sample_k))  # real U = U_part(factor*ln(L_k))*L_q 构建96*25的随机数
        random_K = K_dim_expand[:, :, torch.arange(L_Q).unsqueeze(1), random_index, :]
        Q_randomK = torch.matmul(Q.unsqueeze(-2), random_K.transpose(-2, -1)).squeeze()  # 96个Q和25个K之间的关系

        # find the Top_k query with sparisty measurement
        Q_randomK_Max = Q_randomK.max(-1)[0] - torch.div(Q_randomK.sum(-1), L_K)  # 96个Q中每一个选跟其他K关系最大的值 再计算与均匀分布的差异
        Q_randomK_Maxtop = Q_randomK_Max.topk(n_top, sorted=False)[1]  # 对96个Q的评分中选出25个 返回值1表示要得到索引

        # use the reduced Q to calculate Q_K
        Q_filtered = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   Q_randomK_Maxtop, :]  # factor*ln(L_q) 取出来Q的特征
        filteredQ_K = torch.matmul(Q_filtered, K.transpose(-2, -1))  # factor*ln(L_q)*L_k 25个Q和全部K之间的关系
        return filteredQ_K, Q_randomK_Maxtop

    def set_initial_context_vector(self, V, L_Q):
        B, H, L_V, D = V.shape
        if self.Probattn_mask_flag:
            assert L_Q == L_V, "L_Q must be equal to L_V for self-attention"
            contex = V.cumsum(dim=-2)
        else:
            V_mean = V.mean(dim=-2)
            contex = V_mean.unsqueeze(-2).expand(B, H, L_Q, V_mean.shape[-1]).clone()

        return contex

    def update_context_vector(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.Probattn_mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)
        Prob_attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)
        # 更新上下文向量
        context_in[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(Prob_attn, V).type_as(context_in)  # 对25个有Q的更新V，其余的没变还是均值
        if self.output_attention:
            prob_attns_out = (torch.ones([B, H, L_V, L_V]) / L_V).type_as(Prob_attn).to(Prob_attn.device)
            prob_attns_out[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = Prob_attn
            return (context_in, prob_attns_out)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        Key_selectnum = self.factor * np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k) Key里要选的个数
        Q_selectnum = self.factor * np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

        if Key_selectnum >= L_K:
            Key_selectnum = L_K
        else:
            Key_selectnum = Key_selectnum

        if Q_selectnum >= L_Q:
            Q_selectnum = L_Q
        else:
            Q_selectnum = Q_selectnum

        topscores, topscores_index = self.Prob_QK_similar(queries, keys, sample_k=Key_selectnum, n_top=Q_selectnum)

        # add scale factor
        Prob_scale = self.scale or 1. / sqrt(D)
        if Prob_scale is not None:
            topscores = topscores * Prob_scale
        # get the context
        initial_context = self.set_initial_context_vector(values, L_Q)
        # update the context with selected top_k queries
        final_context, final_attn = self.update_context_vector(initial_context, values, topscores, topscores_index, L_Q, attn_mask)

        return final_context.transpose(2, 1).contiguous(), final_attn


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads,
                 d_keys=None, d_values=None, mix=False):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys if d_keys is not None else (d_model // n_heads)
        d_values = d_values if d_values is not None else (d_model // n_heads)

        self.self_attention = attention
        self.query_linear = nn.Linear(d_model, d_keys * n_heads)
        self.key_linear = nn.Linear(d_model, d_keys * n_heads)
        self.value_linear = nn.Linear(d_model, d_values * n_heads)
        self.out_linear = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_linear(queries).view(B, L, H, -1)
        keys = self.key_linear(keys).view(B, S, H, -1)
        values = self.value_linear(values).view(B, S, H, -1)

        self_attn_out, self_attn = self.self_attention(queries, keys, values, attn_mask)
        if self.mix:
            self_attn_out = self_attn_out.transpose(2, 1).contiguous()
        self_attn_out = self_attn_out.view(B, L, -1)

        return self.out_linear(self_attn_out), self_attn
