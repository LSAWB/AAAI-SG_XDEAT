import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F

from einops import rearrange, repeat
import math
from math import sqrt
from typing import List, Tuple

from rtdl_num_embeddings import PiecewiseLinearEmbeddings


class _CLSEmbedding(nn.Module):
    def __init__(self, d_embedding: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(d_embedding))  # learnable vector of shape [d_embedding]
        self.reset_parameters()

    def reset_parameters(self) -> None:
        d_rsqrt = self.weight.shape[-1] ** -0.5
        nn.init.uniform_(self.weight, -d_rsqrt, d_rsqrt)  # Xavier-style initialization

    def forward(self, batch_dims: Tuple[int]) -> Tensor:
        if not batch_dims:
            raise ValueError('The input must be non-empty')

        return self.weight.expand(*batch_dims, 1, -1)


class _ReGLU(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] % 2:
            raise ValueError(
                'For the ReGLU activation, the last input dimension'
                f' must be a multiple of 2, however: {x.shape[-1]=}'
            )
        a, b = x.chunk(2, dim=-1)
        return a * F.relu(b)
    
    
class ContinuousEmbeddings(nn.Module):
    def __init__(self, n_features: int, d_embedding: int):
        super().__init__()
        self.n_features     = n_features
        self.d_embedding    = d_embedding

        # Create weight and bias as trainable parameters
        self.weight = nn.Parameter(torch.empty(n_features, d_embedding))
        self.bias   = nn.Parameter(torch.empty(n_features, d_embedding))

        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        d_rsqrt = self.d_embedding ** -0.5
        nn.init.uniform_(self.weight, -d_rsqrt, d_rsqrt)
        nn.init.uniform_(self.bias, -d_rsqrt, d_rsqrt)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 2 or x.shape[1] != self.n_features:
            raise ValueError(f"Input should be [B, {self.n_features}], but got {x.shape}")

        x = x.unsqueeze(-1)  
        out = x * self.weight + self.bias  
        return out


class CategoricalEmbeddings(nn.Module):
    def __init__(self, cardinalities: List[int], d_embedding: int, bias: bool = True):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(x, d_embedding) for x in cardinalities])
        self.bias = nn.Parameter(torch.empty(len(cardinalities), d_embedding)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        d_rsqrt = self.embeddings[0].embedding_dim ** -0.5
        for emb in self.embeddings:
            nn.init.uniform_(emb.weight, -d_rsqrt, d_rsqrt)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -d_rsqrt, d_rsqrt)

    def forward(self, x):
        x = torch.stack([self.embeddings[i](x[:, i]) for i in range(len(self.embeddings))], dim=-2)
        if self.bias is not None:
            x = x + self.bias
        return x 


class FullAttention(nn.Module):
    def __init__(self, scale=None, attention_dropout=0.1):
        super(FullAttention, self).__init__()
        self.scale      = scale
        self.dropout    = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values):   # 加入參數
        B, L, H, E  = queries.shape
        _, S, _, D  = values.shape
        scale       = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)         # [B, H, L, S]
        A = torch.softmax(scale * scores, dim=-1)                       # attention scores
        A = self.dropout(A)
        V = torch.einsum("bhls,bshd->blhd", A, values)

        return V.contiguous()


class AttentionLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_values=None, attention_dropout=0.1):
        super(AttentionLayer, self).__init__()
        d_keys          = d_keys or (d_model // n_heads)
        d_values        = d_values or (d_model // n_heads)
        
        self.n_heads    = n_heads
        
        self.inner_attention    = FullAttention(scale=None, attention_dropout=attention_dropout)
        self.query_projection   = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection     = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection   = nn.Linear(d_model, d_values * n_heads)
        self.out_projection     = nn.Linear(d_values * n_heads, d_model)
        

    def forward(self, queries, keys, values):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H       = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys    = self.key_projection(keys).view(B, S, H, -1)
        values  = self.value_projection(values).view(B, S, H, -1)

        out = self.inner_attention(queries, keys, values)
        out = out.view(B, L, -1)
        return self.out_projection(out)


class ASSAAttention(nn.Module):
    def __init__(self, scale=None, attention_dropout=0.1):
        super(ASSAAttention, self).__init__()
        self.scale = scale
        self.dropout = nn.Dropout(attention_dropout)

        # Learnable parameters for weight fusion
        self.a1 = nn.Parameter(torch.tensor(1.0))
        self.a2 = nn.Parameter(torch.tensor(1.0))

    def forward(self, queries, keys, values):
        B, L, H, E = queries.shape
        scale = self.scale or 1. / sqrt(E)

        # QK^T / sqrt(d)
        qk_scores = torch.einsum("blhe,bshe->bhls", queries, keys) * scale

        # === SSA branch: Squared ReLU(QK^T / sqrt(d) + B) ===
        ssa_scores = F.relu(qk_scores) ** 2
        ssa_scores = self.dropout(ssa_scores)
        ssa_output = torch.einsum("bhls,bshd->blhd", ssa_scores, values)
        
        # === DSA branch: Softmax(QK^T / sqrt(d) + B) ===
        dsa_scores = torch.softmax(qk_scores, dim=-1)
        dsa_scores = self.dropout(dsa_scores)
        dsa_output = torch.einsum("bhls,bshd->blhd", dsa_scores, values)

        # === Adaptive fusion ===
        w1 = torch.exp(self.a1)
        w2 = torch.exp(self.a2)
        alpha1 = w1 / (w1 + w2)
        alpha2 = w2 / (w1 + w2)

        output = alpha1 * ssa_output + alpha2 * dsa_output

        return output.contiguous()


class ASSAAttentionLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_values=None, attention_dropout=0.1):
        super(ASSAAttentionLayer, self).__init__()
        d_keys      = d_keys or (d_model // n_heads)
        d_values    = d_values or (d_model // n_heads)

        self.n_heads = n_heads

        self.inner_attention    = ASSAAttention(scale=None, attention_dropout=attention_dropout)
        self.query_projection   = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection     = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection   = nn.Linear(d_model, d_values * n_heads)
        self.out_projection     = nn.Linear(d_values * n_heads, d_model)

    def forward(self, queries, keys, values):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys    = self.key_projection(keys).view(B, S, H, -1)
        values  = self.value_projection(values).view(B, S, H, -1)

        out = self.inner_attention(queries, keys, values)
        out = out.view(B, L, -1)
        return self.out_projection(out)
    

class FeatureWiseAttentionBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=None, attention_dropout=0.1,
             residual_dropout=0.1, ffn_dropout=0.1, apply_feature_attn_norm=True):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.apply_feature_attn_norm = apply_feature_attn_norm
        
        # Dropouts
        self.att_residual_dropout   = nn.Dropout(residual_dropout)
        self.ffn_residual_dropout   = nn.Dropout(residual_dropout)
        self.ffn_dropout            = nn.Dropout(ffn_dropout)
        
        # Feature-wise Module
        self.feature_attention  = ASSAAttentionLayer(d_model, n_heads, attention_dropout=attention_dropout)
        self.norm_feature_attn  = nn.LayerNorm(d_model)
        self.norm_feature_ffn   = nn.LayerNorm(d_model)

        self.feature_ffn = nn.Sequential(
            nn.Linear(d_model, d_ff * 2),
            _ReGLU(),
            self.ffn_dropout,
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        batch, _, enc_num, _ = x.shape
        
        x = rearrange(x, 'b fea_num enc_num d_model -> (b enc_num) fea_num d_model')
        
        x_feature = self.norm_feature_attn(x) if self.apply_feature_attn_norm else x
        x = x + self.att_residual_dropout(
            self.feature_attention(x_feature, x_feature, x_feature)
        )

        x_ffn_input = self.norm_feature_ffn(x)
        x = x + self.ffn_residual_dropout(self.feature_ffn(x_ffn_input))
        
        x = rearrange(x, '(b enc_num) fea_num d_model -> b fea_num enc_num d_model', b=batch, enc_num=enc_num)
        
        return x    
        
         
class FeatureWiseAttentionLayer(nn.Module):
    def __init__(self, 
                 num_layers: int,
                 d_model: int,
                 n_heads: int,
                 d_ff: int = None,
                 attention_dropout: float = 0.1,
                 residual_dropout: float = 0.1,
                 ffn_dropout: float = 0.1,):
        super().__init__()

        self.num_layers         = num_layers

        self.layers = nn.ModuleList([
            FeatureWiseAttentionBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                attention_dropout=attention_dropout,
                residual_dropout=residual_dropout,
                ffn_dropout=ffn_dropout,
                apply_feature_attn_norm=(i != 0)
            )
            for i in range(num_layers)
        ])
        
        self.cls_embedder = _CLSEmbedding(d_embedding=d_model)
        
    def forward(self, x):
        batch, fea_num, enc_num, d_model = x.shape
        
        cls_token = self.cls_embedder((batch,)).unsqueeze(2)
        cls_token = cls_token.expand(-1, 1, enc_num, -1)
        x = torch.cat([cls_token, x], dim=1)
        
        for i, layer in enumerate(self.layers):
            x = layer(x)
            
        cls_token = x[:, 0, :] 
        return cls_token
    

class ENCWiseAttentionBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=None, attention_dropout=0.1,
             residual_dropout=0.1, ffn_dropout=0.1, apply_feature_attn_norm=True):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.apply_feature_attn_norm = apply_feature_attn_norm
        
        # Dropouts
        self.att_residual_dropout   = nn.Dropout(residual_dropout)
        self.ffn_residual_dropout   = nn.Dropout(residual_dropout)
        self.ffn_dropout            = nn.Dropout(ffn_dropout)
        
        # Feature-wise Module
        self.feature_attention  = AttentionLayer(d_model, n_heads, attention_dropout=attention_dropout)
        self.norm_feature_attn  = nn.LayerNorm(d_model)
        self.norm_feature_ffn   = nn.LayerNorm(d_model)

        self.feature_ffn = nn.Sequential(
            nn.Linear(d_model, d_ff * 2),
            _ReGLU(),
            self.ffn_dropout,
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        batch, _, enc_num, _ = x.shape
        
        x = rearrange(x, 'b fea_num enc_num d_model -> (b fea_num) enc_num d_model')
        
        x_feature = self.norm_feature_attn(x) if self.apply_feature_attn_norm else x
        x = x + self.att_residual_dropout(
            self.feature_attention(x_feature, x_feature, x_feature)
        )

        x_ffn_input = self.norm_feature_ffn(x)
        x = x + self.ffn_residual_dropout(self.feature_ffn(x_ffn_input))
        
        x = rearrange(x, '(b fea_num) enc_num d_model -> b fea_num enc_num d_model', b=batch, enc_num=enc_num)
        
        return x    
        
         
class ENCWiseAttentionLayer(nn.Module):
    def __init__(self, 
                 num_layers: int,
                 d_model: int,
                 n_heads: int,
                 d_ff: int = None,
                 attention_dropout: float = 0.1,
                 residual_dropout: float = 0.1,
                 ffn_dropout: float = 0.1,):
        super().__init__()

        self.num_layers         = num_layers

        self.layers = nn.ModuleList([
            ENCWiseAttentionBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                attention_dropout=attention_dropout,
                residual_dropout=residual_dropout,
                ffn_dropout=ffn_dropout,
                apply_feature_attn_norm=(i != 0)  # 第一層不用
            )
            for i in range(num_layers)
        ])
        
        self.cls_embedder = _CLSEmbedding(d_embedding=d_model)
        
    def forward(self, x):
        batch, fea_num, enc_num, d_model = x.shape
        
        cls_token = self.cls_embedder((batch,)).unsqueeze(2)
        cls_token = cls_token.expand(-1, fea_num, 1, -1)
        x = torch.cat([cls_token, x], dim=2)
        
        for i, layer in enumerate(self.layers):
            x = layer(x)
     
        cls_token = x[:, :, 0] 
        
        return cls_token
    
    
class DualTransformer(nn.Module):
    def __init__(self, 
                 ori_cardinalities: List[int],
                 num_features: int,
                 cat_features: int,
                 dim_model: int,
                 num_heads: int,
                 dim_ff: int,
                 num_layers_cross: int,
                 num_labels: int,
                 att_dropout: float,
                 res_dropout: float,
                 ffn_dropout: float,
                 PLE: PiecewiseLinearEmbeddings):
        
        super().__init__()

        # === Embedding ===
        self.has_cat                    = len(ori_cardinalities) > 0
        self.cat_ori_embedding_layer    = CategoricalEmbeddings(ori_cardinalities, dim_model) if self.has_cat else None
        self.num_ori_embedding_layer    = ContinuousEmbeddings(n_features=num_features, d_embedding=dim_model)
        self.cat_tar_embedding_layer    = ContinuousEmbeddings(n_features=cat_features, d_embedding=dim_model)
        self.num_tar_embedding_layer    = PLE
        
        self.d_model = dim_model
        
        # === Two-Stage Attention Encoder ===
        self.FEAWAL = FeatureWiseAttentionLayer(
            num_layers=num_layers_cross,
            d_model=dim_model,
            n_heads=num_heads,
            d_ff=dim_ff,
            attention_dropout=att_dropout,
            residual_dropout=res_dropout,
            ffn_dropout=ffn_dropout,
        )
        
        # self.ENCWAL = ENCWiseAttentionLayer(
        #     num_layers=num_layers_cross,
        #     d_model=dim_model,
        #     n_heads=num_heads,
        #     d_ff=dim_ff,
        #     attention_dropout=att_dropout,
        #     residual_dropout=res_dropout,
        #     ffn_dropout=ffn_dropout,
        # )

        # self.classifier = nn.Sequential(
        #     nn.LayerNorm(dim_model*2),
        #     nn.ReLU(),
        #     nn.Linear(dim_model*2, num_labels)
        # )

        self.classifier = nn.Sequential(
            nn.LayerNorm(dim_model),
            nn.ReLU(),
            nn.Linear(dim_model, num_labels)
        )

    def make_parameter_groups(self):
        def get_parameters(module):
            return [] if module is None else list(module.parameters())

        #  no weight decay group
        zero_wd_params = set()

        # === 1. Embedding-related modules
        zero_wd_params.update(get_parameters(self.FEAWAL.cls_embedder))
        # zero_wd_params.update(get_parameters(self.ENCWAL.cls_embedder))
        zero_wd_params.update(get_parameters(self.num_ori_embedding_layer))
        zero_wd_params.update(get_parameters(self.cat_ori_embedding_layer))
        zero_wd_params.update(get_parameters(self.cat_tar_embedding_layer))
        zero_wd_params.update(get_parameters(self.num_tar_embedding_layer))

        # === 2. All LayerNorm parameters
        for name, module in self.named_modules():
            if isinstance(module, nn.LayerNorm):
                zero_wd_params.update(module.parameters())
                
        # === 3. All bias parameters
        for name, param in self.named_parameters():
            if name.endswith('.bias'):
                zero_wd_params.add(param)

        # === 4. Create parameter groups
        decay_group = {
            'params': [p for p in self.parameters() if p not in zero_wd_params],
        }

        no_decay_group = {
            'params': list(zero_wd_params),
            'weight_decay': 0.0
        }

        return [decay_group, no_decay_group]


    def forward(self, cat_data, num_data, est_data):
        
        # === cat/num lengths ===
        batch_size = num_data.shape[0]
        cat_len = cat_data.shape[1] if cat_data is not None and cat_data.shape[1] > 0 else 0
        num_len = num_data.shape[1] if num_data is not None and num_data.shape[1] > 0 else 0

        # === ORI CAT ===
        if cat_len > 0:
            ori_cat_emb = self.cat_ori_embedding_layer(cat_data).unsqueeze(2)
        else:
            ori_cat_emb = torch.empty(batch_size, 0, 1, self.num_ori_embedding_layer.d_embedding, device=num_data.device)
                     
        # === ORI NUM ===
        if num_len > 0:
            ori_num_emb = self.num_ori_embedding_layer(num_data).unsqueeze(2)
        else:
            ori_num_emb = torch.empty(batch_size, 0, 1, self.num_ori_embedding_layer.d_embedding, device=cat_data.device)
        
        # === TAR CAT ===
        if cat_len > 0:
            tar_cat_emb = self.cat_tar_embedding_layer(est_data).unsqueeze(2)
        else:
            tar_cat_emb = torch.empty(batch_size, 0, 1, self.cat_tar_embedding_layer.d_embedding, device=cat_data.device)
        
        # === TAR NUM ===
        if num_len > 0:
            tar_num_emb = self.num_tar_embedding_layer(num_data).unsqueeze(2)
        else:
            tar_num_emb = torch.empty(batch_size, 0, 1, self.num_ori_embedding_layer.d_embedding, device=cat_data.device)
        
        # === Combine ===
        ori_embedding = torch.cat([ori_cat_emb, ori_num_emb], dim=1)
        tar_embedding = torch.cat([tar_cat_emb, tar_num_emb], dim=1)
        all_embedding = torch.cat([tar_embedding, ori_embedding], dim=2)
        
        # === Two-Stage Attention ===
        fea_final_out   = self.FEAWAL(all_embedding).mean(1)
        # enc_final_out   = self.ENCWAL(all_embedding).mean(1)
        
        # === Classification ===
        # logits = self.classifier(torch.cat([fea_final_out, enc_final_out], dim=1))
        logits = self.classifier(fea_final_out)

        return logits.squeeze(1)




# class FineWiseAttentionBlock(nn.Module):
#     def __init__(self, d_model, n_heads, d_ff=None, attention_dropout=0.1,
#              residual_dropout=0.1, ffn_dropout=0.1, apply_feature_attn_norm=True):
#         super().__init__()
#         d_ff = d_ff or 4 * d_model
#         self.d_model = d_model
#         self.apply_feature_attn_norm = apply_feature_attn_norm
        
#         # Dropouts
#         self.att_residual_dropout   = nn.Dropout(residual_dropout)
#         self.ffn_residual_dropout   = nn.Dropout(residual_dropout)
#         self.ffn_dropout            = nn.Dropout(ffn_dropout)
        
#         self.feature_attention  = ASSAAttentionLayer(d_model, n_heads, attention_dropout=attention_dropout)
#         self.norm_feature_attn  = nn.LayerNorm(d_model)
#         self.norm_feature_ffn   = nn.LayerNorm(d_model)

#         self.feature_ffn = nn.Sequential(
#             nn.Linear(d_model, d_ff * 2),
#             _ReGLU(),
#             self.ffn_dropout,
#             nn.Linear(d_ff, d_model)
#         )

#     def forward(self, x):
        
#         x_feature = self.norm_feature_attn(x) if self.apply_feature_attn_norm else x
#         x = x + self.att_residual_dropout(
#             self.feature_attention(x_feature, x_feature, x_feature)
#         )

#         x_ffn_input = self.norm_feature_ffn(x)
#         x = x + self.ffn_residual_dropout(self.feature_ffn(x_ffn_input))

#         return x    


# class FineWiseAttentionLayer(nn.Module):
    # def __init__(self, 
    #              num_layers: int,
    #              d_model: int,
    #              n_heads: int,
    #              d_ff: int = None,
    #              attention_dropout: float = 0.1,
    #              residual_dropout: float = 0.1,
    #              ffn_dropout: float = 0.1,):
    #     super().__init__()

    #     self.num_layers         = num_layers

    #     self.layers = nn.ModuleList([
    #         FineWiseAttentionBlock(
    #             d_model=d_model,
    #             n_heads=n_heads,
    #             d_ff=d_ff,
    #             attention_dropout=attention_dropout,
    #             residual_dropout=residual_dropout,
    #             ffn_dropout=ffn_dropout,
    #             apply_feature_attn_norm=(i != 0)  # 第一層不用
    #         )
    #         for i in range(num_layers)
    #     ])
        
    #     self.cls_embedder = _CLSEmbedding(d_embedding=d_model)
        
    # def forward(self, x):
    #     batch, _, _ = x.shape

    #     cls_token = self.cls_embedder((batch,))
    #     x = torch.cat([cls_token, x], dim=1)

    #     for i, layer in enumerate(self.layers):
    #         x = layer(x)
            
    #     cls_token = x[:, 0, :] 
    #     return cls_token