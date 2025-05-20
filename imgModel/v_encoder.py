import torch
import torch.nn as nn
from spikingjelly.clock_driven.neuron import MultiStepLIFNode, surrogate

from positional import tAPE
from v_layers import SSA_rel_scl, MLP, MutualCrossAttention, SpikLinearLayer
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from einops.layers.torch import Rearrange

__all__ = ['spikformer']

class Split2Patch(nn.Module):
    def __init__(self, img_size, patch_size) -> None:
        """
        input: [T B L]
        output: [T B N P]    
        """
        super().__init__()
        
        img_h, img_w = img_size
        patch_h, patch_w = patch_size
        
        assert img_h % patch_h == 0 and img_w % patch_w == 0, "Image dimensions must be divisible by the patch size."
        self.patch_size = patch_size

        self.to_patch = Rearrange('t b c (h p1) (w p2) -> t b (h w) (p1 p2 c)', p1 = patch_h, p2 = patch_w)


    def forward(self, z):
        
        return self.to_patch(z)


class Embedding(nn.Module):
    def __init__(self, patch_size, num_patches, pe=False, stride=2, embed_dim=128, dropout=0, bias=False, tau=2.0) -> None:
        super().__init__()
        
        self.num_patches = num_patches
        self.patch_size = patch_size
        patch_h, patch_w = patch_size
        in_channel = 3
        self.stride = stride
        
        self.pe = pe
        
        patch_dim = in_channel * patch_h * patch_w
        
        self.emb1_linear = nn.Linear(patch_dim, embed_dim, bias=bias)
        self.emb1_bn = nn.BatchNorm1d(embed_dim)
        self.emb1_lif = MultiStepLIFNode(tau=tau, v_threshold=0.5, detach_reset=True, backend='cupy', surrogate_function=surrogate.ATan())
        
        # positional encoding
        if pe:
            self.tape = tAPE(embed_dim, max_len=num_patches)
            self.ape_lif = MultiStepLIFNode(tau=tau, v_threshold=0.5, detach_reset=True, backend='cupy', surrogate_function=surrogate.ATan())
        
        # Residual dropout
        # self.dropout = nn.Dropout(dropout)

    def forward(self, x:torch.tensor, pe=True): 
        
        T, B, N, P = x.shape
        
        x = self.emb1_linear(x.flatten(0, 1)) # have some fire value [TB N P]
        x = self.emb1_bn(x.transpose(-1, -2).contiguous()).reshape(T, B, -1, N).contiguous() # [T B P N]
        x = self.emb1_lif(x) # [T B D N]

    
        # positional embedding
        if pe:
            x = self.tape(x)
            x = self.ape_lif(x)
        
        x = x.transpose(-1, -2).contiguous()  # [T B N D]
        
        return x
 
    
class Block(nn.Module):
    def __init__(self, dim, seq_len, num_heads, mlp_ratio=2., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                drop_path=0., norm_layer=nn.LayerNorm, sr_ratio=1, lif_bias=False, tau=2.0, mutual=False, attn='MSSA'):
        super().__init__()
        

        self.attn = SSA_rel_scl(dim, seq_len, num_heads, lif_bias=lif_bias, tau=tau, attn=attn) 
        mlp_hidden_dim = int(dim * mlp_ratio)
        if mutual: self.mca = MutualCrossAttention(dim=dim, lif_bias=lif_bias, num_heads=num_heads, tau=tau, attn=attn)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop, lif_bias=lif_bias, tau=tau)

        
    def forward(self, x: torch.Tensor, mx=None):

        x = x * (1. - self.attn(x)) # token mixer
        if mx is not None: x = x * (1. - self.mca(x, mx)) 
        x = x * (1. - self.mlp(x)) # channel mixer
        
        return x
  
class TemporalBlock(nn.Module):
    def __init__(self, num_layers, embed_dim=[64, 128, 256], ratio=8, lif_bias=False, tau=2.0, num_heads=8, mutual=False):
        super().__init__()
        
        self.num_layers = num_layers
        
        self.layers = nn.ModuleList([SpikLinearLayer(embed_dim, embed_dim, lif_bias=lif_bias, tau=tau) for l in range(num_layers)])

    def forward(self, x):
        
        assert x.shape[2] == x.shape[0], f"num_pathces ({x.shape[2]}) is not equivalent to time steps ({x.shape[0]})."
        x = x.transpose(0, 2).contiguous().mean(dim=-2, keepdim=True)
        # x = x.mean(dim=-2, keepdim=True)

        for l in range(self.num_layers):
            x = self.layers[l](x)

        return x
