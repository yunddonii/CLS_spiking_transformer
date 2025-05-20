import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.clock_driven.neuron import MultiStepLIFNode, surrogate

from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg 
from timm.models.layers import trunc_normal_
# from timm.optim.optim_factory import create_optimizer_v2

# from timm.models import create_model
from timm.utils import *

from layers import SpkEncoder
from encoder import Split2Patch, Embedding, Block, TemporalBlock
from layers import SpkEncoder, AttentionPool1d, MutualCrossAttention, Consolidation, SSA_rel_scl, TokenMixer, MemoryUpdate


__all__ = ['spikformer']

class Decoder(nn.Module):
    def __init__(self,  embed_dim, patch_size=[16, 32], ratio=8, tau=2.0, bias=True) -> None:
        super().__init__()

        self.reconstruction = nn.Linear(embed_dim, patch_size, bias=bias)
        
    def forward(self, x, out_rec=True):
        """
        [original] x: N x L x C(embed_dim)
        [MyModel] x: T x B x D
        
        out: reconstructed output -> N x L x c_out
        if expand is True: out's shape becomes [B X L]
        """
        
        T, B, N, D = x.shape

        rec_x = self.reconstruction(x.flatten(0, 1)).reshape(T, B, N, -1).contiguous()
        return rec_x

class Spikformer(nn.Module):
    def __init__(self, gating=['original', 'ablation'], 
                 train_mode=['training', 'testing', 'visual'], 
                 window_size=192,
                data_patch_size=63, 
                num_classes=2, 
                time_num_layers=2,
                embed_dim=[64, 128, 256], 
                num_heads=[1, 2, 4], 
                mlp_ratios=1, 
                qkv_bias=False, 
                qk_scale=None,
                drop_rate=0., 
                attn_drop_rate=0., 
                drop_path_rate=0., 
                norm_layer=nn.LayerNorm,
                depths=[6, 8, 6], 
                sr_ratios=[8, 4, 2], 
                T = 4, 
                data_patching_stride=2, 
                padding_patches=None, 
                lif_bias=False, 
                tau=2.0, 
                spk_encoding=False,
                attn=['SSA', 'MSSA'],
                pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None,
                ):
        super().__init__()
        
        self.train_mode = train_mode
        self.gating = gating
        
        self.T = T  # time step
        self.spk_encoding = spk_encoding
        self.data_patch_size = data_patch_size
        self.data_patching_stride = data_patching_stride
        
        mlp_ratios = 2
        
        self.data_num_patches = T
        self.time_num_patches = T
        
        # if padding_patches == "end":
        # self.data_num_patches += 1

        # print(f"sequence of patches >> data_num_patches = {self.data_num_patches}, time_patch_size = {self.time_patch_size}")
        print(f"len of tokens in sequence >> {self.data_patch_size}")

        self.split_data = Split2Patch(self.data_patch_size, self.data_patching_stride, None)

        self.spk_encoder = SpkEncoder(T) if spk_encoding else None
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]  # stochastic depth decay rule

        self.encoding = Embedding(num_patches=self.data_num_patches,
                                    patch_size=self.data_patch_size,
                                    embed_dim=embed_dim,
                                    # pe=False,
                                    pe=True,
                                    bias=lif_bias,
                                    tau=tau
                                    )

        self.data_block = nn.ModuleList([Block(
                    dim=embed_dim, seq_len=self.data_num_patches, num_heads=num_heads, mlp_ratio=mlp_ratios, qkv_bias=qkv_bias,
                    qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[j], attn=attn,
                    norm_layer=norm_layer, sr_ratio=sr_ratios, lif_bias=lif_bias, tau=tau, mutual=True)
                    for j in range(depths)])
        

        
        self.time_block = TemporalBlock(T=T,
                                        num_layers=time_num_layers,
                                        patch_size=data_patch_size,
                                        embed_dim=embed_dim,
                                        lif_bias=lif_bias,
                                        num_heads=num_heads,
                                        tau=tau,
                                        mutual=False)
        
        self.replay = MemoryUpdate(seq_len=T, dim=embed_dim, out_seq=True, lif_bias=lif_bias, num_heads=num_heads, tau=tau)

        if self.train_mode == 'training' : 
            self.weak_decoder = Decoder(embed_dim=embed_dim, patch_size=self.data_patch_size, ratio=8, tau=tau, bias=lif_bias)
            self.head = nn.Linear(embed_dim, num_classes, bias=lif_bias) if num_classes > 0 else nn.Identity()

        if gating == 'ablation':
            self._init_ablation()

        self.apply(self._init_weights)
        
        if self.train_mode != 'training':
            for module in self.modules():
                if isinstance(module, nn.BatchNorm1d):
                    module.eval()

    @torch.jit.ignore
    def _get_pos_embed(self, pos_embed, patch_embed, N):
        if N == self.patch_embed1.num_patches:
            return pos_embed
        else:
            return F.interpolate(
                pos_embed.reshape(1, patch_embed.H, patch_embed.W, -1).permute(0, 3, 1, 2),
                size=(N), mode="bilinear").reshape(1, -1, N).permute(0, 2, 1)

    def _init_weights(self, m):
        # if isinstance(m, nn.Linear): 
        #     if isinstance(m, nn.Linear) and m.bias is not None:
        #         nn.init.xavier_normal_(m.weight)
        #         nn.init.constant_(m.bias, 0)
        #     else:
        #         nn.init.xavier_normal_(m.weight)
        
        # elif isinstance(m, nn.LayerNorm):
        #     nn.init.constant_(m.bias, 0)
        #     nn.init.constant_(m.weight, 1.0)

        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def forward(self, x):
        
        if self.spk_encoding:
            x = self.spk_encoder(x)
        else: 
            x = (x.unsqueeze(0)).repeat(self.T, 1, 1) # [T B L]
        
        x = self.split_data(x) # [T B N P]
        self.org_x = x.clone().detach().transpose(0, 2).contiguous().mean(2, keepdim=True)
        x = self.encoding(x)
        
        temporal_emb = self.time_block(x) if (self.gating != 'ablation') else None
        
        for dblk in self.data_block:
            x_data = dblk(x, temporal_emb)
            
        z = self.head(x_data.mean(0).mean(1))
        
        if (self.gating != 'ablation') and (self.train_mode == 'training'):
            
            temporal_emb = temporal_emb * (1. - self.replay(temporal_emb, x_data))
            rec_x = self.weak_decoder(temporal_emb)

            assert self.org_x.shape == rec_x.shape, f"original x_time' shape is {self.org_x.shape}, and reconstructed x_time' shape is {rec_x.shape}"
            
            return z, self.org_x, rec_x
        
        elif (self.train_mode == 'testing') or (self.gating == 'ablation'):
            # self._init_ablation()
            return z
        
        elif self.train_mode == 'visual':
            feature = x_data.mean(2)

            return {
                # 'attn_map' : attn_map.clone().detach(),
                'x_time_emb' : temporal_emb.clone().detach().squeeze(2).mean(0),
                'x_time_temporal' : temporal_emb.clone().detach().squeeze(2).mean(-1).transpose(0, 1),
                # 'x_data_emb' : x_data.clone().detach().mean(2).mean(0),
                # 'x_data_temporal' : x_data.clone().detach().mean(2).mean(-1).transpose(0, 1),
                'after_attn_gate' : feature.mean(0),
                'after_attn_gate_temporal' : feature.mean(-1).transpose(0, 1),
            }
            
        else:
            raise ValueError
            
    # TODO
    def _init_ablation(self):
        
        delattr(self, 'time_block')
        # delattr(self, 'gate_attn')
        if hasattr(self, 'weak_decoder'): delattr(self, 'weak_decoder')
        if hasattr(self, 'replay') : delattr(self, 'replay')
    

@register_model
def spikformer(pretrained=False, **kwargs):
    model = Spikformer(pretrained=pretrained, **kwargs)
    model.default_cfg = _cfg()
    return model