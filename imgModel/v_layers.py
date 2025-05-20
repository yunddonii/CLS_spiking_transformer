import torch
import torch.nn as nn

# snn
from spikingjelly.clock_driven.neuron import MultiStepLIFNode, surrogate
# from spikingjelly.clock_driven import functional

from einops import rearrange
from spikingjelly.activation_based.encoding import PoissonEncoder


__all__ = ['spikformer']

def make_look_ahead_mask(x):
    T, B, N, D = x.shape
    device = x.device
    mask = torch.triu(torch.ones((N, N))).reshape(1, 1, 1, N, N).to(device)
    
    return mask

class SpikLinearLayer(nn.Module):
    def __init__(self, in_dim, out_dim=None, tau=2.0, lif_bias=False):
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim if out_dim is not None else in_dim
        
        self.linear = nn.Linear(in_dim, self.out_dim, bias=lif_bias)
        self.bn = nn.BatchNorm1d(self.out_dim)
        self.lif = MultiStepLIFNode(tau=tau, v_threshold=0.5, detach_reset=True, backend='cupy', surrogate_function=surrogate.ATan())
        
    def forward(self, x):
        
        x_shape = x.shape
        
        x = self.linear(x.flatten(0, 1))
        x = self.bn(x.transpose(-1, -2).contiguous()).transpose(-1, -2).contiguous()
        x = self.lif(x.reshape(x_shape[0], x_shape[1], x_shape[2], -1).contiguous())
        
        return x


class SpkEncoder(nn.Module):
    def __init__(self, time_steps) -> None:
        super().__init__()
        
        self.T = time_steps
        # self.batch_size = batch_size
        # self.seq_len = seq_len
        # self.device = device
        
        self.encoder = PoissonEncoder()
    
    def forward(self, x):
        
        device = x.device
        
        spk_encoding = torch.zeros((self.T, x.shape[0], x.shape[1]), device=device)
        
        for t in range(self.T):
            spk_encoding[t] = self.encoder(x)
            
        return spk_encoding
    
    

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0., lif_bias=False, tau=2.0):
        super().__init__()
        
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1_linear = nn.Linear(in_features, hidden_features, bias=lif_bias)
        self.fc1_bn = nn.BatchNorm1d(hidden_features)
        self.fc1_lif = MultiStepLIFNode(tau=tau, v_threshold=0.5, detach_reset=True, backend='cupy', surrogate_function=surrogate.ATan())

        self.fc2_linear = nn.Linear(hidden_features, out_features, bias=lif_bias)
        self.fc2_bn = nn.BatchNorm1d(out_features)
        self.fc2_lif = MultiStepLIFNode(tau=tau, v_threshold=0.5, detach_reset=True, backend='cupy', surrogate_function=surrogate.ATan())

        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x):
        T, B, N, C = x.shape
        x = x.flatten(0, 1)
        x = self.fc1_linear(x)
        x = self.fc1_bn(x.transpose(-1, -2).contiguous()).transpose(-1, -2).contiguous().reshape(T, B, N, self.c_hidden).contiguous()
        x = self.fc1_lif(x)

        x = self.fc2_linear(x.flatten(0,1))
        x = self.fc2_bn(x.transpose(-1, -2).contiguous()).transpose(-1, -2).contiguous().reshape(T, B, N, C).contiguous()
        x = self.fc2_lif(x)
        return x
    


    
class SSA_rel_scl(nn.Module):
    def __init__(self, dim, seq_len, num_heads=8, pe=True, lif_bias=False, tau=2.0, attn='MSSA') -> None:
        super().__init__()
        
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.scale = dim ** -0.5
        self.pe = pe
        
        self.attn = attn
        
        self.q_linear = nn.Linear(dim, dim, bias=lif_bias)
        self.q_bn = nn.BatchNorm1d(dim)
        if attn == 'SSA' :  self.q_lif = MultiStepLIFNode(tau=tau, v_threshold=0.5, detach_reset=True, backend='cupy', surrogate_function=surrogate.ATan())

        self.k_linear = nn.Linear(dim, dim, bias=lif_bias)
        self.k_bn = nn.BatchNorm1d(dim)
        if attn == 'SSA' : self.k_lif = MultiStepLIFNode(tau=tau, v_threshold=0.5, detach_reset=True, backend='cupy', surrogate_function=surrogate.ATan())
        
        
        if attn != 'SSA' : self.mixer = TokenMixer(seq_len=seq_len, tau=tau, bias=lif_bias)

        self.v_linear = nn.Linear(dim, dim, bias=lif_bias)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif = MultiStepLIFNode(tau=tau, v_threshold=0.5, detach_reset=True, backend='cupy', surrogate_function=surrogate.ATan())
        
        self.attn_lif = MultiStepLIFNode(tau=tau, v_threshold=0.5, detach_reset=True, backend='cupy', surrogate_function=surrogate.ATan())

        self.proj_linear = nn.Linear(dim, dim, bias=lif_bias)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = MultiStepLIFNode(tau=tau, v_threshold=0.5, detach_reset=True, backend='cupy', surrogate_function=surrogate.ATan())
        
        if pe:
            self.relative_bias_table = nn.Parameter(torch.zeros((2 * self.seq_len -1), num_heads))
            
            coords = torch.meshgrid((torch.arange(1), torch.arange(self.seq_len)), indexing='ij')
            coords = torch.flatten(torch.stack(coords), 1)
            
            relative_coords = coords[:, :, None] - coords[:, None, :]
            relative_coords[1] += self.seq_len - 1
            relative_coords = rearrange(relative_coords, 'c h w -> h w c')
            relative_coords = relative_coords.contiguous()
            
            relative_idx = relative_coords.sum(-1).flatten().unsqueeze(1)
            self.register_buffer("relative_idx", relative_idx)
        
        # self.rel_lif = MultiStepLIFNode(tau=tau, v_threshold=0.5, detach_reset=True, backend='cupy')
        
        # self.dropout = nn.Dropout(dropout)
        # self.to_out = nn.LayerNorm(dim)
        
    def forward(self, x, mask=None):
        T,B,N,C = x.shape
        
        # T, batch_size, seq_len, _ = x.shape 
        x_for_qkv = x.flatten(0, 1)  # TB, N, D
        
        k = self.k_linear(x_for_qkv)
        if self.attn == 'MSSA' :
            k = self.k_bn(k.transpose(-1, -2).contiguous()).transpose(-1, -2).contiguous().reshape(T, B, N, C).contiguous()
        elif self.attn == 'SSA' :
            k = self.k_lif(self.k_bn(k.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, C).contiguous())
        else:
            raise ValueError
        k = k.reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        v = self.v_linear(x_for_qkv)
        # v = self.v_bn(v.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, C).contiguous()
        v = self.v_lif(self.v_bn(v.transpose(-1, -2).contiguous()).transpose(-1, -2).contiguous().reshape(T, B, N, C).contiguous())
        if self.attn =='MSSA' : v = self.mixer(v)
        v = v.reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()
        
        q = self.q_linear(x_for_qkv)
        if self.attn == 'MSSA' :
            q = self.q_bn(q.transpose(-1, -2).contiguous()).transpose(-1, -2).contiguous().reshape(T, B, N, C).contiguous()
        elif self.attn == 'SSA' :
            q = self.q_lif(self.q_bn(q.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, C).contiguous())
        else:
            raise ValueError
        q = q.reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        # q = q + attn_w.reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()
        
        attn_scores = (q @ k.transpose(-2, -1))  # attn shape = [T B head N N]

        if mask is not None:
            attn_scores = attn_scores + mask * float('-inf')
           
        if self.pe:
            relative_bias = self.relative_bias_table.gather(0, self.relative_idx.repeat(1, self.num_heads))
            relative_bias = rearrange(relative_bias, '(h w) c -> 1 c h w', h=1*self.seq_len, w=1*self.seq_len)
            
            attn_scores = attn_scores + relative_bias
        
        x = (attn_scores @ v) * self.scale
        

        x = x.transpose(2, 3).reshape(T, B, N, C).contiguous()
        x = self.attn_lif(x) 
        x = x.flatten(0, 1)
        x = self.proj_lif(self.proj_bn(self.proj_linear(x).transpose(-1, -2).contiguous()).transpose(-1, -2).contiguous().reshape(T, B, N, C).contiguous())
        return x
    

class AttentionPool1d(nn.Module):
    def __init__(self, dim, seq_len, num_heads=8, lif_bias=False, tau=2.0) -> None:
        pass


class MutualCrossAttention(nn.Module):
    def __init__(self, dim, out_seq=True, lif_bias=False, num_heads=8, tau=2.0, attn='MSSA') -> None:
        super().__init__()
        
        self.scale = dim ** -0.5
        self.num_heads = num_heads
        self.out_seq = out_seq
        
        self.attn = attn
        
        # matrix decomposition

        # self.q_lif1 = MultiStepLIFNode(tau=tau, v_threshold=0.5, detach_reset=True, backend='cupy', surrogate_function=surrogate.ATan())
        self.q_linear2 = nn.Linear(dim, dim, bias=lif_bias)
        self.q_bn2 = nn.BatchNorm1d(dim)
        if attn == 'SSA' : self.q_lif2 = MultiStepLIFNode(tau=tau, v_threshold=0.5, detach_reset=True, backend='cupy', surrogate_function=surrogate.ATan())

        self.k_linear = nn.Linear(dim, dim, bias=lif_bias)
        self.k_bn = nn.BatchNorm1d(dim)
        if attn == 'SSA' : self.k_lif = MultiStepLIFNode(tau=tau, v_threshold=0.5, detach_reset=True, backend='cupy', surrogate_function=surrogate.ATan())

        self.v_linear = nn.Linear(dim, dim, bias=lif_bias)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif = MultiStepLIFNode(tau=tau, v_threshold=0.5, detach_reset=True, backend='cupy', surrogate_function=surrogate.ATan())
        
        self.attn_lif = MultiStepLIFNode(tau=tau, v_threshold=0.5, detach_reset=True, backend='cupy', surrogate_function=surrogate.ATan())
        
        self.proj_linear = nn.Linear(dim, dim, bias=lif_bias)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = MultiStepLIFNode(tau=tau, v_threshold=0.5, detach_reset=True, backend='cupy', surrogate_function=surrogate.ATan())
        
    def forward(self, q:torch.Tensor, kv:torch.Tensor):
        T, B, Nq, D = q.shape
        T, B, Nkv, D = kv.shape
        
        q = self.q_linear2(q.flatten(0, 1))
        if self.attn == 'MSSA' : 
            q = self.q_bn2(q.transpose(-1,-2).contiguous()).transpose(-1, -2).contiguous().reshape(T, B, Nq, self.num_heads, D//self.num_heads).contiguous()
        elif self.attn == 'SSA' :
            q = self.q_lif2(self.q_bn2(q.transpose(-1,-2)).transpose(-1, -2).reshape(T, B, Nq, self.num_heads, D//self.num_heads).contiguous())
        else:
            raise ValueError
        q = q.transpose(-2, -3).contiguous() #[T B h N D//h]
        
        k = self.k_linear(kv.flatten(0, 1))
        if self.attn == 'MSSA' :
            k = self.k_bn(k.transpose(-1, -2).contiguous()).transpose(-1, -2).contiguous().reshape(T, B, Nkv, self.num_heads, D//self.num_heads).contiguous()
        elif self.attn == 'SSA' :
            k = self.k_lif(self.k_bn(k.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, Nkv, self.num_heads, D//self.num_heads).contiguous())
        else:
            raise ValueError
        k = k.transpose(-2, -3).contiguous()
        
        v = self.v_linear(kv.flatten(0, 1))
        # v = self.v_bn(v.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, Nkv, self.num_heads, D//self.num_heads).contiguous()
        v = self.v_lif(self.v_bn(v.transpose(-1, -2).contiguous()).transpose(-1, -2).contiguous().reshape(T, B, Nkv, self.num_heads, D//self.num_heads).contiguous())
        v = v.transpose(-2, -3).contiguous()
        
        k = k.transpose(0, 3).contiguous() #[1 B h T' D//h]
        v = v.transpose(0, 3).contiguous()
        
        self.attn_scores = (q @ k.transpose(-1, -2)) # [T B h N T']
        # attn_scores = self.attn_lif(self.attn_scores)
        
        z = (self.attn_scores @ v) * self.scale # [T B h N D//h]
        
        z = z.permute(3, 1, 0, 2, 4).contiguous()
        z = z.reshape(T, B, -1, D).contiguous()
        z = self.attn_lif(z.reshape(T, B, -1, D).contiguous())
        z = z.flatten(0, 1)
        z = self.proj_lif(self.proj_bn(self.proj_linear(z).transpose(-1, -2).contiguous()).transpose(-1, -2).contiguous().reshape(T, B, -1, D).contiguous())
            
        return z
    
    
class MemoryUpdate(nn.Module):
    def __init__(self, seq_len, dim, out_seq=True, lif_bias=False, num_heads=8, tau=2.0) -> None:
        super().__init__()
        
        self.scale = dim ** -0.5
        self.num_heads = num_heads
        self.out_seq = out_seq

        self.gate = nn.Linear(dim, dim, bias=lif_bias)
        self.gate_lif = MultiStepLIFNode(tau=tau, v_threshold=0.5, detach_reset=True, backend='cupy', surrogate_function=surrogate.ATan())
        
    def forward(self, q:torch.Tensor, kv:torch.Tensor):
        T, B, Nq, D = q.shape
        T, B, Nkv, D = kv.shape
        
        k = self.gate(kv.flatten(0, 1)).reshape(T, B, Nkv, D).contiguous() # 어떤 정보가 저장될 지 골라짐
        update = k.transpose(0, 2).contiguous().mean(2, keepdim=True) * q # 메모리에 저장됨
        update = self.gate_lif(update)
        # consolidation = update
        
        return update
    

class TokenMixer(nn.Module):
    def __init__(self, seq_len, tau=2.0, bias=False):
        super().__init__()
        
        self.linear1 = nn.Linear(seq_len, seq_len, bias=bias)
        self.bn1 = nn.BatchNorm1d(seq_len)
        self.lif1 = MultiStepLIFNode(tau=tau, v_threshold=0.5, detach_reset=True, backend='cupy', surrogate_function=surrogate.ATan())
        
        # self.linear2 = nn.Linear(seq_len, seq_len, bias=bias)
        # self.bn2 = nn.BatchNorm1d(seq_len)
        # self.lif2 = MultiStepLIFNode(tau=tau, v_threshold=0.5, detach_reset=True, backend='cupy', surrogate_function=surrogate.ATan())
        
    def forward(self, x):
        
        T, B, N, D = x.shape
        
        x = self.linear1(x.flatten(0, 1).transpose(-1, -2).contiguous()) # [TB N D]
        x = self.lif1(self.bn1(x.transpose(-1, -2).contiguous()).transpose(-1, -2).reshape(T, B, -1, D).contiguous())
        
        # x = self.linear2(x.flatten(0, 1).transpose(-1, -2).contiguous()) # [TB N D]
        # x = self.lif2(self.bn2(x.transpose(-1, -2).contiguous()).transpose(-1, -2).reshape(T, B, -1, D).contiguous())
        
        return x
        
    
class Consolidation(nn.Module):
    def __init__(self, dim, tau=2.0, bias=False):
        super().__init__()
        
        self.linear1 = nn.Linear(dim, dim, bias=bias)
        self.bn1 = nn.BatchNorm1d(dim)
        self.lif1 = MultiStepLIFNode(tau=tau, v_threshold=0.5, detach_reset=True, backend='cupy', surrogate_function=surrogate.ATan())
        
        
    def forward(self, x, mx):
        r"""forward

        Args:
            x (torch.tensor): temporal_emb
            mx (torch.tensor): `x_data` for teaching

        Returns:
            torch.tensor: tensor directed by `x_data`, shape = [T, B, N2, D]
        """
        T, B, N1, D = x.shape
        
        # mx = mx.clone().detach()
        mx = mx.transpose(0, 2).contiguous()
        T, B, N2, D = mx.shape

        
        mx = self.linear1(mx.flatten(0, 1))
        mx = self.lif1(self.bn1(mx.transpose(-1,-2)).transpose(-1, -2).reshape(T, B, -1, D).contiguous())
        # x = x.transpose(-2, -3).contiguous()
        
        x = x * mx
        
        return x
    
