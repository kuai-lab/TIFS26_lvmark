import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from math import log, pi

def exists(val):
    return val is not None

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)



class AxialRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_freq = 10):
        super().__init__()
        self.dim = dim
        scales = torch.logspace(0., log(max_freq / 2) / log(2), self.dim // 4, base = 2)
        self.register_buffer('scales', scales)

    def forward(self, h, w, device):
        scales = rearrange(self.scales, '... -> () ...')
        scales = scales.to(device)

        h_seq = torch.linspace(-1., 1., steps = h, device = device)
        h_seq = h_seq.unsqueeze(-1)

        w_seq = torch.linspace(-1., 1., steps = w, device = device)
        w_seq = w_seq.unsqueeze(-1)

        h_seq = h_seq * scales * pi
        w_seq = w_seq * scales * pi

        x_sinu = repeat(h_seq, 'i d -> i j d', j = w)
        y_sinu = repeat(w_seq, 'j d -> i j d', i = h)

        sin = torch.cat((x_sinu.sin(), y_sinu.sin()), dim = -1)
        cos = torch.cat((x_sinu.cos(), y_sinu.cos()), dim = -1)

        sin, cos = map(lambda t: rearrange(t, 'i j d -> (i j) d'), (sin, cos))
        sin, cos = map(lambda t: repeat(t, 'n d -> () n (d j)', j = 2), (sin, cos))
        return sin, cos

class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freqs = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freqs', inv_freqs)

    def forward(self, n, device):
        seq = torch.arange(n, device = device)
        freqs = einsum('i, j -> i j', seq, self.inv_freqs)
        freqs = torch.cat((freqs, freqs), dim = -1)
        freqs = rearrange(freqs, 'n d -> () n d')
        return freqs.sin(), freqs.cos()


class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)
    
def attn(q, k, v, mask = None):
    sim = einsum('b i d, b j d -> b i j', q, k)

    if exists(mask):
        max_neg_value = -torch.finfo(sim.dtype).max
        sim.masked_fill_(~mask, max_neg_value)

    attn = sim.softmax(dim = -1)
    out = einsum('b i j, b j d -> b i d', attn, v)
    return out

class CrossAttention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        dropout = 0.
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_k = nn.Linear(dim, inner_dim, bias = False)
        self.to_v = nn.Linear(dim, inner_dim, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, low, einops_from, einops_to, mask = None, rot_emb = None, **einops_dims):
        h = self.heads

        q = self.to_q(low)
        k = self.to_k(x)
        v = self.to_v(x)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        q = q * self.scale

        # rearrange across time or space
        q, k, v = map(lambda t: rearrange(t, f'{einops_from} -> {einops_to}', **einops_dims), (q, k, v))

        # expand cls token keys and values across time or space and concat
        # attention
        out = attn(q, k, v, mask = mask)
        out = rearrange(out, f'{einops_to} -> {einops_from}', **einops_dims)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)

        # combine heads out
        return self.to_out(out)

class FeatureFusion(nn.Module):
    def __init__(
        self,
        *,
        dim = 512,
        num_frames = 16,
        image_size = 128,
        patch_size = 8,
        channels = 3,
        depth = 8,
        heads = 8,
        dim_head = 64,
        attn_dropout = 0.,
        ff_dropout = 0.,
        shift_tokens = False,
    ):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_size // patch_size) ** 2
        
        num_positions = num_frames * num_patches
        patch_dim = channels * patch_size ** 2

        self.heads = heads
        self.patch_size = patch_size

        # self.pos_emb = nn.Embedding(num_positions, dim)
        self.frame_rot_emb = RotaryEmbedding(dim_head)
        self.image_rot_emb = AxialRotaryEmbedding(dim_head)

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            ff = FeedForward(dim, dropout = ff_dropout)
            time_attn = CrossAttention(dim, dim_head = dim_head, heads = heads, dropout = attn_dropout)
            spatial_attn = CrossAttention(dim, dim_head = dim_head, heads = heads, dropout = attn_dropout)
            time_attn, spatial_attn, ff = map(lambda t: PreNorm(dim, t), (time_attn, spatial_attn, ff))
            self.layers.append(nn.ModuleList([time_attn, spatial_attn, ff]))

    def forward(self, x, low, frame_mask = None):
        device = x.device
        f, hp, wp = x.size(2), x.size(3), x.size(4)
        n = hp * wp
        x = rearrange(x, 'b c f h w -> b (f h w) c')
        low  = rearrange(low, 'b c f h w -> b (f h w) c')
        # high = rearrange(high, 'b c f h w -> b (f h w) c')

        # positional embedding
        frame_pos_emb = None
        image_pos_emb = None

        # x += self.pos_emb(torch.arange(x.shape[1], device = device))
        # low  += self.pos_emb(torch.arange(x.shape[1], device = device))
        frame_pos_emb = self.frame_rot_emb(f, device = device)
        image_pos_emb = self.image_rot_emb(hp, wp, device = device)
        # high += self.pos_emb(torch.arange(x.shape[1], device = device))
        
        # time and space attention
        for (time_attn, spatial_attn, ff) in self.layers:
            x = time_attn(x, low,'b (f n) d', '(b n) f d', n = n, mask = frame_mask, rot_emb = frame_pos_emb) + x
            x = spatial_attn(x, low,'b (f n) d', '(b f) n d', f = f, rot_emb = image_pos_emb) + x
            x = ff(x) + x
        return x


class HVDM_with_Resnet_v3(nn.Module):
    def __init__(self, resnet_model, dim = 512, embed_dim = 4, num_frames = 16, image_size = 128, patch_size = 8, phi_dimension=32):
        super().__init__()

        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.dim = dim
        self.resnet = resnet_model
        self.image_size = image_size
        self.num_frames = num_frames
        self.phi_dimension = phi_dimension

        self.feature_extractor = nn.Sequential(*list(self.resnet.children())[:-2])


        self.low_freq = nn.Sequential(
        torch.nn.Conv3d(in_channels=3, out_channels=64, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),
        torch.nn.BatchNorm3d(64),
        torch.nn.GELU(),
        
        torch.nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),
        torch.nn.BatchNorm3d(128),
        torch.nn.GELU(),

        torch.nn.Conv3d(in_channels=128, out_channels=256, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),
        torch.nn.BatchNorm3d(256),
        torch.nn.GELU(),

        torch.nn.Conv3d(in_channels=256, out_channels=512, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),
        torch.nn.BatchNorm3d(512),
        torch.nn.GELU(),

        torch.nn.Conv3d(in_channels=512, out_channels=1024, kernel_size=(1,3,3), stride=1, padding=(0,1,1)),
        torch.nn.BatchNorm3d(1024),
        torch.nn.GELU(),
        
        torch.nn.Conv3d(in_channels=1024, out_channels=2048, kernel_size=(1,3,3), stride=1, padding=(0,1,1)),
        torch.nn.BatchNorm3d(2048),
        torch.nn.GELU(),
        )


        self.interaction = FeatureFusion(dim=self.dim,
                                          image_size=self.image_size,
                                          num_frames=self.num_frames,
                                          depth=2,
                                          patch_size=self.patch_size)

        self.gelu = torch.nn.GELU()
        self.pre_freq  = nn.Sequential(
            torch.nn.Conv2d(2048, 1024, 3, 2, 1),
            torch.nn.BatchNorm2d(1024),
            torch.nn.GELU(),

            torch.nn.Conv2d(1024, 1024, 3, 2, 1),
            torch.nn.BatchNorm2d(1024),
            torch.nn.GELU(),
        )

        self.pool2d = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.message_decoder = torch.nn.Linear(1024, self.phi_dimension) 

    def forward(self, x, dwt_3D): 
        
        res_hidden_states = self.feature_extractor(x)  
        res_hidden_states = res_hidden_states.permute(1,0,2,3).unsqueeze(0)

        low_freq_, high1, high2, high3, high4, high5, high6, high7 = dwt_3D
        low_freq = self.low_freq(low_freq_) 
        low_freq = low_freq.view(low_freq.shape[0], -1 ,low_freq_.shape[2], int(low_freq_.shape[-1]/16), int(low_freq_.shape[-2]/16)) # 1, 2048, 4, 8, 8
        
        # high_freq_ = torch.cat([high1, high2, high3, high4, high5, high6, high7], dim=1)        
        # high_freq = self.high_freq(high_freq_)
        # high_freq = high_freq.view(high_freq_.shape[0], -1 ,high1.shape[2], int(high1.shape[-1]/16), int(high1.shape[-2]/16))
        
        z_low_freq = low_freq.repeat_interleave(2, dim=2)
        # z_high_freq = high_freq.repeat_interleave(2, dim=2)

        cross_attention_featuremap = self.interaction(res_hidden_states, z_low_freq)
        # cross_attention_featuremap = self.interaction(res_hidden_states, z_low_freq, z_high_freq) 
        cross_attention_featuremap = self.gelu(cross_attention_featuremap)
   
        cross_attention_featuremap = rearrange(cross_attention_featuremap, 'b (t h w) c -> b c t h w', c=2048, t=int(res_hidden_states.shape[2]), h=int(res_hidden_states.shape[3]),
                                               w=int(res_hidden_states.shape[4]))
        
        # print(cross_attention_featuremap.shape)
        cross_attention_featuremap = cross_attention_featuremap.squeeze(0).permute(1,0,2,3)
        cross_attention_featuremap = self.pre_freq(cross_attention_featuremap)

        hidden_states = self.pool2d(cross_attention_featuremap)

        hidden_states = hidden_states.squeeze()
        hidden_states = self.message_decoder(hidden_states)

        return  hidden_states
    


class HVDM_with_Resnet_v5(nn.Module):
    def __init__(self,  dim = 512, embed_dim = 4, num_frames = 16, image_size = 128, patch_size = 8, phi_dimension=32):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.dim = dim
        self.image_size = image_size
        self.num_frames = num_frames
        self.phi_dimension = phi_dimension
        self.rgb = nn.Sequential(
        torch.nn.Conv3d(in_channels=3, out_channels=64, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),
        torch.nn.BatchNorm3d(64),
        torch.nn.GELU(),
        torch.nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),
        torch.nn.BatchNorm3d(128),
        torch.nn.GELU(),
        torch.nn.Conv3d(in_channels=128, out_channels=256, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),
        torch.nn.BatchNorm3d(256),
        torch.nn.GELU(),
        torch.nn.Conv3d(in_channels=256, out_channels=512, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),
        torch.nn.BatchNorm3d(512),
        torch.nn.GELU()
        )
        self.low_freq = nn.Sequential(
        torch.nn.Conv3d(in_channels=3, out_channels=64, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),
        torch.nn.BatchNorm3d(64),
        torch.nn.Tanh(),
        torch.nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),
        torch.nn.BatchNorm3d(128),
        torch.nn.Tanh(),
        torch.nn.Conv3d(in_channels=128, out_channels=256, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),
        torch.nn.BatchNorm3d(256),
        torch.nn.Tanh(),
        torch.nn.Conv3d(in_channels=256, out_channels=512, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1)),
        torch.nn.BatchNorm3d(512),
        torch.nn.Tanh()
        )
        self.interaction = FeatureFusion(dim=self.dim,
                                          image_size=self.image_size,
                                          num_frames=self.num_frames,
                                          depth=2,
                                          patch_size=self.patch_size)
        self.tanh = torch.nn.Tanh()
        self.pre_freq  = nn.Sequential(
            torch.nn.Conv2d(512, 256, 3, 1, 1),
            torch.nn.BatchNorm2d(256),
            torch.nn.GELU(),
            torch.nn.Conv2d(256, 256, 3, 1, 1),
            torch.nn.BatchNorm2d(256),
            torch.nn.GELU(),
        )
        self.pool2d = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.message_decoder = torch.nn.Linear(256, self.phi_dimension)
    def forward(self, x, dwt_3D):
        x = rearrange(x, 'f c h w -> c f h w').unsqueeze(0)
        res_hidden_states = self.rgb(x)  # 1 c f h w
        low_freq_, high1, high2, high3, high4, high5, high6, high7 = dwt_3D
        low_freq = self.low_freq(low_freq_)
        z_low_freq = low_freq.repeat_interleave(2, dim=2)

        cross_attention_featuremap = self.interaction(res_hidden_states, z_low_freq)
        cross_attention_featuremap = self.tanh(cross_attention_featuremap)
        cross_attention_featuremap = rearrange(cross_attention_featuremap, 'b (t h w) c -> b c t h w', c=self.dim, t=int(res_hidden_states.shape[2]), h=int(res_hidden_states.shape[3]),
                                               w=int(res_hidden_states.shape[4]))
        # print(cross_attention_featuremap.shape)
        cross_attention_featuremap = cross_attention_featuremap.squeeze(0).permute(1,0,2,3)
        cross_attention_featuremap = self.pre_freq(cross_attention_featuremap)
        hidden_states = self.pool2d(cross_attention_featuremap)
        hidden_states = hidden_states.squeeze()
        hidden_states = self.message_decoder(hidden_states)
        return  hidden_states



class HVDM_with_Resnet_v6(nn.Module):
    def __init__(self,  dim = 512, embed_dim = 4, num_frames = 16, image_size = 128, patch_size = 8, phi_dimension=32):
        super().__init__()

        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.dim = dim
        self.image_size = image_size
        self.num_frames = num_frames
        self.phi_dimension = phi_dimension

        self.rgb = nn.Sequential(
        torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1),
        torch.nn.BatchNorm2d(64),
        torch.nn.GELU(),
        
        torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
        torch.nn.BatchNorm2d(128),
        torch.nn.GELU(),

        torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1),
        torch.nn.BatchNorm2d(128),
        torch.nn.GELU(),

        torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
        torch.nn.BatchNorm2d(256),
        torch.nn.GELU(),

        # torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1),
        # torch.nn.BatchNorm2d(64),
        # torch.nn.GELU(),

        )

        self.low_freq = nn.Sequential(
            torch.nn.Conv3d(in_channels=3, out_channels=64, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),
            torch.nn.BatchNorm3d(64, 64),
            torch.nn.Tanh(),

            torch.nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(1,3,3), stride=1, padding=(0,1,1)),
            torch.nn.BatchNorm3d(128, 128),
            torch.nn.Tanh(),

            torch.nn.Conv3d(in_channels=128, out_channels=128, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),
            torch.nn.BatchNorm3d(128, 128),
            torch.nn.Tanh(),

        )

        self.low_freq_mid = nn.Sequential(
            torch.nn.Conv3d(in_channels=128, out_channels=256, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1)),
            torch.nn.BatchNorm3d(256, 256),
            torch.nn.Tanh(),

            # torch.nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
            # torch.nn.BatchNorm3d(64, 64),
            # torch.nn.Tanh()
            )



        self.interaction = FeatureFusion(dim=self.dim,
                                          image_size=self.image_size,
                                          num_frames=self.num_frames,
                                          depth=2,
                                          patch_size=self.patch_size)

        self.tanh = torch.nn.Tanh()
        
        self.pre_freq  = nn.Sequential(
            torch.nn.Conv3d(in_channels=256, out_channels=self.phi_dimension, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1)),
            torch.nn.BatchNorm3d(self.phi_dimension),
            torch.nn.GELU(),
        )

        self.pool2d = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.message_decoder = torch.nn.Linear(self.phi_dimension, self.phi_dimension) 

    def forward(self, x, dwt_3D): 
        # x = rearrange(x, 't c h w -> c t h w').unsqueeze(0)
        hidden_states = self.rgb(x)  # 1 c f h w # 1 512 8 128 128
        hidden_states = rearrange(hidden_states, 't c h w -> c t h w').unsqueeze(0)

        low_freq_, high1, high2, high3, high4, high5, high6, high7 = dwt_3D
        
        low_freq = self.low_freq(low_freq_)  # torch.Size([1, 512, 4, 128, 128]
        low_freq = self.low_freq_mid(low_freq)
        z_low_freq = low_freq.repeat_interleave(2, dim=2)

        cross_attention_featuremap = self.interaction(hidden_states, z_low_freq)
         
        cross_attention_featuremap = self.tanh(cross_attention_featuremap)

        cross_attention_featuremap = rearrange(cross_attention_featuremap, 'b (t h w) c -> b c t h w', c=self.dim, t=int(hidden_states.shape[2]), h=int(hidden_states.shape[3]),
                                               w=int(hidden_states.shape[4]))
        
        cross_attention_featuremap = self.pre_freq(cross_attention_featuremap)

        cross_attention_featuremap = rearrange(cross_attention_featuremap.squeeze(0), 'c t h w -> t c h w') # 8 256 16 16
        cross_attention_featuremap = self.pool2d(cross_attention_featuremap).squeeze() 

        message = self.message_decoder(cross_attention_featuremap)
      

        return  message



class HVDM_with_Resnet_v7(nn.Module):
    def __init__(self,  dim = 512, embed_dim = 4, num_frames = 16, image_size = 128, patch_size = 8, phi_dimension=32):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.dim = dim
        self.image_size = image_size
        self.num_frames = num_frames
        self.phi_dimension = phi_dimension
        self.rgb = nn.Sequential(
        torch.nn.Conv3d(in_channels=3, out_channels=128, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),
        torch.nn.BatchNorm3d(128),
        torch.nn.Tanh(),
        torch.nn.Conv3d(in_channels=128, out_channels=128, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),
        torch.nn.BatchNorm3d(128),
        torch.nn.Tanh(),
        torch.nn.Conv3d(in_channels=128, out_channels=256, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),
        torch.nn.BatchNorm3d(256),
        torch.nn.Tanh(),
        torch.nn.Conv3d(in_channels=256, out_channels=512, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),
        torch.nn.BatchNorm3d(512),
        torch.nn.Tanh()
        )
        self.low_freq = nn.Sequential(
        torch.nn.Conv3d(in_channels=3, out_channels=128, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),
        torch.nn.BatchNorm3d(128),
        torch.nn.Tanh(),
        torch.nn.Conv3d(in_channels=128, out_channels=128, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),
        torch.nn.BatchNorm3d(128),
        torch.nn.Tanh(),
        torch.nn.Conv3d(in_channels=128, out_channels=256, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),
        torch.nn.BatchNorm3d(256),
        torch.nn.Tanh(),
        torch.nn.Conv3d(in_channels=256, out_channels=512, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1)),
        torch.nn.BatchNorm3d(512),
        torch.nn.Tanh()
        )
        self.interaction = FeatureFusion(dim=self.dim,
                                          image_size=self.image_size,
                                          num_frames=self.num_frames,
                                          depth=2,
                                          patch_size=self.patch_size)
        self.tanh = torch.nn.Tanh()
        # self.pre_freq  = nn.Sequential(
        #     torch.nn.Conv3d(512, 256, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),
        #     torch.nn.BatchNorm3d(256),
        #     torch.nn.Tanh(),
        #     # torch.nn.Conv2d(256, 256, 3, 1, 1),
        #     # torch.nn.BatchNorm2d(256),
        #     # torch.nn.GELU(),
        # )
        self.pool2d = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.message_decoder = torch.nn.Linear(512, self.phi_dimension)
    def forward(self, x, dwt_3D):
        x = rearrange(x, 'f c h w -> c f h w').unsqueeze(0)
        res_hidden_states = self.rgb(x)  # 1 c f h w
        low_freq_, high1, high2, high3, high4, high5, high6, high7 = dwt_3D
        low_freq = self.low_freq(low_freq_)
        z_low_freq = low_freq.repeat_interleave(2, dim=2)

        cross_attention_featuremap = self.interaction(res_hidden_states, z_low_freq)
        cross_attention_featuremap = self.tanh(cross_attention_featuremap)
        cross_attention_featuremap = rearrange(cross_attention_featuremap, 'b (t h w) c -> b c t h w', c=self.dim, t=int(res_hidden_states.shape[2]), h=int(res_hidden_states.shape[3]),
                                               w=int(res_hidden_states.shape[4]))
        # print(cross_attention_featuremap.shape)
        # cross_attention_featuremap = self.pre_freq(cross_attention_featuremap)
        cross_attention_featuremap = cross_attention_featuremap.squeeze(0).permute(1,0,2,3)
        hidden_states = self.pool2d(cross_attention_featuremap)
        hidden_states = hidden_states.squeeze()
        hidden_states = self.message_decoder(hidden_states)
        return  hidden_states






class HVDM_with_Resnet_v8(nn.Module):
    def __init__(self, resnet_model, dim = 512, embed_dim = 4, num_frames = 16, image_size = 128, patch_size = 8):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.dim = dim
        self.resnet = resnet_model
        self.image_size = image_size
        self.num_frames = num_frames
        self.feature_extractor = nn.Sequential(*list(self.resnet.children())[:-2])
        self.pool2d = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.message_decoder = nn.Sequential(*list(self.resnet.children())[-1:])
        self.interaction = FeatureFusion(dim=self.dim,
                                          image_size=self.image_size,
                                          num_frames=self.num_frames,
                                          depth=2,
                                          patch_size=self.patch_size)
        self.low_freq = nn.Sequential(
            torch.nn.Conv3d(in_channels=3, out_channels=128, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),
            torch.nn.GroupNorm(128, 128),
            torch.nn.Tanh(),
            torch.nn.Conv3d(in_channels=128, out_channels=256, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),
            torch.nn.GroupNorm(256, 256),
            torch.nn.Tanh(),
            torch.nn.Conv3d(in_channels=256, out_channels=512, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),
            torch.nn.GroupNorm(512, 512),
            torch.nn.Tanh(),
            torch.nn.Conv3d(in_channels=512, out_channels=1024, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),
            torch.nn.GroupNorm(1024, 1024),
            torch.nn.Tanh(),
            torch.nn.Conv3d(in_channels=1024, out_channels=2048, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1)),
            torch.nn.GroupNorm(2048, 2048),
            torch.nn.Tanh(),

        )
        
    def forward(self, x, dwt_3D):
        """
        arguments:
            x : Video Diffusion으로 생성된 비디오로 (t c h w ) shape으로 입력을 받을 예정
            dwt_x : HVDM의 Low Frequency만을 가지고 만들것이기때문에 우선 입력으로 (b c t h w) 로 받고, b=1 으로 설정할 예정.
        return :
            output으로, [t,phi_dimension] shape으로 Message를 복원하는 과정으로 최종출력을 진행할 예정
            1) ResNet50을 기반으로 Feature-Map을 [t,c',h',w']을 뽑고,
            2) HVDM의 Frequency Encoder를 통해서 [b, t, c h/2, w/2] 생성.
            이후 1,2의 Output을 통해 연산을 진행하여, Decoder를 생성하는 방식으로 해야할듯 .
        """
        res_hidden_states = self.feature_extractor(x)
        res_hidden_states = res_hidden_states.permute(1,0,2,3).unsqueeze(0)
        low_freq = self.low_freq(dwt_3D)
        # low_freq = self.low_freq_mid(low_freq)
        z_low_freq = low_freq.repeat_interleave(2, dim=2)
        cross_attention_featuremap = self.interaction(res_hidden_states, z_low_freq)
        cross_attention_featuremap = torch.tanh(cross_attention_featuremap)
        cross_attention_featuremap = rearrange(cross_attention_featuremap, 'b (t h w) c -> b c t h w', c=2048, t=int(res_hidden_states.shape[2]), h=int(res_hidden_states.shape[3]),
                                               w=int(res_hidden_states.shape[4]))
        hidden_states = self.pool2d(cross_attention_featuremap.squeeze(0).permute(1,0,2,3))
        hidden_states = hidden_states.squeeze()
        hidden_states = self.message_decoder(hidden_states)
        return  hidden_states
    
class HVDM_with_Resnet_v9(nn.Module):
    def __init__(self, resnet_model, dim = 512, embed_dim = 4, num_frames = 16, image_size = 128, patch_size = 8, phi_dimension=3, fusion_depth=2):
        super().__init__()

        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.dim = dim
        self.resnet = resnet_model
        self.image_size = image_size
        self.num_frames = num_frames
        self.phi_dimension = phi_dimension
        self.fusion_depth = fusion_depth

        self.feature_extractor = nn.Sequential(*list(self.resnet.children())[:-2])
        self.message_decoder = nn.Sequential(*list(self.resnet.children())[-1:])

        self.low_freq = nn.Sequential(
        torch.nn.Conv3d(in_channels=3, out_channels=64, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),
        torch.nn.GroupNorm(64,64),
        torch.nn.Tanh(),
        
        torch.nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),
        torch.nn.GroupNorm(128,128),
        torch.nn.Tanh(),

        torch.nn.Conv3d(in_channels=128, out_channels=256, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),
        torch.nn.GroupNorm(256,256),
        torch.nn.Tanh(),

        torch.nn.Conv3d(in_channels=256, out_channels=512, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),
        torch.nn.GroupNorm(512,512),
        torch.nn.Tanh(),

        torch.nn.Conv3d(in_channels=512, out_channels=1024, kernel_size=(1,3,3), stride=1, padding=(0,1,1)),
        torch.nn.GroupNorm(1024,1024),
        torch.nn.Tanh(),
        
        torch.nn.Conv3d(in_channels=1024, out_channels=2048, kernel_size=(1,3,3), stride=1, padding=(0,1,1)),
        torch.nn.GroupNorm(2048,2048),
        torch.nn.Tanh(),
        )


        self.interaction = FeatureFusion(dim=self.dim,
                                          image_size=self.image_size,
                                          num_frames=self.num_frames,
                                          depth=self.fusion_depth,
                                          patch_size=self.patch_size)

        self.tanh = torch.nn.Tanh()

        self.pool2d = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, x, dwt_3D): 
        
        res_hidden_states = self.feature_extractor(x)  
        res_hidden_states = res_hidden_states.permute(1,0,2,3).unsqueeze(0)

        low_freq_, high1, high2, high3, high4, high5, high6, high7 = dwt_3D
        low_freq = self.low_freq(low_freq_) 
        low_freq = low_freq.view(low_freq.shape[0], -1 ,low_freq_.shape[2], int(low_freq_.shape[-1]/16), int(low_freq_.shape[-2]/16)) # 1, 2048, 4, 8, 8
        
        z_low_freq = low_freq.repeat_interleave(2, dim=2)

        cross_attention_featuremap = self.interaction(res_hidden_states, z_low_freq)
        cross_attention_featuremap = self.tanh(cross_attention_featuremap)
   
        cross_attention_featuremap = rearrange(cross_attention_featuremap, 'b (t h w) c -> b c t h w', c=2048, t=int(res_hidden_states.shape[2]), h=int(res_hidden_states.shape[3]),
                                               w=int(res_hidden_states.shape[4]))
        
        cross_attention_featuremap = cross_attention_featuremap.squeeze(0).permute(1,0,2,3)

        hidden_states = self.pool2d(cross_attention_featuremap)

        hidden_states = hidden_states.squeeze()
        hidden_states = self.message_decoder(hidden_states)

        return  hidden_states
    




class HVDM_with_Resnet_v11(nn.Module):
    def __init__(self,  dim = 512, embed_dim = 4, num_frames = 16, image_size = 128, patch_size = 8, phi_dimension=32, fusion_depth = 2):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.dim = dim
        self.image_size = image_size
        self.num_frames = num_frames
        self.phi_dimension = phi_dimension
        self.fusion_depth = fusion_depth

        self.rgb = nn.Sequential(
        torch.nn.Conv3d(in_channels=3, out_channels=64, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),
        torch.nn.GroupNorm(64,64),
        torch.nn.Tanh(),
        
        torch.nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1)),
        torch.nn.GroupNorm(128,128),
        torch.nn.Tanh(),

        torch.nn.Conv3d(in_channels=128, out_channels=128, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),
        torch.nn.GroupNorm(128,128),
        torch.nn.Tanh(),

        torch.nn.Conv3d(in_channels=128, out_channels=128, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1)),
        torch.nn.GroupNorm(128,128),
        torch.nn.Tanh(),

        torch.nn.Conv3d(in_channels=128, out_channels=128, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),
        torch.nn.GroupNorm(128,128),
        torch.nn.Tanh(),

        torch.nn.Conv3d(in_channels=128, out_channels=128, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1)),
        torch.nn.GroupNorm(128,128),
        torch.nn.Tanh(),

        torch.nn.Conv3d(in_channels=128, out_channels=256, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),
        torch.nn.GroupNorm(256,256),
        torch.nn.Tanh(),

        )

        self.low_freq = nn.Sequential(
        torch.nn.Conv3d(in_channels=3, out_channels=64, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),
        torch.nn.GroupNorm(64,64),
        torch.nn.Tanh(),
        
        torch.nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1)),
        torch.nn.GroupNorm(128,128),
        torch.nn.Tanh(),

        torch.nn.Conv3d(in_channels=128, out_channels=128, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),
        torch.nn.GroupNorm(128,128),
        torch.nn.Tanh(),

        torch.nn.Conv3d(in_channels=128, out_channels=128, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1)),
        torch.nn.GroupNorm(128,128),
        torch.nn.Tanh(),

        torch.nn.Conv3d(in_channels=128, out_channels=256, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),
        torch.nn.GroupNorm(256,256),
        torch.nn.Tanh(),
        )
        self.interaction = FeatureFusion(dim=self.dim,
                                          image_size=self.image_size,
                                          num_frames=self.num_frames,
                                          depth=self.fusion_depth,
                                          patch_size=self.patch_size)
        self.tanh = torch.nn.Tanh()

        self.pool2d = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.message_decoder = torch.nn.Linear(self.dim, self.phi_dimension)

    def forward(self, x, dwt_3D):
        x = rearrange(x, 'f c h w -> c f h w').unsqueeze(0)
        res_hidden_states = self.rgb(x)  # 1 c f h w
        low_freq_, high1, high2, high3, high4, high5, high6, high7 = dwt_3D
        low_freq = self.low_freq(low_freq_)
        z_low_freq = low_freq.repeat_interleave(2, dim=2)

        cross_attention_featuremap = self.interaction(res_hidden_states, z_low_freq)
        cross_attention_featuremap = self.tanh(cross_attention_featuremap)
        cross_attention_featuremap = rearrange(cross_attention_featuremap, 'b (t h w) c -> b c t h w', c=self.dim, t=int(res_hidden_states.shape[2]), h=int(res_hidden_states.shape[3]),
                                               w=int(res_hidden_states.shape[4]))
        
        # cross_attention_featuremap = self.pre_freq(cross_attention_featuremap)
        cross_attention_featuremap = cross_attention_featuremap.squeeze(0).permute(1,0,2,3)
        hidden_states = self.pool2d(cross_attention_featuremap)
        hidden_states = hidden_states.squeeze()
        hidden_states = self.message_decoder(hidden_states)
        return  hidden_states

########################################################################################################

class HVDM_with_Resnet_v12_resnet(nn.Module):
    def __init__(self, resnet_model, dim = 512, embed_dim = 4, num_frames = 16, image_size = 128, patch_size = 8, phi_dimension=3, fusion_depth=2):
        super().__init__()

        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.dim = dim
        self.resnet = resnet_model
        self.image_size = image_size
        self.num_frames = num_frames
        self.phi_dimension = phi_dimension
        self.fusion_depth = fusion_depth

        self.feature_extractor = nn.Sequential(*list(self.resnet.children())[:-2])
        self.message_decoder = nn.Sequential(*list(self.resnet.children())[-1:])

        self.low_freq = nn.Sequential(
        torch.nn.Conv3d(in_channels=3, out_channels=64, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),
        torch.nn.GroupNorm(64,64),
        torch.nn.Tanh(),
        
        torch.nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),
        torch.nn.GroupNorm(128,128),
        torch.nn.Tanh(),

        torch.nn.Conv3d(in_channels=128, out_channels=256, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),
        torch.nn.GroupNorm(256,256),
        torch.nn.Tanh(),

        torch.nn.Conv3d(in_channels=256, out_channels=512, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),
        torch.nn.GroupNorm(512,512),
        torch.nn.Tanh(),

        torch.nn.Conv3d(in_channels=512, out_channels=1024, kernel_size=(1,3,3), stride=1, padding=(0,1,1)),
        torch.nn.GroupNorm(1024,1024),
        torch.nn.Tanh(),
        
        torch.nn.Conv3d(in_channels=1024, out_channels=2048, kernel_size=(1,3,3), stride=1, padding=(0,1,1)),
        torch.nn.GroupNorm(2048,2048),
        torch.nn.Tanh(),
        )


        self.interaction = FeatureFusion(dim=self.dim,
                                          image_size=self.image_size,
                                          num_frames=self.num_frames,
                                          depth=self.fusion_depth,
                                          patch_size=self.patch_size)

        self.tanh = torch.nn.Tanh()

        self.pool2d = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, x, dwt_3D): 
        x = rearrange(x, 'b c f h w -> (b f) c h w')
        res_hidden_states = self.feature_extractor(x)
        res_hidden_states = rearrange(res_hidden_states, '(b f) c h w -> b c f h w', f=self.num_frames)  
        # res_hidden_states = res_hidden_states.permute(1,0,2,3)

        low_freq_, high1, high2, high3, high4, high5, high6, high7 = dwt_3D
        low_freq = self.low_freq(low_freq_) 
        low_freq = low_freq.view(low_freq.shape[0], -1 ,low_freq_.shape[2], int(low_freq_.shape[-1]/16), int(low_freq_.shape[-2]/16)) # 1, 2048, 4, 8, 8
        
        z_low_freq = low_freq.repeat_interleave(2, dim=2)
        # print(res_hidden_states.shape)
        # print(z_low_freq.shape)
        cross_attention_featuremap = self.interaction(res_hidden_states, z_low_freq)
        cross_attention_featuremap = self.tanh(cross_attention_featuremap)
   
        cross_attention_featuremap = rearrange(cross_attention_featuremap, 'b (t h w) c -> b c t h w', c=2048, t=int(res_hidden_states.shape[2]), h=int(res_hidden_states.shape[3]),
                                               w=int(res_hidden_states.shape[4]))
        

        hidden_states = self.pool2d(cross_attention_featuremap)

        hidden_states = rearrange(hidden_states.squeeze(), 'b c t -> (b t) c')
        hidden_states = self.message_decoder(hidden_states)

        return  hidden_states
    

class HVDM_with_Resnet_v12_resnet18(nn.Module):
    def __init__(self, resnet_model, dim = 512, embed_dim = 4, num_frames = 16, image_size = 128, patch_size = 8, phi_dimension=3, fusion_depth=2):
        super().__init__()

        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.dim = dim
        self.resnet = resnet_model
        self.image_size = image_size
        self.num_frames = num_frames
        self.phi_dimension = phi_dimension
        self.fusion_depth = fusion_depth

        self.feature_extractor = nn.Sequential(*list(self.resnet.children())[:-2])
        self.message_decoder = nn.Sequential(*list(self.resnet.children())[-1:])

        self.low_freq = nn.Sequential(
        torch.nn.Conv3d(in_channels=3, out_channels=64, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),
        torch.nn.GroupNorm(64,64),
        torch.nn.Tanh(),
        
        torch.nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),
        torch.nn.GroupNorm(128,128),
        torch.nn.Tanh(),

        torch.nn.Conv3d(in_channels=128, out_channels=128, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),
        torch.nn.GroupNorm(128,128),
        torch.nn.Tanh(),

        torch.nn.Conv3d(in_channels=128, out_channels=128, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),
        torch.nn.GroupNorm(128,128),
        torch.nn.Tanh(),

        torch.nn.Conv3d(in_channels=128, out_channels=256, kernel_size=(1,3,3), stride=1, padding=(0,1,1)),
        torch.nn.GroupNorm(256,256),
        torch.nn.Tanh(),
        
        torch.nn.Conv3d(in_channels=256, out_channels=512, kernel_size=(1,3,3), stride=1, padding=(0,1,1)),
        torch.nn.GroupNorm(512,512),
        torch.nn.Tanh(),
        )


        self.interaction = FeatureFusion(dim=self.dim,
                                          image_size=self.image_size,
                                          num_frames=self.num_frames,
                                          depth=self.fusion_depth,
                                          patch_size=self.patch_size)

        self.tanh = torch.nn.Tanh()

        self.pool2d = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, x, dwt_3D): 
        x = rearrange(x, 'b c f h w -> (b f) c h w')
        # b, c, f, h, w = x.shape
        # x = x.view()
        res_hidden_states = self.feature_extractor(x)
        res_hidden_states = rearrange(res_hidden_states, '(b f) c h w -> b c f h w', f=self.num_frames)  
        # res_hidden_states = res_hidden_states.permute(1,0,2,3)

        low_freq_, high1, high2, high3, high4, high5, high6, high7 = dwt_3D
        low_freq = self.low_freq(low_freq_) 
        low_freq = low_freq.view(low_freq.shape[0], -1 ,low_freq_.shape[2], int(low_freq_.shape[-1]/16), int(low_freq_.shape[-2]/16)) # 1, 2048, 4, 8, 8
        
        z_low_freq = low_freq.repeat_interleave(2, dim=2)
        # print(res_hidden_states.shape)
        # print(z_low_freq.shape)
        cross_attention_featuremap = self.interaction(res_hidden_states, z_low_freq)
        cross_attention_featuremap = self.tanh(cross_attention_featuremap)
   
        cross_attention_featuremap = rearrange(cross_attention_featuremap, 'b (t h w) c -> b c t h w', c=self.dim, t=int(res_hidden_states.shape[2]), h=int(res_hidden_states.shape[3]),
                                               w=int(res_hidden_states.shape[4]))
        

        hidden_states = self.pool2d(cross_attention_featuremap)
        # print(hidden_states.shape)
        hidden_states = rearrange(hidden_states.squeeze(3,4), 'b c t -> (b t) c')
        hidden_states = self.message_decoder(hidden_states)

        return  hidden_states



class HVDM_with_Resnet_v12_resnet34(nn.Module):
    def __init__(self, resnet_model, dim = 512, embed_dim = 4, num_frames = 16, image_size = 128, patch_size = 8, phi_dimension=3, fusion_depth=2):
        super().__init__()

        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.dim = dim
        self.resnet = resnet_model
        self.image_size = image_size
        self.num_frames = num_frames
        self.phi_dimension = phi_dimension
        self.fusion_depth = fusion_depth

        self.feature_extractor = nn.Sequential(*list(self.resnet.children())[:-2])
        self.message_decoder = nn.Sequential(*list(self.resnet.children())[-1:])

        self.low_freq = nn.Sequential(
        torch.nn.Conv3d(in_channels=3, out_channels=64, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),
        torch.nn.GroupNorm(64,64),
        torch.nn.Tanh(),
        
        torch.nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),
        torch.nn.GroupNorm(128,128),
        torch.nn.Tanh(),

        torch.nn.Conv3d(in_channels=128, out_channels=128, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),
        torch.nn.GroupNorm(128,128),
        torch.nn.Tanh(),

        torch.nn.Conv3d(in_channels=128, out_channels=128, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),
        torch.nn.GroupNorm(128,128),
        torch.nn.Tanh(),

        torch.nn.Conv3d(in_channels=128, out_channels=256, kernel_size=(1,3,3), stride=1, padding=(0,1,1)),
        torch.nn.GroupNorm(256,256),
        torch.nn.Tanh(),
        
        torch.nn.Conv3d(in_channels=256, out_channels=512, kernel_size=(1,3,3), stride=1, padding=(0,1,1)),
        torch.nn.GroupNorm(512,512),
        torch.nn.Tanh(),
        )


        self.interaction = FeatureFusion(dim=self.dim,
                                          image_size=self.image_size,
                                          num_frames=self.num_frames,
                                          depth=self.fusion_depth,
                                          patch_size=self.patch_size)

        self.tanh = torch.nn.Tanh()

        self.pool2d = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, x, dwt_3D): 
        b, c, f, h, w = x.shape
        x = rearrange(x, 'b c f h w -> (b f) c h w')
        # x = x.view()
        res_hidden_states = self.feature_extractor(x)
        res_hidden_states = rearrange(res_hidden_states, '(b f) c h w -> b c f h w', f=f)  
        # res_hidden_states = res_hidden_states.permute(1,0,2,3)

        low_freq_, high1, high2, high3, high4, high5, high6, high7 = dwt_3D
        low_freq = self.low_freq(low_freq_) 
        # low_freq = low_freq.view(low_freq.shape[0], -1 ,low_freq_.shape[2], int(low_freq_.shape[-1]/16), int(low_freq_.shape[-2]/16)) # 1, 2048, 4, 8, 8
        
        z_low_freq = low_freq.repeat_interleave(2, dim=2)
        # print(res_hidden_states.shape)
        # print(z_low_freq.shape)
        cross_attention_featuremap = self.interaction(res_hidden_states, z_low_freq)
        cross_attention_featuremap = self.tanh(cross_attention_featuremap)
   
        cross_attention_featuremap = rearrange(cross_attention_featuremap, 'b (t h w) c -> b c t h w', c=self.dim, t=int(res_hidden_states.shape[2]), h=int(res_hidden_states.shape[3]),
                                               w=int(res_hidden_states.shape[4]))
        

        hidden_states = self.pool2d(cross_attention_featuremap)
        # print(hidden_states.shape)
        hidden_states = rearrange(hidden_states.squeeze(3,4), 'b c t -> (b t) c')
        hidden_states = self.message_decoder(hidden_states)

        return  hidden_states




class HVDM_with_Resnet_v12_resnet50(nn.Module):
    def __init__(self, resnet_model, dim = 512, embed_dim = 4, num_frames = 16, image_size = 128, patch_size = 8, phi_dimension=3, fusion_depth=2):
        super().__init__()

        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.dim = dim
        self.resnet = resnet_model
        self.image_size = image_size
        self.num_frames = num_frames
        self.phi_dimension = phi_dimension
        self.fusion_depth = fusion_depth

        self.feature_extractor = nn.Sequential(*list(self.resnet.children())[:-2])
        self.message_decoder = nn.Sequential(*list(self.resnet.children())[-1:])

        self.low_freq = nn.Sequential(
        torch.nn.Conv3d(in_channels=3, out_channels=64, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),
        torch.nn.GroupNorm(64,64),
        torch.nn.Tanh(),
        
        torch.nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),
        torch.nn.GroupNorm(128,128),
        torch.nn.Tanh(),

        torch.nn.Conv3d(in_channels=128, out_channels=128, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),
        torch.nn.GroupNorm(128,128),
        torch.nn.Tanh(),

        torch.nn.Conv3d(in_channels=128, out_channels=128, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),
        torch.nn.GroupNorm(128,128),
        torch.nn.Tanh(),

        torch.nn.Conv3d(in_channels=128, out_channels=256, kernel_size=(1,3,3), stride=1, padding=(0,1,1)),
        torch.nn.GroupNorm(256,256),
        torch.nn.Tanh(),
        
        torch.nn.Conv3d(in_channels=256, out_channels=512, kernel_size=(1,3,3), stride=1, padding=(0,1,1)),
        torch.nn.GroupNorm(512,512),
        torch.nn.Tanh(),

        torch.nn.Conv3d(in_channels=512, out_channels=1024, kernel_size=(1,3,3), stride=1, padding=(0,1,1)),
        torch.nn.GroupNorm(1024,1024),
        torch.nn.Tanh(),

        torch.nn.Conv3d(in_channels=1024, out_channels=2048, kernel_size=(1,3,3), stride=1, padding=(0,1,1)),
        torch.nn.GroupNorm(2048,2048),
        torch.nn.Tanh(),
        )


        self.interaction = FeatureFusion(dim=self.dim,
                                          image_size=self.image_size,
                                          num_frames=self.num_frames,
                                          depth=self.fusion_depth,
                                          patch_size=self.patch_size)

        self.tanh = torch.nn.Tanh()

        self.pool2d = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, x, dwt_3D): 
        b, c, f, h, w = x.shape
        x = rearrange(x, 'b c f h w -> (b f) c h w')
        # x = x.view()
        res_hidden_states = self.feature_extractor(x)
        res_hidden_states = rearrange(res_hidden_states, '(b f) c h w -> b c f h w', f=f)  
        # res_hidden_states = res_hidden_states.permute(1,0,2,3)

        low_freq_, high1, high2, high3, high4, high5, high6, high7 = dwt_3D
        low_freq = self.low_freq(low_freq_) 
        low_freq = low_freq.view(low_freq.shape[0], -1 ,low_freq_.shape[2], int(low_freq_.shape[-1]/16), int(low_freq_.shape[-2]/16)) # 1, 2048, 4, 8, 8
        
        z_low_freq = low_freq.repeat_interleave(2, dim=2)
        # print(res_hidden_states.shape)
        # print(z_low_freq.shape)
        cross_attention_featuremap = self.interaction(res_hidden_states, z_low_freq)
        cross_attention_featuremap = self.tanh(cross_attention_featuremap)
   
        cross_attention_featuremap = rearrange(cross_attention_featuremap, 'b (t h w) c -> b c t h w', c=self.dim, t=int(res_hidden_states.shape[2]), h=int(res_hidden_states.shape[3]),
                                               w=int(res_hidden_states.shape[4]))
        

        hidden_states = self.pool2d(cross_attention_featuremap)
        # print(hidden_states.shape)
        hidden_states = rearrange(hidden_states.squeeze(3,4), 'b c t -> (b t) c')
        hidden_states = self.message_decoder(hidden_states)

        return  hidden_states






class HVDM_with_Resnet50(nn.Module):
    def __init__(self, resnet_model, dim = 512, embed_dim = 4, num_frames = 16, image_size = 128, patch_size = 8, phi_dimension=32):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.dim = dim
        self.resnet = resnet_model
        self.image_size = image_size
        self.num_frames = num_frames
        self.phi_dimension = phi_dimension
        self.feature_extractor = nn.Sequential(*list(self.resnet.children())[:-2])
        self.message_decoder = nn.Sequential(*list(self.resnet.children())[-1:])

        self.low_freq = nn.Sequential(
        torch.nn.Conv3d(in_channels=3, out_channels=64, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),
        torch.nn.GroupNorm(64,64),
        torch.nn.Tanh(),
        torch.nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),
        torch.nn.GroupNorm(128,128),
        torch.nn.Tanh(),
        torch.nn.Conv3d(in_channels=128, out_channels=256, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),
        torch.nn.GroupNorm(256,256),
        torch.nn.Tanh(),
        torch.nn.Conv3d(in_channels=256, out_channels=512, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),
        torch.nn.GroupNorm(512,512),
        torch.nn.Tanh(),
        torch.nn.Conv3d(in_channels=512, out_channels=1024, kernel_size=(1,3,3), stride=1, padding=(0,1,1)),
        torch.nn.GroupNorm(1024,1024),
        torch.nn.Tanh(),
        torch.nn.Conv3d(in_channels=1024, out_channels=2048, kernel_size=(1,3,3), stride=1, padding=(0,1,1)),
        torch.nn.GroupNorm(2048,2048),
        torch.nn.Tanh(),
        )
        self.interaction = FeatureFusion(dim=self.dim,
                                          image_size=self.image_size,
                                          num_frames=self.num_frames,
                                          depth=2,
                                          patch_size=self.patch_size)
        self.tanh = torch.nn.Tanh()
        self.pool2d = nn.AdaptiveAvgPool2d(output_size=(1, 1))
    def forward(self, x, dwt_3D):
        b, c, f, h, w = x.shape
        x = rearrange(x, 'b c f h w -> (b f) c h w')
        res_hidden_states = self.feature_extractor(x)
        res_hidden_states = rearrange(res_hidden_states, '(b f) c h w -> b c f h w', f=f)  

        low_freq_, high1, high2, high3, high4, high5, high6, high7 = dwt_3D
        low_freq = self.low_freq(low_freq_)
        # low_freq = low_freq.view(low_freq.shape[0], -1 ,dwt_3D.shape[2], int(dwt_3D.shape[-1]/16), int(dwt_3D.shape[-2]/16)) # 1, 2048, 4, 8, 8
        z_low_freq = low_freq.repeat_interleave(2, dim=2)
        cross_attention_featuremap = self.interaction(res_hidden_states, z_low_freq)
        cross_attention_featuremap = self.tanh(cross_attention_featuremap)
        cross_attention_featuremap = rearrange(cross_attention_featuremap, 'b (t h w) c -> b c t h w', c=self.dim, t=int(res_hidden_states.shape[2]), h=int(res_hidden_states.shape[3]),
                                               w=int(res_hidden_states.shape[4]))
        cross_attention_featuremap = cross_attention_featuremap.squeeze(0).permute(1,0,2,3)
        hidden_states = self.pool2d(cross_attention_featuremap)
        hidden_states = hidden_states.squeeze()
        hidden_states = self.message_decoder(hidden_states)
        return  hidden_states



class HVDM_with_Resnet50_high_freq(nn.Module):
    def __init__(self, resnet_model, dim = 512, embed_dim = 4, num_frames = 16, image_size = 128, patch_size = 8, phi_dimension=32):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.dim = dim
        self.resnet = resnet_model
        self.image_size = image_size
        self.num_frames = num_frames
        self.phi_dimension = phi_dimension
        self.feature_extractor = nn.Sequential(*list(self.resnet.children())[:-2])
        self.message_decoder = nn.Sequential(*list(self.resnet.children())[-1:])

        self.high_freq_block = nn.Sequential(
        torch.nn.Conv3d(in_channels=21, out_channels=64, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),
        torch.nn.GroupNorm(64,64),
        torch.nn.Tanh(),
        torch.nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),
        torch.nn.GroupNorm(128,128),
        torch.nn.Tanh(),
        torch.nn.Conv3d(in_channels=128, out_channels=256, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),
        torch.nn.GroupNorm(256,256),
        torch.nn.Tanh(),
        torch.nn.Conv3d(in_channels=256, out_channels=512, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),
        torch.nn.GroupNorm(512,512),
        torch.nn.Tanh(),
        torch.nn.Conv3d(in_channels=512, out_channels=1024, kernel_size=(1,3,3), stride=1, padding=(0,1,1)),
        torch.nn.GroupNorm(1024,1024),
        torch.nn.Tanh(),
        torch.nn.Conv3d(in_channels=1024, out_channels=2048, kernel_size=(1,3,3), stride=1, padding=(0,1,1)),
        torch.nn.GroupNorm(2048,2048),
        torch.nn.Tanh(),
        )
        self.interaction = FeatureFusion(dim=self.dim,
                                          image_size=self.image_size,
                                          num_frames=self.num_frames,
                                          depth=2,
                                          patch_size=self.patch_size)
        self.tanh = torch.nn.Tanh()
        self.pool2d = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        
    def forward(self, x, dwt_3D):
        b, c, f, h, w = x.shape
        x = rearrange(x, 'b c f h w -> (b f) c h w')
        res_hidden_states = self.feature_extractor(x)
        res_hidden_states = rearrange(res_hidden_states, '(b f) c h w -> b c f h w', f=f)  

        low_freq_, high1, high2, high3, high4, high5, high6, high7 = dwt_3D
        concatenated_highs = torch.cat([high1, high2, high3, high4, high5, high6, high7], dim=1)  # [1, 21, 4, 128, 128]
        high_freq = self.high_freq_block(concatenated_highs)
        z_high_freq = high_freq.repeat_interleave(2, dim=2)
        cross_attention_featuremap = self.interaction(res_hidden_states, z_high_freq)
        cross_attention_featuremap = self.tanh(cross_attention_featuremap)
        cross_attention_featuremap = rearrange(cross_attention_featuremap, 'b (t h w) c -> b c t h w', c=self.dim, t=int(res_hidden_states.shape[2]), h=int(res_hidden_states.shape[3]),
                                               w=int(res_hidden_states.shape[4]))
        cross_attention_featuremap = cross_attention_featuremap.squeeze(0).permute(1,0,2,3)
        hidden_states = self.pool2d(cross_attention_featuremap)
        hidden_states = hidden_states.squeeze()
        hidden_states = self.message_decoder(hidden_states)
        return  hidden_states
    
    
class HVDM_with_Resnet50_2d_DWT(nn.Module):
    def __init__(self, resnet_model, dim = 512, embed_dim = 4, num_frames = 16, image_size = 128, patch_size = 8, phi_dimension=32):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.dim = dim
        self.resnet = resnet_model
        self.image_size = image_size
        self.num_frames = num_frames
        self.phi_dimension = phi_dimension
        self.feature_extractor = nn.Sequential(*list(self.resnet.children())[:-2])
        self.message_decoder = nn.Sequential(*list(self.resnet.children())[-1:])

        self.low_freq = nn.Sequential(
        torch.nn.Conv3d(in_channels=3, out_channels=64, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),
        torch.nn.GroupNorm(64,64),
        torch.nn.Tanh(),
        torch.nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),
        torch.nn.GroupNorm(128,128),
        torch.nn.Tanh(),
        torch.nn.Conv3d(in_channels=128, out_channels=256, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),
        torch.nn.GroupNorm(256,256),
        torch.nn.Tanh(),
        torch.nn.Conv3d(in_channels=256, out_channels=512, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),
        torch.nn.GroupNorm(512,512),
        torch.nn.Tanh(),
        torch.nn.Conv3d(in_channels=512, out_channels=1024, kernel_size=(1,3,3), stride=1, padding=(0,1,1)),
        torch.nn.GroupNorm(1024,1024),
        torch.nn.Tanh(),
        torch.nn.Conv3d(in_channels=1024, out_channels=2048, kernel_size=(1,3,3), stride=1, padding=(0,1,1)),
        torch.nn.GroupNorm(2048,2048),
        torch.nn.Tanh(),
        )
        self.interaction = FeatureFusion(dim=self.dim,
                                          image_size=self.image_size,
                                          num_frames=self.num_frames,
                                          depth=2,
                                          patch_size=self.patch_size)
        self.tanh = torch.nn.Tanh()
        self.pool2d = nn.AdaptiveAvgPool2d(output_size=(1, 1))
    def forward(self, x, dwt_2D):
        # pdb.set_trace()
        b, c, f, h, w = x.shape # torch.Size([1, 3, 8, 256, 256])
        x = rearrange(x, 'b c f h w -> (b f) c h w')
        res_hidden_states = self.feature_extractor(x) # torch.Size([8, 2048, 8, 8])
        res_hidden_states = rearrange(res_hidden_states, '(b f) c h w -> b c f h w', f=f)   # torch.Size([1, 2048, 8, 8, 8])

        # low_freq_, high1, high2, high3, high4, high5, high6, high7 = dwt_3D # torch.Size([1, 3, 4, 128, 128])
        z_low_freq = self.low_freq(dwt_2D) # torch.Size([1, 2048, 4, 8, 8])
        # low_freq = low_freq.view(low_freq.shape[0], -1 ,dwt_3D.shape[2], int(dwt_3D.shape[-1]/16), int(dwt_3D.shape[-2]/16)) # 1, 2048, 4, 8, 8
        # z_low_freq = low_freq.repeat_interleave(2, dim=2) # torch.Size([1, 2048, 8, 8, 8])
        cross_attention_featuremap = self.interaction(res_hidden_states, z_low_freq) # torch.Size([1, 512, 2048])
        cross_attention_featuremap = self.tanh(cross_attention_featuremap)
        cross_attention_featuremap = rearrange(cross_attention_featuremap, 'b (t h w) c -> b c t h w', c=self.dim, t=int(res_hidden_states.shape[2]), h=int(res_hidden_states.shape[3]),
                                               w=int(res_hidden_states.shape[4])) # torch.Size([1, 2048, 8, 8, 8])
        cross_attention_featuremap = cross_attention_featuremap.squeeze(0).permute(1,0,2,3) # torch.Size([8, 2048, 8, 8])
        hidden_states = self.pool2d(cross_attention_featuremap) # torch.Size([8, 2048, 1, 1])
        hidden_states = hidden_states.squeeze() # torch.Size([8, 2048])
        hidden_states = self.message_decoder(hidden_states) # torch.Size([8, 48])
        return  hidden_states





class HVDM_with_Resnet50_just_concat(nn.Module):
    def __init__(self, resnet_model, dim = 512, embed_dim = 4, num_frames = 16, image_size = 128, patch_size = 8, phi_dimension=32):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.dim = dim
        self.resnet = resnet_model
        self.image_size = image_size
        self.num_frames = num_frames
        self.phi_dimension = phi_dimension
        self.feature_extractor = nn.Sequential(*list(self.resnet.children())[:-2])
        self.message_decoder = nn.Sequential(*list(self.resnet.children())[-1:])

        self.low_freq = nn.Sequential(
        torch.nn.Conv3d(in_channels=3, out_channels=64, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),
        torch.nn.GroupNorm(64,64),
        torch.nn.Tanh(),
        torch.nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),
        torch.nn.GroupNorm(128,128),
        torch.nn.Tanh(),
        torch.nn.Conv3d(in_channels=128, out_channels=256, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),
        torch.nn.GroupNorm(256,256),
        torch.nn.Tanh(),
        torch.nn.Conv3d(in_channels=256, out_channels=512, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),
        torch.nn.GroupNorm(512,512),
        torch.nn.Tanh(),
        torch.nn.Conv3d(in_channels=512, out_channels=1024, kernel_size=(1,3,3), stride=1, padding=(0,1,1)),
        torch.nn.GroupNorm(1024,1024),
        torch.nn.Tanh(),
        torch.nn.Conv3d(in_channels=1024, out_channels=2048, kernel_size=(1,3,3), stride=1, padding=(0,1,1)),
        torch.nn.GroupNorm(2048,2048),
        torch.nn.Tanh(),
        )
        self.interaction = FeatureFusion(dim=self.dim,
                                          image_size=self.image_size,
                                          num_frames=self.num_frames,
                                          depth=2,
                                          patch_size=self.patch_size)
        self.tanh = torch.nn.Tanh()
        self.pool2d = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.channel_reduction = nn.Conv2d(4096, 2048, kernel_size=1) # 4096 ->2048 
    def forward(self, x, dwt_3D):
        b, c, f, h, w = x.shape
        x = rearrange(x, 'b c f h w -> (b f) c h w')
        res_hidden_states = self.feature_extractor(x)
        res_hidden_states = rearrange(res_hidden_states, '(b f) c h w -> b c f h w', f=f)  

        low_freq_, high1, high2, high3, high4, high5, high6, high7 = dwt_3D
        low_freq = self.low_freq(low_freq_)
        # low_freq = low_freq.view(low_freq.shape[0], -1 ,dwt_3D.shape[2], int(dwt_3D.shape[-1]/16), int(dwt_3D.shape[-2]/16)) # 1, 2048, 4, 8, 8
        z_low_freq = low_freq.repeat_interleave(2, dim=2) # 1, 2048, 8, 8, 8
        


        concat_hidden_states = torch.concat([res_hidden_states, z_low_freq],dim=1)
        concat_hidden_states = rearrange(concat_hidden_states,'b c f h w ->(b f) c h w')
        concat_hidden_states = self.channel_reduction(concat_hidden_states)
        
        hidden_states = self.pool2d(concat_hidden_states)
        hidden_states = hidden_states.squeeze()
        hidden_states = self.message_decoder(hidden_states)
        return  hidden_states