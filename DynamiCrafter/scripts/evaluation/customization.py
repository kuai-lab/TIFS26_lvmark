import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from lvdm.modules.networks.ae_modules import ResnetBlock, AttnBlock, Upsample
from attribution import FullyConnectedLayer
import math

def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)

def customize_vae_decoder(vae, phi_dimension, lr_multiplier):
    def add_affine_conv(vaed):
        for layer in vaed.children():
            if type(layer) == ResnetBlock:
                layer.affine1 = FullyConnectedLayer(phi_dimension, layer.conv1.weight.shape[1], lr_multiplier=lr_multiplier, bias_init=1)
                layer.affine_norm1 = torch.nn.LayerNorm(layer.conv1.weight.shape[1])
                layer.affine2 = FullyConnectedLayer(phi_dimension, layer.conv2.weight.shape[1], lr_multiplier=lr_multiplier, bias_init=1)
                layer.affine_norm2 = torch.nn.LayerNorm(layer.conv2.weight.shape[1])
            else:
                add_affine_conv(layer)

    def add_affine_attn(vaed):
        for layer in vaed.children():
            if type(layer) == AttnBlock:
                layer.affine_q = FullyConnectedLayer(phi_dimension, layer.q.weight.shape[1], lr_multiplier=lr_multiplier, bias_init=1)
                layer.affine_norm_q = torch.nn.LayerNorm(layer.q.weight.shape[1])
                layer.affine_k = FullyConnectedLayer(phi_dimension, layer.k.weight.shape[1], lr_multiplier=lr_multiplier, bias_init=1)
                layer.affine_norm_k = torch.nn.LayerNorm(layer.k.weight.shape[1])
                layer.affine_v = FullyConnectedLayer(phi_dimension, layer.v.weight.shape[1], lr_multiplier=lr_multiplier, bias_init=1)
                layer.affine_norm_v = torch.nn.LayerNorm(layer.v.weight.shape[1])
            else:
                add_affine_attn(layer)


    def change_forward(vaed, layer_type, new_forward):
        for layer in vaed.children():
            if type(layer) == layer_type:
                bound_method = new_forward.__get__(layer, layer.__class__)
                setattr(layer, 'forward', bound_method)
            else:
                change_forward(layer, layer_type, new_forward)

    def new_forward_RB(self, x, temb):
        x, encoded_fingerprint, layer_name, affine_list = x
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)

        if str(layer_name + "conv1.weight") in affine_list:
            phis = self.affine1(encoded_fingerprint)
            phis = self.affine_norm1(phis) + 1
            batch_size = phis.shape[0]
            weight = phis.view(batch_size, 1, -1, 1, 1) * self.conv1.weight.unsqueeze(0)
            h = F.conv2d(h.contiguous().view(1, -1, h.shape[-2], h.shape[-1]), weight.view(-1, weight.shape[-3], weight.shape[-2], weight.shape[-1]), padding=1, groups=batch_size).view(batch_size, weight.shape[1], h.shape[-2], h.shape[-1]) + self.conv1.bias.view(1, -1, 1, 1)
        else:
            h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:,:,None,None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        
        if str(layer_name + "conv2.weight") in affine_list:
            phis = self.affine2(encoded_fingerprint)
            phis = self.affine_norm2(phis) + 1
            batch_size = phis.shape[0]
            weight = phis.view(batch_size, 1, -1, 1, 1) * self.conv2.weight.unsqueeze(0)
            
            h = F.conv2d(h.contiguous().view(1, -1, h.shape[-2], h.shape[-1]), weight.view(-1, weight.shape[-3], weight.shape[-2], weight.shape[-1]), padding=1, groups=batch_size).view(batch_size, weight.shape[1], h.shape[-2], h.shape[-1]) + self.conv2.bias.view(1, -1, 1, 1)
        else:
            h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h

    def new_forward_AB(self, x):
        h_, encoded_fingerprint = x
        x = h_
        batch, channel, height, width = h_.shape
        h_ = self.norm(h_)
        h_ = h_.view(batch, channel, height * width).transpose(1, 2)

        phis_q = self.affine_q(encoded_fingerprint)
        phis_q = self.affine_norm_q(phis_q) + 1
        phis_k = self.affine_k(encoded_fingerprint)
        phis_k = self.affine_norm_k(phis_k) + 1
        phis_v = self.affine_v(encoded_fingerprint)
        phis_v = self.affine_norm_v(phis_v) + 1
        # print(self.q.weight.shape) # 512 512 1 1
        q_weight_reshaped = self.q.weight.view(channel, -1)  # (channels, channels)
        k_weight_reshaped = self.k.weight.view(channel, -1)  # (channels, channels)
        v_weight_reshaped = self.v.weight.view(channel, -1)  # (channels, channels)
                                               
        q = torch.bmm(h_, phis_q.unsqueeze(-1) * q_weight_reshaped.t().unsqueeze(0)) + self.q.bias
        k = torch.bmm(h_, phis_k.unsqueeze(-1) * k_weight_reshaped.t().unsqueeze(0)) + self.k.bias
        v = torch.bmm(h_, phis_v.unsqueeze(-1) * v_weight_reshaped.t().unsqueeze(0)) + self.v.bias

        # q = self.q(h_)
        # k = self.k(h_)
        # v = self.v(h_)

        # compute attention
        q = q.reshape(batch,channel, height, width)
        b,c,h,w = q.shape
        q = q.reshape(b,c,h*w) # bcl
        q = q.permute(0,2,1)   # bcl -> blc l=hw
        k = k.reshape(b,c,h*w) # bcl
        
        w_ = torch.bmm(q,k)    # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b,c,h*w)
        w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b,c,h,w)

        h_ = self.proj_out(h_)

        return x+h_

    def new_forward_vaed(self, z, enconded_fingerprint, affine_list):
        #assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # print(f'decoder-input={z.shape}')
        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)
        # print(f'decoder-conv in feat={h.shape}')

        # middle
        h = self.mid.block_1((h,enconded_fingerprint, "mid.block_1." ,affine_list), temb)
        h = self.mid.attn_1((h,enconded_fingerprint))
        h = self.mid.block_2((h,enconded_fingerprint, "mid.block_2." ,affine_list), temb)
        # print(f'decoder-mid feat={h.shape}')

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block]((h,enconded_fingerprint, f"up.{i_level}.block.{i_block}." ,affine_list), temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block]((h,enconded_fingerprint ,affine_list))
                # print(f'decoder-up feat={h.shape}')
            if i_level != 0:
                h = self.up[i_level].upsample(h)
                # print(f'decoder-upsample feat={h.shape}')

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        # print(f'decoder-conv_out feat={h.shape}')
        if self.tanh_out:
            h = torch.tanh(h)
        return h


    def new_decode(self, z: torch.FloatTensor, encoded_fingerprint: torch.Tensor, affine_list, **kwargs):
        z = self.post_quant_conv(z)
        dec = self.decoder(z,encoded_fingerprint,affine_list)
        return dec

    add_affine_conv(vae.decoder)
    add_affine_attn(vae.decoder)

    change_forward(vae.decoder, ResnetBlock, new_forward_RB)
    change_forward(vae.decoder, AttnBlock, new_forward_AB)

    setattr(vae.decoder, 'forward', new_forward_vaed.__get__(vae.decoder, vae.decoder.__class__))
    setattr(vae, 'decode', new_decode.__get__(vae, vae.__class__))

    return vae