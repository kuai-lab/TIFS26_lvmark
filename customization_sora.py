# import torch
# import torch.nn.functional as F
# from dataclasses import dataclass
# from diffusers.utils import BaseOutput
# from typing import Any, Dict, List, Optional, Tuple, Union
# from diffusers.models.unet_2d_blocks import UNetMidBlock2D, UpDecoderBlock2D, CrossAttnDownBlock2D, DownBlock2D, UNetMidBlock2DCrossAttn, UpBlock2D, CrossAttnUpBlock2D
# from diffusers.models.resnet import ResnetBlock2D
# from diffusers.models.attention import AttentionBlock
# from diffusers.models.cross_attention import CrossAttention
# from attribution import FullyConnectedLayer
# import math
# import pdb



import torch
import torch.nn.functional as F
from dataclasses import dataclass
from diffusers.utils import BaseOutput, is_torch_version
from typing import Any, Dict, List, Optional, Tuple, Union
from diffusers.models.unets.unet_2d_blocks import UNetMidBlock2D, UpDecoderBlock2D
from diffusers.models.autoencoders.vae import Decoder as SpatialDecoder
from opensora.models.vae.vae_temporal import Decoder as TemporalDecoder
from opensora.models.vae.vae_temporal import CausalConv3d, ResBlock
from diffusers.models.resnet import ResnetBlock2D
from diffusers.models.attention import Attention
from attribution import FullyConnectedLayer
import math
import pdb
from einops import rearrange


def customize_vae_decoder(vae, phi_dimension, lr_multiplier):
    def add_affine_conv(vaed):
        for layer in vaed.children():
            if type(layer) == ResnetBlock2D:
                layer.affine1 = FullyConnectedLayer(phi_dimension, layer.conv1.weight.shape[1], lr_multiplier=lr_multiplier, bias_init=1)
                layer.affine2 = FullyConnectedLayer(phi_dimension, layer.conv2.weight.shape[1], lr_multiplier=lr_multiplier, bias_init=1)
            else:
                add_affine_conv(layer)

    def add_affine_conv_in(vaed):
        for layer in vaed.children():
            if type(layer) == SpatialDecoder:
                layer.affine1 = FullyConnectedLayer(phi_dimension, layer.conv_in.weight.shape[1], lr_multiplier=lr_multiplier, bias_init=1)
            else:
                add_affine_conv_in(layer)

    def add_affine_attn(vaed):
        for layer in vaed.children():
            if type(layer) == Attention:
                layer.affine_q = FullyConnectedLayer(phi_dimension, layer.to_q.weight.shape[1], lr_multiplier=lr_multiplier, bias_init=1)
                layer.affine_k = FullyConnectedLayer(phi_dimension, layer.to_k.weight.shape[1], lr_multiplier=lr_multiplier, bias_init=1)
                layer.affine_v = FullyConnectedLayer(phi_dimension, layer.to_v.weight.shape[1], lr_multiplier=lr_multiplier, bias_init=1)
            else:
                add_affine_attn(layer)
                
    def add_affine_conv_temporal(vaed):
        for layer in vaed.children():
            if type(layer) == CausalConv3d:
                layer.affine1 = FullyConnectedLayer(phi_dimension, layer.conv.weight.shape[1], lr_multiplier=lr_multiplier, bias_init=1)
            else:
                add_affine_conv_temporal(layer)

    def change_forward(vaed, layer_type, new_forward):
        for layer in vaed.children():
            if type(layer) == layer_type:
                bound_method = new_forward.__get__(layer, layer.__class__)
                setattr(layer, 'forward', bound_method)
            else:
                change_forward(layer, layer_type, new_forward)
                
    def change_processor(vaed, layer_type, new_forward):
        for layer in vaed.children():
            if type(layer) == layer_type:
                bound_method = new_forward.__get__(layer, layer.__class__)
                setattr(layer, 'processor', bound_method)
            else:
                change_processor(layer, layer_type, new_forward)

    
    @dataclass
    class DecoderOutput(BaseOutput):
        """
        Output of decoding method.
        Args:
            sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
                Decoded output sample of the model. Output of the last layer of the model.
        """

        sample: torch.FloatTensor

    # Reference: https://github.com/huggingface/diffusers
    def new_forward_vae(self, x, enconded_fingerprint, param_threshold):
        assert self.cal_loss, "This method is only available when cal_loss is True"
        z, posterior, x_z = self.encode(x)
        x_rec, x_z_rec = self.decode(z, enconded_fingerprint, param_threshold, num_frames=x_z.shape[2])
        return x_rec, x_z_rec, z, posterior, x_z
    
    def new_decode(self, z, encoded_fingerprint, param_threshold, num_frames=None):
        if not self.cal_loss:
            z = z * self.scale.to(z.dtype) + self.shift.to(z.dtype)
        if self.micro_frame_size is None:
            x_z = self.temporal_vae.decode(z, encoded_fingerprint, param_threshold, num_frames=num_frames)
            x = self.spatial_vae.decode(x_z, encoded_fingerprint, param_threshold)
        else:
            x_z_list = []
            for i in range(0, z.size(2), self.micro_z_frame_size):
                z_bs = z[:, :, i : i + self.micro_z_frame_size]
                x_z_bs = self.temporal_vae.decode(z_bs, encoded_fingerprint, param_threshold, num_frames=min(self.micro_frame_size, num_frames))
                x_z_list.append(x_z_bs)
                num_frames -= self.micro_frame_size
            x_z = torch.cat(x_z_list, dim=2)
            x = self.spatial_vae.decode(x_z, encoded_fingerprint, param_threshold) # torch.Size([1, 4, 51, 80, 80])

        if self.cal_loss:
            return x, x_z
        else:
            return x
    
    
    def new_spatial_decode(self, x, encoded_fingerprint, param_threshold, **kwargs):
        # x: (B, C, T, H, W)
        B = x.shape[0]
        x = rearrange(x, "B C T H W -> (B T) C H W")
        if self.micro_batch_size is None:
            x = self.module.decode(x / self.scaling_factor, encoded_fingerprint, param_threshold).sample
        else:
            # NOTE: cannot be used for training
            bs = self.micro_batch_size
            x_out = []
            for i in range(0, x.shape[0], bs):
                x_bs = x[i : i + bs]
                x_bs = self.module.decode(x_bs / self.scaling_factor, encoded_fingerprint[i : i + bs, :], param_threshold).sample # x_bs: torch.Size([16, 4, 80, 80]) (micro_batch, c, h, w)
                x_out.append(x_bs)
            x = torch.cat(x_out, dim=0)
        x = rearrange(x, "(B T) C H W -> B C T H W", B=B)
        return x
    
    
    def new_spatial_module_decode(
        self, z: torch.FloatTensor, encoded_fingerprint, param_threshold, return_dict: bool = True, generator=None
    ) -> Union[DecoderOutput, torch.FloatTensor]:
        """
        Decode a batch of images.

        Args:
            z (`torch.FloatTensor`): Input batch of latent vectors.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~models.vae.DecoderOutput`] instead of a plain tuple.

        Returns:
            [`~models.vae.DecoderOutput`] or `tuple`:
                If return_dict is True, a [`~models.vae.DecoderOutput`] is returned, otherwise a plain `tuple` is
                returned.

        """
        if self.use_slicing and z.shape[0] > 1:
            decoded_slices = [self._decode(z_slice, encoded_fingerprint, param_threshold).sample for z_slice in z.split(1)]
            decoded = torch.cat(decoded_slices)
        else:
            decoded = self._decode(z, encoded_fingerprint, param_threshold).sample # z: torch.Size([16, 4, 80, 80])

        if not return_dict:
            return (decoded,)

        return DecoderOutput(sample=decoded)
    
    def new_spatial_module_decode_(self, z: torch.FloatTensor, encoded_fingerprint, param_threshold, return_dict: bool = True) -> Union[DecoderOutput, torch.FloatTensor]:
        if self.use_tiling and (z.shape[-1] > self.tile_latent_min_size or z.shape[-2] > self.tile_latent_min_size):
            return self.tiled_decode(z, return_dict=return_dict)

        z = self.post_quant_conv(z)
        dec = self.decoder(z, encoded_fingerprint, param_threshold) # z: torch.Size([16, 4, 80, 80])

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)
    
    
    def new_forward_spatial_MB(self, hidden_states, encoded_fingerprint, param_threshold, temb=None):
        hidden_states = self.resnets[0]((hidden_states, encoded_fingerprint, param_threshold), temb)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            if attn is not None:
                hidden_states = attn((hidden_states, encoded_fingerprint, param_threshold))
            hidden_states = resnet((hidden_states, encoded_fingerprint, param_threshold), temb)

        return hidden_states

    def new_forward_UDB(self, hidden_states, encoded_fingerprint, param_threshold, temb: Optional[torch.FloatTensor] = None):
        for resnet in self.resnets:
            hidden_states = resnet((hidden_states, encoded_fingerprint, param_threshold), temb=None)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)

        return hidden_states

    def new_forward_RB(self, input_tensor, temb):
        input_tensor, encoded_fingerprint, param_threshold = input_tensor
        hidden_states = input_tensor

        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)

        if self.upsample is not None:
            # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
            if hidden_states.shape[0] >= 64:
                input_tensor = input_tensor.contiguous()
                hidden_states = hidden_states.contiguous()
            input_tensor = self.upsample(input_tensor)
            hidden_states = self.upsample(hidden_states)
        elif self.downsample is not None:
            input_tensor = self.downsample(input_tensor)
            hidden_states = self.downsample(hidden_states)

        if torch.abs(self.conv1.weight).mean() < param_threshold:
            phis = self.affine1(encoded_fingerprint)
            batch_size = phis.shape[0]
            weight = phis.view(batch_size, 1, -1, 1, 1) * self.conv1.weight.unsqueeze(0)
            hidden_states = F.conv2d(hidden_states.contiguous().view(1, -1, hidden_states.shape[-2], hidden_states.shape[-1]), weight.view(-1, weight.shape[-3], weight.shape[-2], weight.shape[-1]), padding=1, groups=batch_size).view(batch_size, weight.shape[1], hidden_states.shape[-2], hidden_states.shape[-1]) + self.conv1.bias.view(1, -1, 1, 1)
        else:
            hidden_states = self.conv1(hidden_states)
        

        if temb is not None:
            temb = self.time_emb_proj(self.nonlinearity(temb))[:, :, None, None]

        if temb is not None and self.time_embedding_norm == "default":
            hidden_states = hidden_states + temb

        hidden_states = self.norm2(hidden_states)

        if temb is not None and self.time_embedding_norm == "scale_shift":
            scale, shift = torch.chunk(temb, 2, dim=1)
            hidden_states = hidden_states * (1 + scale) + shift

        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.dropout(hidden_states)

        if torch.abs(self.conv2.weight).mean() < param_threshold:
            phis = self.affine2(encoded_fingerprint)
            batch_size = phis.shape[0]
            weight = phis.view(batch_size, 1, -1, 1, 1) * self.conv2.weight.unsqueeze(0)
            hidden_states = F.conv2d(hidden_states.contiguous().view(1, -1, hidden_states.shape[-2], hidden_states.shape[-1]), weight.view(-1, weight.shape[-3], weight.shape[-2], weight.shape[-1]), padding=1, groups=batch_size).view(batch_size, weight.shape[1], hidden_states.shape[-2], hidden_states.shape[-1]) + self.conv2.bias.view(1, -1, 1, 1)
        else: 
            batch_size = encoded_fingerprint.shape[0]
            weight = self.conv2.weight.unsqueeze(0)
            hidden_states = self.conv2(hidden_states)
        

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

        return output_tensor
    
    def new_forward_spatial_Dec(
        self,
        sample: torch.FloatTensor,
        encoded_fingerprint,
        param_threshold,
        latent_embeds: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        r"""The forward method of the `Decoder` class."""
        
        # pdb.set_trace()
        if torch.abs(self.conv_in.weight).mean() < param_threshold:
            phis = self.affine1(encoded_fingerprint)
            batch_size = phis.shape[0]
            weight = phis.view(batch_size, 1, -1, 1, 1) * self.conv_in.weight.unsqueeze(0)
            sample = F.conv2d(sample.contiguous().view(1, -1, sample.shape[-2], sample.shape[-1]), weight.view(-1, weight.shape[-3], weight.shape[-2], weight.shape[-1]), padding=1, groups=batch_size).view(batch_size, weight.shape[1], sample.shape[-2], sample.shape[-1]) + self.conv_in.bias.view(1, -1, 1, 1)
        else:
            sample = self.conv_in(sample)


        upscale_dtype = next(iter(self.up_blocks.parameters())).dtype
        if self.training and self.gradient_checkpointing:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            if is_torch_version(">=", "1.11.0"):
                # middle
                sample = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.mid_block),
                    sample,
                    encoded_fingerprint,
                    param_threshold,
                    latent_embeds,
                    use_reentrant=False,
                )
                sample = sample.to(upscale_dtype)

                # up
                for up_block in self.up_blocks:
                    sample = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(up_block),
                        sample,
                        encoded_fingerprint,
                        param_threshold,
                        latent_embeds,
                        use_reentrant=False,
                    )
            else:
                # middle
                sample = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.mid_block), sample, encoded_fingerprint, param_threshold, latent_embeds
                )
                sample = sample.to(upscale_dtype)

                # up
                for up_block in self.up_blocks:
                    sample = torch.utils.checkpoint.checkpoint(create_custom_forward(up_block), sample, encoded_fingerprint, param_threshold, latent_embeds)
        else:
            # middle
            sample = self.mid_block(sample, encoded_fingerprint, param_threshold, latent_embeds)
            sample = sample.to(upscale_dtype)

            # up
            for up_block in self.up_blocks:
                sample = up_block(sample, encoded_fingerprint, param_threshold, latent_embeds)

        # post-process
        if latent_embeds is None:
            sample = self.conv_norm_out(sample)
        else:
            sample = self.conv_norm_out(sample, latent_embeds)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return sample
    
    
    
    
    
    
    def new_processor_AB(
        self,
        attn: Attention,
        hidden_states,
        encoder_hidden_states,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        
        hidden_states, encoded_fingerprint, param_threshold = hidden_states
        residual = hidden_states
        
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        
        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        # proj to q, k, v with message
        if torch.abs(attn.to_q.weight).mean() < param_threshold and torch.abs(attn.to_k.weight).mean() < param_threshold and torch.abs(attn.to_v.weight).mean() < param_threshold:
            phis_q = self.affine_q(encoded_fingerprint)
            query_proj = torch.bmm(hidden_states, phis_q.unsqueeze(-1) * attn.to_q.weight.t().unsqueeze(0)) + attn.to_q.bias

            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

            phis_k = self.affine_k(encoded_fingerprint)
            key_proj = torch.bmm(encoder_hidden_states, phis_k.unsqueeze(-1) * attn.to_k.weight.t().unsqueeze(0)) + attn.to_k.bias

            phis_v = self.affine_v(encoded_fingerprint)
            value_proj = torch.bmm(encoder_hidden_states, phis_v.unsqueeze(-1) * attn.to_v.weight.t().unsqueeze(0)) + attn.to_v.bias
        else:
            query_proj = attn.to_q(hidden_states)
            
            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
                
            key_proj = attn.to_k(encoder_hidden_states)
            value_proj = attn.to_v(encoder_hidden_states)

        inner_dim = key_proj.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query_proj.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key_proj.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value_proj.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
        
        
    ### temporal ###
    
    def new_temporal_decode(self, z, encoded_fingerprint, param_threshold, num_frames=None):
        time_padding = (
            0
            if (num_frames % self.time_downsample_factor == 0)
            else self.time_downsample_factor - num_frames % self.time_downsample_factor
        )
        z = self.post_quant_conv(z)
        x = self.decoder(z, encoded_fingerprint, param_threshold)
        x = x[:, :, time_padding:]
        return x
    
    
    def new_forward_temporal_Dec(self, x, encoded_fingerprint, param_threshold):
        x = self.conv1(x, encoded_fingerprint, param_threshold)
        for i in range(self.num_res_blocks):
            x = self.res_blocks[i](x, encoded_fingerprint, param_threshold)
        for i in reversed(range(self.num_blocks)):
            for j in range(self.num_res_blocks):
                x = self.block_res_blocks[i][j](x, encoded_fingerprint, param_threshold)
            if i > 1:
                t_stride = 2 if self.temporal_downsample[i - 1] else 1
                x = self.conv_blocks[i - 1](x, encoded_fingerprint, param_threshold)
                x = rearrange(
                    x,
                    "B (C ts hs ws) T H W -> B C (T ts) (H hs) (W ws)",
                    ts=t_stride,
                    hs=self.s_stride,
                    ws=self.s_stride,
                )

        x = self.norm1(x)
        x = self.activate(x)
        x = self.conv_out(x, encoded_fingerprint, param_threshold)
        return x
    
    
    def new_forward_temporal_RB(self, x, encoded_fingerprint, param_threshold):
        residual = x
        x = self.norm1(x)
        x = self.activate(x)
        x = self.conv1(x, encoded_fingerprint, param_threshold)
        x = self.norm2(x)
        x = self.activate(x)
        x = self.conv2(x, encoded_fingerprint, param_threshold)
        if self.in_channels != self.filters:  # SCH: ResBlock X->Y
            residual = self.conv3(residual, encoded_fingerprint, param_threshold)
        return x + residual
    
    
    def new_forward_temporal_Conv3d(self, x, encoded_fingerprint, param_threshold):
        x = F.pad(x, self.time_causal_padding, mode=self.pad_mode)
        if torch.abs(self.conv.weight).mean() < param_threshold:
        
            phis = self.affine1(encoded_fingerprint) # [8, 512]
            phis = phis.mean(dim=0) # 512
            weight = phis.view(1,1,-1,1,1,1) * self.conv.weight.unsqueeze(0) 
            x = F.conv3d(x.contiguous().view(1,-1,x.shape[-3],x.shape[-2],x.shape[-1]), weight.view(-1, weight.shape[-4], weight.shape[-3], weight.shape[-2], weight.shape[-1]), padding=(0,0,0), groups=1)
            if self.conv.bias is not None:
                x += self.conv.bias.view(1,-1,1,1,1)
        else:
            x = self.conv(x)
        
        return x
    
    
    
    # pdb.set_trace()

    add_affine_conv(vae.spatial_vae.module.decoder)
    add_affine_conv_in(vae.spatial_vae.module)
    add_affine_attn(vae.spatial_vae.module.decoder)
    add_affine_conv_temporal(vae.temporal_vae.decoder)
    
    setattr(vae, 'forward', new_forward_vae.__get__(vae, vae.__class__))
    setattr(vae, 'decode', new_decode.__get__(vae, vae.__class__))
    # pdb.set_trace()
    setattr(vae.spatial_vae, 'decode', new_spatial_decode.__get__(vae.spatial_vae, vae.spatial_vae.__class__))
    setattr(vae.spatial_vae.module, 'decode', new_spatial_module_decode.__get__(vae.spatial_vae.module, vae.spatial_vae.module.__class__))
    setattr(vae.spatial_vae.module, '_decode', new_spatial_module_decode_.__get__(vae.spatial_vae.module, vae.spatial_vae.module.__class__))
    change_forward(vae.spatial_vae.module, SpatialDecoder, new_forward_spatial_Dec)
    change_forward(vae.spatial_vae.module.decoder, UNetMidBlock2D, new_forward_spatial_MB)
    change_forward(vae.spatial_vae.module.decoder, UpDecoderBlock2D, new_forward_UDB)
    change_forward(vae.spatial_vae.module.decoder, ResnetBlock2D, new_forward_RB)
    change_processor(vae.spatial_vae.module.decoder, Attention, new_processor_AB)
    
    
    
    setattr(vae.temporal_vae, 'decode', new_temporal_decode.__get__(vae.temporal_vae, vae.temporal_vae.__class__))
    change_forward(vae.temporal_vae, TemporalDecoder, new_forward_temporal_Dec)
    change_forward(vae.temporal_vae.decoder, ResBlock, new_forward_temporal_RB)
    change_forward(vae.temporal_vae.decoder, CausalConv3d, new_forward_temporal_Conv3d)
    
    
    

    return vae