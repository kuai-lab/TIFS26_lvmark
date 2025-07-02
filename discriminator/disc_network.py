# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import torch
from torch import Tensor
from omegaconf import OmegaConf
import pdb

from discriminator.torch_utils import misc
# from discriminator.torch_utils import persistence
# from discriminator.torch_utils.ops import conv2d_resample, upfirdn2d, bias_act, fma
from discriminator.torch_utils.ops import upfirdn2d
# from discriminator.training.motion import MotionMappingNetwork
from discriminator.training.layers import (
    FullyConnectedLayer,
    GenInput,
    TemporalDifferenceEncoder,
    Conv2dLayer,
    MappingNetwork,
)


# @persistence.persistent_class
class DiscriminatorBlock(torch.nn.Module):
    def __init__(self,
        in_channels,                        # Number of input channels, 0 = first block.
        tmp_channels,                       # Number of intermediate channels.
        out_channels,                       # Number of output channels.
        resolution,                         # Resolution of this block.
        img_channels,                       # Number of input color channels.
        first_layer_idx,                    # Index of the first layer.
        architecture        = 'resnet',     # Architecture: 'orig', 'skip', 'resnet'.
        activation          = 'lrelu',      # Activation function: 'relu', 'lrelu', etc.
        resample_filter     = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp          = None,         # Clamp the output of convolution layers to +-X, None = disable clamping.
        use_fp16            = False,        # Use FP16 for this block?
        fp16_channels_last  = False,        # Use channels-last memory format with FP16?
        freeze_layers       = 0,            # Freeze-D: Number of layers to freeze.
        cfg                 = {},           # Main config.
    ):
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()

        self.cfg = cfg
        self.in_channels = in_channels
        self.resolution = resolution
        self.img_channels = img_channels
        self.first_layer_idx = first_layer_idx
        self.architecture = architecture
        self.use_fp16 = use_fp16
        self.channels_last = (use_fp16 and fp16_channels_last)
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))

        self.num_layers = 0
        def trainable_gen():
            while True:
                layer_idx = self.first_layer_idx + self.num_layers
                trainable = (layer_idx >= freeze_layers)
                self.num_layers += 1
                yield trainable
        trainable_iter = trainable_gen()
        conv0_in_channels = in_channels if in_channels > 0 else tmp_channels

        if in_channels == 0 or architecture == 'skip':
            self.fromrgb = Conv2dLayer(img_channels, tmp_channels, kernel_size=1, activation=activation,
                trainable=next(trainable_iter), conv_clamp=conv_clamp, channels_last=self.channels_last)

        self.conv0 = Conv2dLayer(conv0_in_channels, tmp_channels, kernel_size=3, activation=activation,
                trainable=next(trainable_iter), conv_clamp=conv_clamp, channels_last=self.channels_last)

        self.conv1 = Conv2dLayer(tmp_channels, out_channels, kernel_size=3, activation=activation, down=2,
            trainable=next(trainable_iter), resample_filter=resample_filter, conv_clamp=conv_clamp, channels_last=self.channels_last)

        if architecture == 'resnet':
            self.skip = Conv2dLayer(conv0_in_channels, out_channels, kernel_size=1, bias=False, down=2,
                trainable=next(trainable_iter), resample_filter=resample_filter, channels_last=self.channels_last)

    def forward(self, x, img, force_fp32=False):
        dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
        memory_format = torch.channels_last if self.channels_last and not force_fp32 else torch.contiguous_format

        # Input.
        if x is not None:
            misc.assert_shape(x, [None, self.in_channels, self.resolution, self.resolution])
            x = x.to(dtype=dtype, memory_format=memory_format)

        # FromRGB.
        if self.in_channels == 0 or self.architecture == 'skip':
            misc.assert_shape(img, [None, self.img_channels, self.resolution, self.resolution])
            img = img.to(dtype=dtype, memory_format=memory_format)
            y = self.fromrgb(img)
            x = x + y if x is not None else y
            img = upfirdn2d.downsample2d(img, self.resample_filter) if self.architecture == 'skip' else None

        # Main layers.
        if self.architecture == 'resnet':
            y = self.skip(x, gain=np.sqrt(0.5))
            x = self.conv0(x)
            x = self.conv1(x, gain=np.sqrt(0.5))
            x = y.add_(x)
        else:
            x = self.conv0(x)
            x = self.conv1(x)

        assert x.dtype == dtype
        return x, img

#----------------------------------------------------------------------------

# @persistence.persistent_class
class MinibatchStdLayer(torch.nn.Module):
    def __init__(self, group_size, num_channels=1):
        super().__init__()
        self.group_size = group_size
        self.num_channels = num_channels

    def forward(self, x):
        N, C, H, W = x.shape
        with misc.suppress_tracer_warnings(): # as_tensor results are registered as constants
            G = torch.min(torch.as_tensor(self.group_size), torch.as_tensor(N)) if self.group_size is not None else N
        F = self.num_channels
        c = C // F

        y = x.reshape(G, -1, F, c, H, W)    # [GnFcHW] Split minibatch N into n groups of size G, and channels C into F groups of size c.
        y = y - y.mean(dim=0)               # [GnFcHW] Subtract mean over group.
        y = y.square().mean(dim=0)          # [nFcHW]  Calc variance over group.
        y = (y + 1e-8).sqrt()               # [nFcHW]  Calc stddev over group.
        y = y.mean(dim=[2,3,4])             # [nF]     Take average over channels and pixels.
        y = y.reshape(-1, F, 1, 1)          # [nF11]   Add missing dimensions.
        y = y.repeat(G, 1, H, W)            # [NFHW]   Replicate over group and pixels.
        x = torch.cat([x, y], dim=1)        # [N(C+1)HW]   Append to input as new channels.
        return x

#----------------------------------------------------------------------------

# @persistence.persistent_class
class DiscriminatorEpilogue(torch.nn.Module):
    def __init__(self,
        in_channels,                    # Number of input channels.
        cmap_dim,                       # Dimensionality of mapped conditioning label, 0 = no label.
        resolution,                     # Resolution of this block.
        img_channels,                   # Number of input color channels.
        architecture        = 'resnet', # Architecture: 'orig', 'skip', 'resnet'.
        mbstd_group_size    = 4,        # Group size for the minibatch standard deviation layer, None = entire minibatch.
        mbstd_num_channels  = 1,        # Number of features for the minibatch standard deviation layer, 0 = disable.
        activation          = 'lrelu',  # Activation function: 'relu', 'lrelu', etc.
        conv_clamp          = None,     # Clamp the output of convolution layers to +-X, None = disable clamping.
        cfg                 = {},       # Architecture config.
    ):
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()

        self.cfg = cfg
        self.in_channels = in_channels
        self.cmap_dim = cmap_dim
        self.resolution = resolution
        self.img_channels = img_channels
        self.architecture = architecture

        if architecture == 'skip':
            self.fromrgb = Conv2dLayer(img_channels, in_channels, kernel_size=1, activation=activation)
        self.mbstd = MinibatchStdLayer(group_size=mbstd_group_size, num_channels=mbstd_num_channels) if mbstd_num_channels > 0 else None
        self.conv = Conv2dLayer(in_channels + mbstd_num_channels, in_channels, kernel_size=3, activation=activation, conv_clamp=conv_clamp)
        self.fc = FullyConnectedLayer(in_channels * (resolution ** 2), in_channels, activation=activation)
        self.out = FullyConnectedLayer(in_channels, 1 if cmap_dim == 0 else cmap_dim)

    def forward(self, x, img, cmap, force_fp32=False):
        misc.assert_shape(x, [None, self.in_channels, self.resolution, self.resolution]) # [NCHW]
        _ = force_fp32 # unused
        dtype = torch.float32
        memory_format = torch.contiguous_format

        # FromRGB.
        x = x.to(dtype=dtype, memory_format=memory_format)
        if self.architecture == 'skip':
            misc.assert_shape(img, [None, self.img_channels, self.resolution, self.resolution])
            img = img.to(dtype=dtype, memory_format=memory_format)
            x = x + self.fromrgb(img)

        # Main layers.
        if self.mbstd is not None:
            x = self.mbstd(x)

        x = self.conv(x)
        x = self.fc(x.flatten(1))
        x = self.out(x) # [batch_size, out_dim]

        # Conditioning.
        if self.cmap_dim > 0:
            misc.assert_shape(cmap, [None, self.cmap_dim])
            x = (x * cmap).sum(dim=1, keepdim=True) * (1 / np.sqrt(self.cmap_dim)) # [batch_size, 1]

        assert x.dtype == dtype
        return x

#----------------------------------------------------------------------------
# 0, 256, 3, 

    # disc_kwargs = dnnlib.EasyDict(
    #     c_dim                   = kwarg('label_size',           0),
    #     img_resolution          = kwarg('resolution',           256),
    #     img_channels            = kwarg('num_channels',         3),
    #     architecture            = kwarg('architecture',         'resnet'),
    #     channel_base            = kwarg('fmap_base',            16384) * 2,
    #     channel_max             = kwarg('fmap_max',             512),
    #     num_fp16_res            = kwarg('num_fp16_res',         0),
    #     conv_clamp              = kwarg('conv_clamp',           None),
    #     cmap_dim                = kwarg('mapping_fmaps',        None),
    #     block_kwargs = dnnlib.EasyDict(
    #         activation          = kwarg('nonlinearity',         'lrelu'),
    #         resample_filter     = kwarg('resample_kernel',      [1,3,3,1]),
    #         freeze_layers       = kwarg('freeze_layers',        0),
    #     ),
    #     mapping_kwargs = dnnlib.EasyDict(
    #         num_layers          = kwarg('mapping_layers',       0),
    #         embed_features      = kwarg('mapping_fmaps',        None),
    #         layer_features      = kwarg('mapping_fmaps',        None),
    #         activation          = kwarg('nonlinearity',         'lrelu'),
    #         lr_multiplier       = kwarg('mapping_lrmul',        0.1),
    #     ),
    #     epilogue_kwargs = dnnlib.EasyDict(
    #         mbstd_group_size    = kwarg('mbstd_group_size',     None),
    #         mbstd_num_channels  = kwarg('mbstd_num_features',   1),
    #         activation          = kwarg('nonlinearity',         'lrelu'),
    #     ),
    # )


# @persistence.persistent_class
class Discriminator(torch.nn.Module):
    def __init__(self,
        c_dim               = 0,        # Conditioning label (C) dimensionality.
        img_resolution      = 256,      # Input resolution.
        img_channels        = 3,        # Number of input color channels.
        architecture        = 'resnet', # Architecture: 'orig', 'skip', 'resnet'.
        channel_base        = 32768,    # Overall multiplier for the number of channels.
        channel_max         = 512,      # Maximum number of channels in any layer.
        num_fp16_res        = 0,        # Use FP16 for the N highest resolutions.
        conv_clamp          = None,     # Clamp the output of convolution layers to +-X, None = disable clamping.
        cmap_dim            = None,     # Dimensionality of mapped conditioning label, None = default.
        block_kwargs        = {
                                    'activation': 'lrelu',
                                    'resample_filter': [1, 3, 3, 1],
                                    'freeze_layers': 0,
                                },       # Arguments for DiscriminatorBlock.
        mapping_kwargs      = {
                                    'num_layers': 0,
                                    'embed_features': None,
                                    'layer_features': None,
                                    'activation': 'lrelu',
                                    'lr_multiplier': 0.1,
                                },       # Arguments for MappingNetwork.
        epilogue_kwargs     = {
                                        'mbstd_group_size': None,
                                        'mbstd_num_channels': 1,
                                        'activation': 'lrelu',
                                    },       # Arguments for DiscriminatorEpilogue.
        cfg                 = {},       # Additional config.
    ):
        super().__init__()

        self.cfg = cfg
        self.c_dim = c_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.block_resolutions = [2 ** i for i in range(self.img_resolution_log2, 2, -1)]
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions + [4]}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)
        self.cfg['sampling'] = {}
        self.cfg['sampling']['num_frames_per_video'] = 3
        self.cfg['sampling']['max_num_frames'] = 8
        self.cfg['concat_res'] = 16
        self.cfg['num_frames_div_factor'] = 2
        self.cfg['dummy_c'] = False
        self.cfg['sampling']['type'] = 'random'

        if cmap_dim is None:
            cmap_dim = channels_dict[4]

        if self.cfg['sampling']['num_frames_per_video'] > 1:
            self.time_encoder = TemporalDifferenceEncoder(self.cfg)
            assert self.time_encoder.get_dim() > 0
        else:
            self.time_encoder = None

        if self.c_dim == 0 and self.time_encoder is None:
            cmap_dim = 0

        common_kwargs = dict(img_channels=img_channels, architecture=architecture, conv_clamp=conv_clamp)
        total_c_dim = c_dim + (0 if self.time_encoder is None else self.time_encoder.get_dim())
        cur_layer_idx = 0

        for res in self.block_resolutions:
            in_channels = channels_dict[res] if res < img_resolution else 0
            tmp_channels = channels_dict[res]
            out_channels = channels_dict[res // 2]

            if res // 2 == self.cfg['concat_res']:
                out_channels = out_channels // self.cfg['num_frames_div_factor']
            if res == self.cfg['concat_res']:
                in_channels = (in_channels // self.cfg['num_frames_div_factor']) * self.cfg['sampling']['num_frames_per_video']

            use_fp16 = (res >= fp16_resolution)
            block = DiscriminatorBlock(in_channels, tmp_channels, out_channels, resolution=res,
                first_layer_idx=cur_layer_idx, use_fp16=use_fp16, cfg=self.cfg, **block_kwargs, **common_kwargs)
            setattr(self, f'b{res}', block)
            cur_layer_idx += block.num_layers

        if self.c_dim > 0 or not self.time_encoder is None:
            self.mapping = MappingNetwork(z_dim=0, c_dim=total_c_dim, w_dim=cmap_dim, num_ws=None, w_avg_beta=None, **mapping_kwargs)
        self.b4 = DiscriminatorEpilogue(channels_dict[4], cmap_dim=cmap_dim, resolution=4, cfg=self.cfg, **epilogue_kwargs, **common_kwargs)

    def forward(self, img, c, t, **block_kwargs):
        assert len(img) == t.shape[0] * t.shape[1], f"Wrong shape: {img.shape}, {t.shape}"
        assert t.ndim == 2, f"Wrong shape: {t.shape}"

        if not self.time_encoder is None:
            # Encoding the time distances
            t_embs = self.time_encoder(t.view(-1, self.cfg['sampling']['num_frames_per_video'])) # [batch_size, t_dim]

            # Concatenate `c` and time embeddings
            c = torch.cat([c, t_embs], dim=1) # [batch_size, c_dim + t_dim]
            c = (c * 0.0) if self.cfg['dummy_c'] else c # [batch_size, c_dim + t_dim]

        x = None
        for res in self.block_resolutions:
            block = getattr(self, f'b{res}')
            if res == self.cfg['concat_res']:
                # Concatenating the frames
                x = x.view(-1, self.cfg['sampling']['num_frames_per_video'], *x.shape[1:]) # [batch_size, num_frames, c, h, w]
                x = x.view(x.shape[0], -1, *x.shape[3:]) # [batch_size, num_frames * c, h, w]
            x, img = block(x, img, **block_kwargs)

        cmap = None
        if self.c_dim > 0 or not self.time_encoder is None:
            assert c.shape[1] > 0
        if c.shape[1] > 0:
            cmap = self.mapping(None, c)
        x = self.b4(x, img, cmap)
        x = x.squeeze(1) # [batch_size]

        return {'image_logits': x}

#----------------------------------------------------------------------------
