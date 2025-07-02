#!/usr/bin/env python
# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import logging
import math
import os
import random
from pathlib import Path
from typing import List, Optional, Union
from omegaconf import OmegaConf

import numpy as np
import torch
import torch.nn.functional as F

from collections import OrderedDict
import datasets
import diffusers
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset, Image
from diffusers import AutoencoderKL, StableDiffusionPipeline, UNet2DConditionModel, EulerDiscreteScheduler
from diffusers.optimization import get_scheduler
# from diffusers.utils import randn_tensor
from diffusers.utils.import_utils import is_xformers_available
from huggingface_hub import HfFolder, Repository, create_repo, whoami
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights, resnet34, ResNet34_Weights
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

import itertools
from attribution import MappingNetwork
from customization import customize_vae_decoder
import inspect
from torchvision.utils import save_image
import lpips
import wandb
from attack_methods.attack_initializer_h264 import attack_initializer #For augmentation
import hydra
from hydra import compose, initialize
from accelerate.utils.dataclasses import DistributedDataParallelKwargs
from data_utils import get_video_dataloader
from utils.utils import instantiate_from_config
from einops import rearrange
from loss.loss_provider import LossProvider
from decoder.hvdm_decoder import HVDM_with_Resnet50
from decoder.utils import DWT_3D
from discriminator.training.layers import sample_frames

logger = get_logger(__name__, log_level="INFO")


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    
    parser.add_argument(
        "--n_frames",
        type=int,
        default=8,
        help="number of frames",
    )
    parser.add_argument(
        "--val_dir",
        type=str,
        default='./',
        help="dataset path",
    )
    parser.add_argument(
        "--train_dir",
        type=str,
        default='./',
        help="dataset path",
    )
    
    parser.add_argument(
        "--steps",
        type=int,
        default=10000,
        help="train steps",
    )
    
    parser.add_argument(
        "--exp_name",
        type=str,
        default="exp_1",
        help="Name of the experiment",
    )
    parser.add_argument(
        "--lr_mult",
        type=float,
        default=1,
        help="Learning rate multiplier for the affine layers",
    )
    parser.add_argument(
        "--pre_latents",
        type=str,
        default=None,
        help="Path to pre-extracted latents for validation",
    )
    parser.add_argument(
        "--phi_dimension",
        type=int,
        default=32,
        help="phi_dimension",
    )
    parser.add_argument(
        "--int_dimension",
        type=int,
        default=128,
        help="intermediate dimension",
    )
    parser.add_argument(
        "--mapping_layer",
        type=int,
        default=2,
        help="FC layers of mapping network",
    )
    parser.add_argument(
        "--attack",
        type=str,
        default='',
        help=(
            "which attack methods to apply ('c' | 'r' | 'g' | 'b' | 'n' | 'e' | 'j' | ... | 'crgbnej' or 'all' || 'AE_b_1' | 'AE_c_6' | ...)"
            "e.g. 'cr' denotes random cropping ('c') and rotation ('r')"
            "Use 'crgbnej' or 'all' for the combined attack in the paper"
        ),
    )
    parser.add_argument(
        "--num_gradient_from_last",
        type=int,
        default=1,
        help="number of getting gradient from last in the denoising loop",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stabilityai/stable-diffusion-2-base",
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="HuggingFaceM4/COCO",
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing an image."
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--train_steps_per_epoch",
        type=int,
        default=1000,
        help="Number of training steps per epoch. If provided, limits the number of iterations for each epoch",
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=50000,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine_with_restarts",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--cosine_cycle",
        type=int,
        default=1000,
        help=(
            "cosine_with_restarts option for cycle"
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=1000,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )

    parser.add_argument(
        "--model_config",
        type=str,
        default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )

    parser.add_argument(
        "--dim",
        type=int,
        default=512,
        help="Dimension of Attention",
    )
    
    parser.add_argument(
        "--lambda_w",
        type=float,
        default=1.0,
        help="Dimension of Attention",
    )

    parser.add_argument(
        "--lambda_l2",
        type=float,
        default=1.0,
        help="Dimension of Attention",
    )
    parser.add_argument(
        "--lambda_patch_l1",
        type=float,
        default=1.0,
        help="Dimension of Attention",
    )
    parser.add_argument(
        "--lambda_lpips",
        type=float,
        default=1.0,
        help="Dimension of Attention",
    )
    
    parser.add_argument(
        "--weight_modulation_rate",
        type=float,
        default=0.8,
        help="weight modulation rate",
    )

    parser.add_argument(
        "--fusion_depth",
        type=int,
        default=2,
        help="Dimension of Attention",
    )
    
    parser.add_argument(
        "--affine_list_path",
        type=str,
        default="",
        help=(
            'affine_list_path'
        ),
    )
    
    # affine_list_path

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args


def get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"


dataset_name_mapping = {
    'lambdalabs/pokemon-blip-captions': ('image', 'text'),
    'HuggingFaceM4/COCO': ('image', 'sentences_raw'),
    'imagenet-1k': ('image', 'label')
}


def get_phis(args, phi_dimension, batch_size ,eps = 1e-8):
    phi_length = phi_dimension
    b = batch_size
    phi = torch.empty(b,args.n_frames,phi_length).uniform_(0,1) # b k
    phi = torch.bernoulli(phi) + eps # b k
    return phi

def check_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def prepare_extra_step_kwargs(generator, eta, scheduler):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs


def prepare_latents(batch_size, num_channels_latents, height, width, dtype, device, generator, vae_scale_factor, scheduler, latents=None):
    shape = (batch_size, num_channels_latents, height // vae_scale_factor, width // vae_scale_factor)
    if isinstance(generator, list) and len(generator) != batch_size:
        raise ValueError(
            f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
            f" size of {batch_size}. Make sure the batch size matches the length of the generators."
        )

    if latents is None:
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
    else:
        latents = latents.to(device)

    # scale the initial noise by the standard deviation required by the scheduler
    latents = latents * scheduler.init_noise_sigma
    return latents

def load_model_checkpoint(model, ckpt):
    state_dict = torch.load(ckpt, map_location="cpu")
    if "state_dict" in list(state_dict.keys()):
        state_dict = state_dict["state_dict"]
        try:
            model.load_state_dict(state_dict, strict=True)
        except:
            ## rename the keys for 256x256 model
            new_pl_sd = OrderedDict()
            for k,v in state_dict.items():
                new_pl_sd[k] = v

            for k in list(new_pl_sd.keys()):
                if "framestride_embed" in k:
                    new_key = k.replace("framestride_embed", "fps_embedding")
                    new_pl_sd[new_key] = new_pl_sd[k]
                    del new_pl_sd[k]
            model.load_state_dict(new_pl_sd, strict=True)
    else:
        # deepspeed
        new_pl_sd = OrderedDict()
        for key in state_dict['module'].keys():
            new_pl_sd[key[16:]]=state_dict['module'][key]
        model.load_state_dict(new_pl_sd)
    print('>>> model checkpoint loaded.')
    return model


def decode_latents(vae, latents, enconded_fingerprint, affine_list):
    # latents = 1 / 0.18215 * latents
    image = accelerator.unwrap_model(vae).decode(latents, enconded_fingerprint, affine_list)
    # image = image.clamp(-1,1)
    return image

def get_params_optimize(vaed, mapping_network, decoding_network):
    params_to_optimize = itertools.chain(vaed.parameters(), mapping_network.parameters(), decoding_network.parameters())
    return params_to_optimize


def acc_calculation(args, phis, decoding_network, generated_image, generated_image_dwt, bsz = None, vae = None):
    reconstructed_keys = decoding_network(generated_image, generated_image_dwt)
    gt_phi = (phis > 0.5).int()
    reconstructed_keys = (torch.sigmoid(reconstructed_keys) > 0.5).int()
    bit_acc = ((gt_phi == reconstructed_keys).sum(dim=1)) / args.phi_dimension

    return bit_acc


def load_val_latents(args, batch_size, val_step):
    val_latents = None
    step = val_step*args.train_batch_size
    for i in range(batch_size):
        vl = torch.load(os.path.join(args.pre_latents, f'{step+i}.pth'))
        if val_latents is None:
            val_latents = vl.unsqueeze(0)
        else:
            val_latents = torch.cat((val_latents, vl.unsqueeze(0)), 0)

    return val_latents
from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights, resnet34, ResNet34_Weights

def psnr(x, y, img_space='img'): # pred, gt
    """ 
    Return PSNR 
    Args:
        x: Image tensor with values approx. between [-1,1]
        y: Image tensor with values approx. between [-1,1], ex: original image
    """
    
    # unnormalize_img = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])
    unnormalize_img = transforms.Normalize(mean=[-1, -1, -1], std=[1/0.5, 1/0.5, 1/0.5]) # Unnormalize (x * 0.5) + 0.5
    delta = torch.clamp(unnormalize_img(x), 0, 1) - torch.clamp(unnormalize_img(y), 0, 1)
    delta = 255 * delta
    # delta = delta.reshape(-1, x.shape[-3], x.shape[-2], x.shape[-1]) # BxCxHxW
    psnr = 20*np.log10(255) - 10*torch.log10(torch.mean(delta**2, dim=(1,2,3)))  # B
    return psnr

def patch_l1_loss(diff):
    patch_size = 8
    f, c, h, w = diff.shape
    # diff = -torch.log(diff + 1e-8) 

    # Unfold the tensor to get patches
    unfolded = diff.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)

    # Shape of unfolded will be (f, c, num_h_patches, num_w_patches, patch_size, patch_size)
    # Now, calculate the mean for each patch (over patch_size * patch_size dimensions)
    patch_means = unfolded.mean(dim=(-1, -2))  # Shape: (f, c, num_h_patches, num_w_patches)

    # Reshape the patch means to (f, c, num_patches) where num_patches = num_h_patches * num_w_patches
    num_patches = (h // patch_size) * (w // patch_size)
    patch_means_flat = patch_means.view(f, c, num_patches)
    
    spatial_att = F.softmax(patch_means_flat, dim=-1) * (h / patch_size)
    spatial_att = spatial_att.view(f, c, h // patch_size, h // patch_size)
    return (patch_means * spatial_att).mean()
    

def val(args, epoch, step, accelerator, weight_dtype,  vae, mapping_network, decoding_network, test_dataloader, valid_aug, resize, metrics, affine_list):
    os.makedirs(os.path.join(args.output_dir,'sample_images'), exist_ok=True)
    list_validation = []

    list_valid_psnr = []
    
    #Change network eval mode
    try:
        vae.eval()
    except:
        vae.module.eval()
    
    mapping_network.eval()
    decoding_network.eval()

    dwt = DWT_3D("haar").to(accelerator.device)
    with torch.no_grad():
        for val_step, batch in enumerate(test_dataloader):
            if (val_step+1)*args.train_batch_size >= 5000:
                break
            # Convert images to latent space
            
            bsz = batch.shape[0]
            batch = rearrange(batch, 'b c f h w -> (b f) c h w').contiguous()
            latents = accelerator.unwrap_model(vae).encode(batch.to(weight_dtype).to(accelerator.device)).sample()
            # latents = latents * 0.18215

            # Sample noise that we'll add to the latents

            # Sampling fingerprints and get the embeddings
            phis = get_phis(args, args.phi_dimension, bsz).to(latents.device)
            phis = rearrange(phis, 'b f k -> (b f) k')
            encoded_phis = mapping_network(phis)

            #Training's validation
            generated_image_latent_0 = resize(decode_latents(vae, latents, encoded_phis, affine_list)) # bf c h w
            
            if 'AE_' in args.attack:
                augmented_image_latent_0 = valid_aug.forward((generated_image_latent_0 / 2 + 0.5).clamp(0, 1))['x_hat'].clamp(0, 1)
            else:
                augmented_image_latent_0 = valid_aug((generated_image_latent_0 / 2 + 0.5).clamp(0, 1))
                
            augmented_image_latent_0 = rearrange(augmented_image_latent_0, '(b f) c h w -> b c f h w', b = bsz, f = args.n_frames)
            
            augmented_wavelet =  dwt(augmented_image_latent_0) # b c f h w
            
            list_validation.extend(acc_calculation(args, phis, decoding_network, augmented_image_latent_0, augmented_wavelet,bsz, vae).tolist())

            psnr_ = psnr(generated_image_latent_0, batch).mean().item()
            list_valid_psnr.append(psnr_)


    # for metric in metrics:
    #     metric.print_results(accelerator, epoch, step)

    #Saving Image
    generated_image_latent_0 = rearrange(generated_image_latent_0, '(b f) c h w -> b f c h w', b= bsz, f = args.n_frames)
    generated_image = (generated_image_latent_0 / 2 + 0.5).clamp(0, 1)
    # save_image(generated_image, '{0}/sample_images/sample_e_{1}_s_{2}.png'.format(args.output_dir,epoch,step),normalize=True, value_range=(0,1), scale_each=True)
    wandb.log({"Examples_val_1": wandb.Image(generated_image[0][:, [2, 1, 0], :, :], caption="sample_e_{0}_s_{1}".format(epoch, step))})
    wandb.log({"Val Acc": float(np.mean(list_validation))})
    wandb.log({"PSNR_val": float(np.mean(list_valid_psnr))})

def main():
    args = parse_args()
    wandb.init(name=args.exp_name, project="WOUAF")
    args.output_dir = os.path.join(args.output_dir, args.exp_name)
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    with initialize(version_base=None, config_path="configs", job_name="test_app"):
        cfg_hydra = compose(config_name="metrics", overrides=[])

    metrics = []
    for _, cb_conf in cfg_hydra.items():
        metrics.append(hydra.utils.instantiate(cb_conf))

    os.environ['WANDB_DISABLE_SERVICE'] = 'true'

    global accelerator
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_dir=logging_dir,
        kwargs_handlers=[kwargs]
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            create_repo(repo_name, exist_ok=True, token=args.hub_token)
            repo = Repository(args.output_dir, clone_from=repo_name, token=args.hub_token)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load scheduler, tokenizer and models.
    config = OmegaConf.load(args.model_config)
    model_config = config.pop("model", OmegaConf.create())

    ## set use_checkpoint as False as when using deepspeed, it encounters an error "deepspeed backend not set"
    model_config['params']['unet_config']['params']['use_checkpoint'] = False
    model = instantiate_from_config(model_config)
    model = model.cuda()
    model.perframe_ae = True
    model = load_model_checkpoint(model, args.pretrained_model_name_or_path)
    vae = model.first_stage_model
    vae.eval()

    vae.requires_grad_(False)
    vae.decoder.requires_grad_(True)
    
    del model

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    mapping_network = MappingNetwork(args.phi_dimension, args.int_dimension, num_layers=args.mapping_layer)
    resnet50_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    resnet50_model.fc = torch.nn.Linear(args.dim, args.phi_dimension)
    decoding_network = HVDM_with_Resnet50(resnet_model = resnet50_model, dim = args.dim, num_frames = args.n_frames, image_size = args.resolution, phi_dimension = args.phi_dimension)
    
    # loss_fn_vgg = lpips.LPIPS(net='vgg').to(accelerator.device)
    provider = LossProvider()
    loss_percep = provider.get_loss_function('Watson-VGG', colorspace='RGB', pretrained=True, reduction='sum')
    loss_percep = loss_percep.to(accelerator.device)
    loss_fn_vgg = lambda imgs_w, imgs: loss_percep((1+imgs_w)/2.0, (1+imgs)/2.0)/ imgs_w.shape[0]
    
    # # Weight modulation to vae's decoder
    vae = customize_vae_decoder(vae, args.int_dimension, args.lr_mult)

    affine_list = []
    with open(args.affine_list_path, 'r') as file:
        for line in file:
            affine_list.append(line.strip())

    # # For mixed precision training we cast the vae weights to half-precision
    # # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # # Move text_encode and vae to gpu and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    mapping_network.to(accelerator.device, dtype=weight_dtype)
    decoding_network.to(accelerator.device, dtype=weight_dtype)

    optimizer = optimizer_cls(
        get_params_optimize(vae.decoder, mapping_network, decoding_network),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # # Get the datasets: you can either provide your own training and evaluation files (see below)

    # Preprocessing the datasets.
    train_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
            transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
            transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    test_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
            transforms.CenterCrop(args.resolution),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    # Data load
    print("Data loading")
    train_dataloader = get_video_dataloader(train_transforms, args.train_dir, args.resolution, args.train_batch_size, n_frames=args.n_frames, num_imgs=10000, shuffle=True, num_workers=args.dataloader_num_workers)
    test_dataloader = get_video_dataloader(test_transforms, args.val_dir, args.resolution, args.train_batch_size, n_frames=args.n_frames, num_imgs=10, shuffle=False, num_workers=args.dataloader_num_workers)
    
    # # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.train_steps_per_epoch is None:
        train_steps_per_epoch = num_update_steps_per_epoch
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * train_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
        num_cycles = args.cosine_cycle * args.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    vae, mapping_network, decoding_network, optimizer, train_dataloader, test_dataloader, lr_scheduler = accelerator.prepare(
        vae, mapping_network, decoding_network, optimizer, train_dataloader, test_dataloader, lr_scheduler
    )


    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.train_steps_per_epoch is None:
        args.train_steps_per_epoch = num_update_steps_per_epoch
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * args.train_steps_per_epoch


    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / args.train_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("text2image-fine-tune", config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    # logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    #Augmentation
    train_aug = attack_initializer(args, is_train = True, device = accelerator.device)
    valid_aug = attack_initializer(args, is_train = False, device = accelerator.device)

    if args.resolution == 256:
        # resize = transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True)
        resize = torch.nn.Identity()
    else:
        resize = torch.nn.Identity()

    # Setup all metrics
    for metric in metrics:
        metric.setup(accelerator, args)

    dwt = DWT_3D("haar").to(accelerator.device)
    
    for epoch in range(first_epoch, args.num_train_epochs):        
        try:
            vae.module.decoder.train()
        except:
            vae.decoder.train()

        mapping_network.train()
        decoding_network.train()

        local_step = 0
        train_loss = 0.0
        list_train_bit_acc = []

        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(accelerator.unwrap_model(vae).decoder), accelerator.accumulate(mapping_network), accelerator.accumulate(decoding_network):
                # Convert images to latent space
                bsz = batch.shape[0]
                batch = rearrange(batch, 'b c f h w -> (b f) c h w').contiguous()
                latents = accelerator.unwrap_model(vae).encode(batch.to(weight_dtype).to(accelerator.device)).sample()

                # Sampling fingerprints and get the embeddings
                phis = get_phis(args, args.phi_dimension, bsz).to(latents.device) # b f k
                phis = rearrange(phis, 'b f k -> (b f) k')
                
                encoded_phis = mapping_network(phis)
                
                generated_image = resize(decode_latents(vae, latents, encoded_phis, affine_list)) # bf c h w                
                
                if 'AE_' in args.attack:
                    augmented_image = train_aug.forward((generated_image / 2 + 0.5).clamp(0, 1))['x_hat'].clamp(0, 1)
                else:
                    augmented_image = train_aug((generated_image / 2 + 0.5).clamp(0, 1))

                augmented_image = rearrange(augmented_image, '(b f) c h w -> b c f h w', b = bsz, f = args.n_frames)
                augmented_dwt = dwt(augmented_image)

                reconstructed_keys = decoding_network(augmented_image , augmented_dwt) # (b f) k

                #Key reconstruction loss = Element-wise BCE
                loss_key = F.binary_cross_entropy_with_logits(reconstructed_keys, phis)
                # loss_lpips_reg = loss_fn_vgg.forward(generated_image, resize(batch)).mean()
                # loss_lpips_reg = loss_fn_vgg(generated_image ,batch)
                loss_lpips_reg = loss_fn_vgg(generated_image ,batch) / 10
                
                diff = torch.abs(generated_image - batch)
                loss_patch_l1 = patch_l1_loss(diff)
                
                loss = args.lambda_w * loss_key  + args.lambda_lpips * loss_lpips_reg + args.lambda_patch_l1 * loss_patch_l1
                
                #Calculate batch accuracy
                gt_phi = (phis > 0.5).int()
                reconstructed_keys = (torch.sigmoid(reconstructed_keys) > 0.5).int()
                bit_acc = ((gt_phi == reconstructed_keys).sum(dim=1)) / args.phi_dimension
                list_train_bit_acc.append(bit_acc.mean().item())

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(get_params_optimize(accelerator.unwrap_model(vae).decoder, mapping_network, decoding_network), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()


            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                local_step += 1
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                accelerator.log({"key_loss": loss_key.item()}, step=global_step)
                accelerator.log({"loss_patch_l1": loss_patch_l1.item()}, step=global_step)
                accelerator.log({"lpips_reg": loss_lpips_reg.item()}, step=global_step)
                accelerator.log({"bit_acc": torch.mean(torch.tensor(list_train_bit_acc)).item()}, step=global_step)
                accelerator.log({"psnr": psnr(generated_image ,batch).mean().item()}, step=global_step)
                train_loss = 0.0

                if (global_step % args.checkpointing_steps) == 0:
                    os.makedirs('./' + args.output_dir + "/ckpt_"+str(global_step), exist_ok=True)
                    torch.save(accelerator.unwrap_model(vae).state_dict(), args.output_dir + "/ckpt_"+str(global_step) + "/vae_decoder.pth")
                    torch.save(mapping_network.state_dict(), args.output_dir + "/ckpt_"+str(global_step) + "/mapping_network.pth")
                    torch.save(decoding_network.state_dict(), args.output_dir + "/ckpt_"+str(global_step) + "/decoding_network.pth")

            logs = {"bit_acc" : bit_acc.mean().item(),
                    "psnr": psnr(generated_image ,batch).mean().item(),
                    "train_loss": loss.item(),
                    "loss_key": loss_key.detach().item(),
                    "loss_patch_l1": loss_patch_l1.item(),
                    "loss_lpips":loss_lpips_reg.item(),
                    "lr": lr_scheduler.get_last_lr()[0]}

            wandb.log({"Train train_loss": loss.item()})
            wandb.log({"loss_key": loss_key.detach().item()})
            wandb.log({"loss_lpips": loss_lpips_reg.item()})
            wandb.log({"loss_patch_l1": loss_patch_l1.item()})
            wandb.log({"bit_acc": bit_acc.mean().item()})
            wandb.log({"psnr": psnr(generated_image ,batch).mean().item()})     

            progress_bar.set_postfix(**logs)

            if local_step >= args.train_steps_per_epoch or global_step >= args.max_train_steps:
                break


        train_acc = torch.mean(torch.tensor(list_train_bit_acc))
        print("Training Acc: Bit-wise Acc in Epoch {0}: {1}".format(epoch, train_acc))
        wandb.log({"Train Acc": train_acc.item()})

        val(args, epoch, step, accelerator, weight_dtype, vae, mapping_network, decoding_network, test_dataloader, valid_aug, resize, metrics, affine_list)
        if global_step >= args.max_train_steps:
            break

    # # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        os.makedirs('./' + args.output_dir + "/latest", exist_ok=True)
        torch.save(accelerator.unwrap_model(vae).state_dict(), args.output_dir + "/latest" + "/vae_decoder.pth")
        torch.save(mapping_network.state_dict(), args.output_dir + "/latest" + "/mapping_network.pth")
        torch.save(decoding_network.state_dict(), args.output_dir + "/latest" + "/decoding_network.pth")
        
        if args.push_to_hub:
            repo.push_to_hub(commit_message="End of training", blocking=False, auto_lfs_prune=True)


    accelerator.end_training()
    wandb.finish()


if __name__ == "__main__":
    main()
