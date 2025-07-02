# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import os
import shutil
from tqdm import tqdm
from pathlib import Path
from PIL import Image
import kornia.augmentation as K
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import io, transforms
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn.functional as F
from einops import rearrange
import cv2
from utils_vid_attack import * 
import csv

from pytorch_fid.fid_score import InceptionV3, calculate_frechet_distance, compute_statistics_of_path
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import lpips
from video_metrics.frechet_video_distance import fvd
from video_metrics.tLP import tLP
from video_metrics.tOF import tOF

# from utils import DWT_3D
from watermark_utils import utils
import utils_img
import pdb
from datasets import video_transforms
from decoder.hvdm_decoder import HVDM_with_Resnet50,HVDM_with_Resnet50_high_freq
from decoder.utils import DWT_3D
import bchlib
import numpy as np
import random
import pickle
# from pytorch_wavelets import DWTForward, DWTInverse
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_json_as_list(data_dir, json_file="prompts_with_keys120_32bit.json"):
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Convert each key-value pair to [key, value] and store in a list
    # key_value_list = [[key, value] for key, value in data.items()]
    prompt_file = []
    message_file = []
        
    # Iterate over key-value pairs and check if the corresponding image file exists
    for key, value in data.items():
        file_path = os.path.join(data_dir, f"prompt_imgs/{key}_.png")
        # print(value)
        # Add to lists only if the file exists
        if os.path.exists(file_path):
            prompt_file.append(key)
            message_file.append(value)
    # print('='*50)
    # print(len(prompt_file))

    return prompt_file, message_file

def save_frames(original_video, watermarked_video, save_dir, num_frames=None, mult=10): # TCHW
     # TCHW
    diff_frame_path = os.path.join(save_dir, "diff") 
    os.makedirs(diff_frame_path, exist_ok=True)
    original_frame_path = os.path.join(save_dir, "original")
    os.makedirs(original_frame_path, exist_ok=True)
    watermark_frame_path = os.path.join(save_dir, "watermark") 
    os.makedirs(watermark_frame_path, exist_ok=True)

    for i in tqdm(range(original_video.shape[0])):
        diff_frames =  np.abs(np.asarray(original_video[i,:,:,:].permute(1,2,0)).astype(int) - np.asarray(watermarked_video[i,:,:,:].permute(1,2,0)).astype(int)) *10
        diff = Image.fromarray(diff_frames.astype(np.uint8))
        original_frame = Image.fromarray(np.asarray(original_video[i,:,:,:].permute(1,2,0)).astype(np.uint8))
        watermarked_frame =  Image.fromarray(np.asarray(watermarked_video[i,:,:,:].permute(1,2,0)).astype(np.uint8))
        
        diff.save(os.path.join(diff_frame_path, f"{i:03d}.png"))
        original_frame.save(os.path.join(original_frame_path ,f"{i:03d}.png"))
        watermarked_frame.save(os.path.join(watermark_frame_path, f"{i:03d}.png"))
    return original_frame_path, watermark_frame_path


def get_img_metric(watermarked_video, original_video, lpips_model, num_imgs=None):
    
    log_stats = []
    psnr, ssim, lpips_ = 0, 0, 0
    for i in tqdm(range(original_video.shape[0])):
        original_frame = np.asarray(original_video[i,:,:,:].permute(1,2,0))
        watermarked_frame =  np.asarray(watermarked_video[i,:,:,:].permute(1,2,0))
        # pdb.set_trace()

        psnr += peak_signal_noise_ratio(original_frame, watermarked_frame,data_range=1)
        ssim += structural_similarity(original_frame, watermarked_frame,data_range=1, channel_axis=2)
        lpips_ += lpips_model(lpips.im2tensor(original_frame*255), lpips.im2tensor(watermarked_frame*255)).item()

        
    log_stat = {
        'psnr': psnr / original_video.shape[0],
        'ssim': ssim / original_video.shape[0],
        'lpips': lpips_ / original_video.shape[0],
    }
    return log_stat

def get_video_metric(original_video, watermarked_video):
    fvd_val = fvd(original_video, watermarked_video)
    tLP_val = tLP(original_video*255, watermarked_video*255)
    tOF_val = tOF(original_video*255, watermarked_video*255)

    vid_metrics = {
    "fvd": fvd_val,
    "tLP": tLP_val,
    "tOF": tOF_val
    }
    return vid_metrics
    

def cached_fid(path1, path2, batch_size=32, device='cuda:0', dims=2048, num_workers=10):
    for p in [path1, path2]:
        if not os.path.exists(p):
            raise RuntimeError('Invalid path: %s' % p)
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device)
    # cache path2
    storage_path = Path.home() / f'.cache/torch/fid/{path2.replace("/", "_")}'
    if (storage_path / 'm.pt').exists():
        m2 = torch.load(storage_path / 'm.pt')
        s2 = torch.load(storage_path / 's.pt')
    else:
        storage_path.mkdir(parents=True)
        m2, s2 = compute_statistics_of_path(str(path2), model, batch_size, dims, device, num_workers)
        torch.save(m2, storage_path / 'm.pt')
        torch.save(s2, storage_path / 's.pt')
    m1, s1 = compute_statistics_of_path(str(path1), model, batch_size, dims, device, num_workers)    
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)
    return fid_value


def add_noise_to_encoded(encoded_byte, flip_probability):
    bit_string = ''.join(f'{byte:08b}' for byte in encoded_byte)
    
    noisy_bit_string = ''.join(
        '1' if bit == '0' and random.random() < flip_probability else 
        '0' if bit == '1' and random.random() < flip_probability else 
        bit
        for bit in bit_string
    )
    
    noisy_bytes = bytearray(int(noisy_bit_string[i:i+8], 2) for i in range(0, len(noisy_bit_string), 8))
    
    return bytes(noisy_bytes)

@torch.no_grad()
def get_bit_accs(args, save_frame_dir, watermarked_video, original_video,msg_decoder: nn.Module, key: torch.Tensor, batch_size: int = 16, attacks: dict = {}):
    key = (key>0.5).int()
    # print(key)
    # key = key.repeat(watermarked_video.shape[0],1).cuda() # 16 32
    
    
    
    BCH_POLYNOMIAL = 8219 
    BCH_BITS = args.bch_bits  
    
    with open(save_frame_dir[:-6]+'bch.pkl', 'rb') as f:
        loaded_data = pickle.load(f)
    BCH_message_byte = loaded_data['BCH_message_byte']
    BCH_encoded_byte = loaded_data['BCH_encoded_byte']
    BCH_keys = loaded_data['keys']
    
    log_stats = {}
    dwt = DWT_3D("haar").to('cuda')

    for name, attack in attacks.items():
        # break
        # print('original', watermarked_video.shape)
        imgs_aug = attack(watermarked_video.to(device)).to('cuda')
        f,c,h,w  =imgs_aug.shape
        pad_h, pad_w = (4 -h %4 ) %4, (4- w%4) %4 
        if pad_h > 0 or pad_w > 0:
            padding = (0, pad_w, 0, pad_h)  # (left, right, top, bottom) 순서
            imgs_aug = F.pad(imgs_aug, padding, mode='constant', value=0)
        # pdb.set_trace()
        # print('name', name, imgs_aug.shape) # f c h w  
        dwt_ = dwt(imgs_aug.permute(1,0,2,3).unsqueeze(0)) 

        decoded = msg_decoder(imgs_aug.permute(1,0,2,3).unsqueeze(0).to(torch.float), dwt_)
        decoded = (torch.sigmoid(decoded)>0.5).int()
        
        
        bit_accs = 0
        bit_accs_org = 0
        # print(name)
        if 'frame_drop' in name or 'frame_swap' in name or 'frame_freeze' in name or 'frame_average' in name:
            if 'frame_drop' in name:
                decoded = torch.cat((decoded, decoded[-1].repeat(16 - decoded.shape[0], 1)), dim=0)
            
            for i in range(16):
                bch = bchlib.BCH(BCH_BITS, BCH_POLYNOMIAL, swap_bits=False)
                decoded_frame = decoded[i].tolist()
                predicted_byte =bytes(np.packbits(decoded_frame, bitorder='little'))

                bit_accs_ = 0
                for idx in range(len(BCH_encoded_byte)):
                    
                
                    data_byte = bytearray(predicted_byte) + bytearray(BCH_encoded_byte[idx])
                    
                    corrupted_data = data_byte[:-bch.ecc_bytes]
                    corrupted_ecc = data_byte[-bch.ecc_bytes:]
                    nerr = bch.decode(corrupted_data, corrupted_ecc)
                    
                    corrected_data = bytearray(corrupted_data)
                    corrected_ecc = bytearray(corrupted_ecc)
                    
                    bch.correct(corrected_data, corrected_ecc)
                    
                    corrected_bit = np.unpackbits(np.frombuffer(corrected_data, dtype=np.uint8), bitorder='little')
                    decoded_bch = torch.tensor(corrected_bit, dtype=torch.int32)
                    tmp =  ((BCH_keys[idx] == decoded_bch).sum()/ args.nbits) / 16
                    if tmp > bit_accs_:
                        bit_accs_ = tmp
                    
                bit_accs += bit_accs_
                    
                    
                bit_accs_org += ((BCH_keys[idx] == torch.tensor(np.array(decoded_frame), dtype=torch.float32)).sum()/ args.nbits) / 16
            
            
            
        else:    
            for i in range(16):
                bch = bchlib.BCH(BCH_BITS, BCH_POLYNOMIAL, swap_bits=False)
                decoded_frame = decoded[i].tolist()
                predicted_byte =bytes(np.packbits(decoded_frame, bitorder='little'))
                idx = ((i+1)%(args.bit_capacity//32))-1
                data_byte = bytearray(predicted_byte) + bytearray(BCH_encoded_byte[idx])
                
                corrupted_data = data_byte[:-bch.ecc_bytes]
                corrupted_ecc = data_byte[-bch.ecc_bytes:]
                nerr = bch.decode(corrupted_data, corrupted_ecc)
                
                corrected_data = bytearray(corrupted_data)
                corrected_ecc = bytearray(corrupted_ecc)
                
                bch.correct(corrected_data, corrected_ecc)
                
                corrected_bit = np.unpackbits(np.frombuffer(corrected_data, dtype=np.uint8), bitorder='little')
                decoded_bch = torch.tensor(corrected_bit, dtype=torch.int32)
                bit_accs += ((BCH_keys[idx] == decoded_bch).sum()/ args.nbits) / 16
                bit_accs_org += ((BCH_keys[idx] == torch.tensor(np.array(decoded_frame), dtype=torch.float32)).sum()/ args.nbits) / 16
                


        try:
            log_stats[f'bit_acc_{name}'] = bit_accs.item()
        except:
            log_stats[f'bit_acc_{name}'] = bit_accs
    
    #############################################################
    decoded_gt = msg_decoder(
    original_video.permute(1, 0, 2, 3).unsqueeze(0).to('cuda'),  # GPU로 이동
    dwt(original_video.permute(1, 0, 2, 3).unsqueeze(0).to('cuda'))  # GPU로 이동
    )

    decoded_gt = (torch.sigmoid(decoded_gt)>0.5).int()
    bit_accs_gt = (((key.repeat(watermarked_video.shape[0],1).cuda() == decoded_gt).sum(dim=1)) / args.nbits).mean()
    
    # diff_gt = (~torch.logical_xor(decoded_gt>0, key>0)) # b k -> b k
    # pdb.set_trace()
    # bit_accs_gt = torch.sum(diff_gt, dim=-1) / diff_gt.shape[-1]
    log_stats[f'bit_acc_GT'] = bit_accs_gt.item()
    #############################################################
    return log_stats


@torch.no_grad()
def get_msgs(img_dir: str, msg_decoder: nn.Module, batch_size: int = 16, attacks: dict = {}):
    # resize crop
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    data_loader = utils.get_dataloader(img_dir, transform, batch_size=batch_size, collate_fn=None)
    # dwt = DWT_3D("haar").to(device)
    log_stats = {ii:{} for ii in range(len(data_loader.dataset))}
    for ii, imgs in enumerate(tqdm.tqdm(data_loader)):

        imgs = imgs.to(device)

        for name, attack in attacks.items():
            imgs_aug = attack(imgs.permute(1,0,2,3).unsqueeze(0))
            # print(dwt(imgs).shape)
            decoded = msg_decoder(imgs_aug)>0 # b c h w -> b k
            for jj in range(decoded.shape[0]):
                img_num = ii*batch_size+jj
                log_stat = log_stats[img_num]
                log_stat[f'decoded_{name}'] = "".join([('1' if el else '0') for el in decoded[jj].detach()])

    log_stats = [{'img': img_num, **log_stats[img_num]} for img_num in range(len(data_loader.dataset))]
    return log_stats

def read_video(video_path):
    import pdb 
    # pdb.set_trace()
    if os.path.isfile(video_path):
        pass
    else:
        video_path = video_path.replace(".avi","_.avi")
    cap = cv2.VideoCapture(video_path)
    video = []
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
            video.append(frame)
        else:
            break
    cap.release()
    # print('aaaa',video)
    video = (rearrange(torch.tensor(video, dtype=torch.float), 't h w c -> t c h w') / 255.0 )
    # print(video.shape)
    return video,video_path


def main(args):

    # Set seeds for reproductibility 
    np.random.seed(args.seed)
    
    lpips_model = lpips.LPIPS(net='vgg')
    
    # Loads hidden decoder
    import pdb 
    # pdb.set_trace()
    print(f'>>> Building hidden decoder with weights from {args.msg_decoder_path}...')
    
    resnet50_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    resnet50_model.fc = torch.nn.Linear(2048, args.nbits)
    # msg_decoder = resnet50_model.to(device, torch.float)
    msg_decoder = HVDM_with_Resnet50(resnet_model = resnet50_model, dim = 2048, num_frames = 16,
                                     image_size=256, phi_dimension = args.nbits)


    # pdb.set_trace()
    msg_decoder_weight = torch.load(args.msg_decoder_path, map_location='cpu')

    new_msg_decoder_weight = {}
    for key, value in msg_decoder_weight.items():
        new_key = key.replace('module.', '') 
        new_msg_decoder_weight[new_key] = value
        
    msg_decoder.load_state_dict(new_msg_decoder_weight)
    msg_decoder.eval()
    
    # Create the directories
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    save_frame_dir = os.path.join(args.output_dir, 'frames')
    args.save_frame_dir = save_frame_dir
    if not os.path.exists(save_frame_dir):
        os.makedirs(save_frame_dir, exist_ok=True)
        
 
    transform = transforms.Compose([
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                # transforms.Resize((256,256))
                # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])


    print(f">>> Loading Videos...")
    print(args.original_video)
    print(args.watermarked_video)
    watermarked_video, watermarked_path = read_video(args.watermarked_video)
    original_video, original_path = read_video(args.original_video)
    
    transform_original_video = transform(original_video)
    transform_watermarked_video = transform(watermarked_video)

    if args.eval_imgs:
        print(f">>> Saving {args.save_n_imgs} diff images...")
        if args.save_n_imgs > 0:
            save_frames(original_video*255, watermarked_video*255, save_frame_dir, num_frames=args.save_n_imgs)

        print(f'>>> Computing img-2-img stats...')
        img_metrics = get_img_metric(watermarked_video, original_video, lpips_model)
        log_path = os.path.join(args.output_dir, 'img_metrics.csv')
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        vid_metrics = get_video_metric(watermarked_video, original_video)
        img_metrics.update(vid_metrics)

        with open(log_path, 'w') as csv_file:  
            writer = csv.writer(csv_file)
            for key, value in img_metrics.items():
                writer.writerow([key, value])
        
        
        print(f"psnr: {img_metrics['psnr']:.4f}")
        print(f"ssim: {img_metrics['ssim']:.4f}")
        print(f"lpips: {img_metrics['lpips']:.4f}")
        print(f"fvd: {img_metrics['fvd']:.4f}")
        print(f"tLP: {img_metrics['tLP']:.4f}")
        print(f"tOF: {img_metrics['tOF']:.4f}")
                
    
    if args.eval_bits:

        msg_decoder = msg_decoder.to(device, torch.float).eval()
        
        # nbit = msg_decoder(torch.zeros(1, 3, 128, 128).to(device)).shape[-1]
        nbit = args.nbits

        if args.attack_mode == 'all':
            attacks = {
                        'none': lambda x: x,
                        'crop_03': lambda x: utils_img.center_crop(x, 0.25),
                        'crop_04': lambda x: utils_img.center_crop(x, 0.36),
                        'crop_05': lambda x: utils_img.center_crop(x, 0.49),
                        'crop_06': lambda x: utils_img.center_crop(x, 0.64),
                        'crop_08': lambda x: utils_img.center_crop(x, 0.81),
                        'resize_03': lambda x: utils_img.resize(x, 0.25),
                        'resize_04': lambda x: utils_img.resize(x, 0.36),
                        'resize_05': lambda x: utils_img.resize(x, 0.49),
                        'resize_06': lambda x: utils_img.resize(x, 0.64),
                        'resize_08': lambda x: utils_img.resize(x, 0.81),
                        'rot_30': lambda x: utils_img.rotate(x, 30),
                        'rot_40': lambda x: utils_img.rotate(x, 40),
                        'rot_50': lambda x: utils_img.rotate(x, 50),
                        'rot_60': lambda x: utils_img.rotate(x, 60),
                        'rot_70': lambda x: utils_img.rotate(x, 70),
                        'gaussian_blur_1' :lambda x :gaussian_blur(x, std=1.0),
                        'gaussian_blur_13' :lambda x :gaussian_blur(x, std=1.3),
                        'gaussian_blur_16' :lambda x :gaussian_blur(x, std=1.6),
                        'gaussian_blur_19' :lambda x :gaussian_blur(x, std=1.9),
                        'gaussian_blur_22' :lambda x :gaussian_blur(x, std=2.2),
                        'gaussian_noise_002' :lambda x :Gaussian_noise(x, std=0.02),
                        'gaussian_noise_003' :lambda x :Gaussian_noise(x, std=0.03),
                        'gaussian_noise_004' :lambda x :Gaussian_noise(x, std=0.04),
                        'gaussian_noise_005' :lambda x :Gaussian_noise(x, std=0.05),
                        'gaussian_noise_006' :lambda x :Gaussian_noise(x, std=0.06),
                        'average_filtering': lambda x :average_filtering(x, kernel_size=5),
                        'median_filtering' :lambda x: median_filtering(x, kernel_size=5),
                        'brightness_1p5': lambda x: utils_img.adjust_brightness(x, 1.5),
                        'brightness_2': lambda x: utils_img.adjust_brightness(x, 2),
                        'contrast_1p5': lambda x: utils_img.adjust_contrast(x, 1.5),
                        'contrast_2': lambda x: utils_img.adjust_contrast(x, 2),
                        'saturation_1p5': lambda x: utils_img.adjust_saturation(x, 1.5),
                        'saturation_2': lambda x: utils_img.adjust_saturation(x, 2),
                        'sharpness_1p5': lambda x: utils_img.adjust_sharpness(x, 1.5),
                        'sharpness_2': lambda x: utils_img.adjust_sharpness(x, 2),
                        'hue_01': lambda x: utils_img.adjust_hue(x, 0.1),
                        'hue_02': lambda x: utils_img.adjust_hue(x, 0.2),
                        'overlay_text': lambda x: utils_img.overlay_text(x, [76,111,114,101,109,32,73,112,115,117,109]),
                        'add_salt_and_pepper_noise_01': lambda x: add_salt_and_pepper_noise(x, salt_prob=0.1, pepper_prob=0.1),
                        'add_salt_and_pepper_noise_03': lambda x: add_salt_and_pepper_noise(x, salt_prob=0.3, pepper_prob=0.3),
                        'add_speckle_noise': lambda x: add_speckle_noise(x, mean=0, std=0.1),
                        'frame_drop_01' : lambda x: frame_drop(x, drop_ratio=0.1),
                        'frame_drop_03' : lambda x: frame_drop(x, drop_ratio=0.3),
                        'frame_drop_05' : lambda x: frame_drop(x, drop_ratio=0.5),
                        'frame_drop_07' : lambda x: frame_drop(x, drop_ratio=0.7),
                        'frame_drop_09' : lambda x: frame_drop(x, drop_ratio=0.9),
                        'frame_swap_1' : lambda x: frame_swap(x, num_swaps=1),
                        'frame_swap_3' : lambda x: frame_swap(x, num_swaps=3),
                        'frame_swap_5' : lambda x: frame_swap(x, num_swaps=5),
                        'frame_swap_7' : lambda x: frame_swap(x, num_swaps=7),
                        'frame_swap_9' : lambda x: frame_swap(x, num_swaps=9),
                        'frame_freeze_2' : lambda x: frame_freezing(x, freeze_duration=2),
                        'frame_freeze_3' : lambda x: frame_freezing(x, freeze_duration=3),
                        'frame_freeze_4' : lambda x: frame_freezing(x, freeze_duration=4),
                        'frame_freeze_5' : lambda x: frame_freezing(x, freeze_duration=5),
                        'frame_freeze_6' : lambda x: frame_freezing(x, freeze_duration=6),
                        'frame_average_2' : lambda x: frame_averaging(x, average_duration=2),
                        'frame_average_3' : lambda x: frame_averaging(x, average_duration=3),
                        'frame_average_4' : lambda x: frame_averaging(x, average_duration=4),
                        'frame_average_5' : lambda x: frame_averaging(x, average_duration=5),
                        'frame_average_6' : lambda x: frame_averaging(x, average_duration=6),
                        'jpeg_40': lambda x: utils_img.jpeg_compress(x, 40),
                        'jpeg_50': lambda x: utils_img.jpeg_compress(x, 50),
                        'jpeg_60': lambda x: utils_img.jpeg_compress(x, 60),
                        'jpeg_70': lambda x: utils_img.jpeg_compress(x, 70),
                        'jpeg_80': lambda x: utils_img.jpeg_compress(x, 80),
                        'H264_crf21' : lambda x : h264_compression(x, crf=21),
                        'H264_crf22' : lambda x : h264_compression(x, crf=22),
                        'H264_crf23' : lambda x : h264_compression(x, crf=23),
                        'H264_crf24' : lambda x : h264_compression(x, crf=24),
                        'H264_crf25' : lambda x : h264_compression(x, crf=25),
                        'comb_crop_bright_jpeg': lambda x: utils_img.jpeg_compress(utils_img.adjust_brightness(utils_img.center_crop(x, 0.25), 1.5), 80),
                        'comb_h264_frame_drop_crop': lambda x : utils_img.center_crop(frame_drop(h264_compression(x), drop_ratio=0.2), 0.25),
                        'comb_h264_frame_drop_rot': lambda x : utils_img.rotate(frame_drop(h264_compression(x), drop_ratio=0.2), 25),
                        
            }
        elif args.attack_mode == 'few':
            attacks = {
                'none': lambda x: x,
                'crop_025': lambda x: utils_img.center_crop(x, 0.25),
                'brightness_2': lambda x: utils_img.adjust_brightness(x, 2),
                'contrast_2': lambda x: utils_img.adjust_contrast(x, 2),
                'jpeg_50': lambda x: utils_img.jpeg_compress(x, 50),
                'comb': lambda x: utils_img.jpeg_compress(utils_img.adjust_brightness(utils_img.center_crop(x, 0.49), 1.5), 80),
            }
        else:
            attacks = {'none': lambda x: x}

        if args.decode_only:
            log_stats = get_msgs(args.img_dir, msg_decoder, batch_size=args.batch_size, attacks=attacks)
        else:    
            # Creating key


            prompt_file, message_file = load_json_as_list(args.prompt_dir, args.json_file)
            prompt_file = [file[:30] for file in prompt_file]
            message = '' 
            
            import pdb 

            # print(message_file)
            for idx,p in enumerate(prompt_file): 
                # pdb.set_trace()
                if p == original_path.split('/')[-1].split('.')[0]:
                    message = message_file[idx] 

            print(args.watermarked_video.split('/')[-1], message)

            key = torch.tensor([int(x) for x in message], dtype=torch.float32)
            log_stats = get_bit_accs(args, save_frame_dir, transform_watermarked_video, transform_original_video, msg_decoder, key, batch_size=args.batch_size, attacks=attacks)
        # pdb.set_trace()
        print(f'>>> Saving log stats to {args.output_dir}...')
        

        log_path = os.path.join(args.output_dir, 'log_stats.csv')
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        with open(log_path, 'w') as csv_file:  
            writer = csv.writer(csv_file)
            for key, value in log_stats.items():
                writer.writerow([key, value])


def get_parser():
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        group.add_argument(*args, **kwargs)

    group = parser.add_argument_group('Data parameters')
    aa("--original_video", type=str, default="", help="")
    aa("--num_frames", type=int, default=None)

    group = parser.add_argument_group('Eval imgs')
    aa("--eval_imgs", type=utils.bool_inst, default=True, help="")
    aa("--watermarked_video", type=str, default="/checkpoint/pfz/2023_logs/0104_aisign_sd_txt2img/_ldm_decoder_ckpt=0_config=0_ckpt=0/samples", help="")
    aa("--img_dir_fid", type=str, default=None, help="")
    aa("--save_n_imgs", type=int, default=10)
    aa("--eval_vid", type=utils.bool_inst, default=True, help="")
    aa("--normal_latent", type=utils.bool_inst, default=False, help="")

    group = parser.add_argument_group('Eval bits')
    aa("--eval_bits", type=utils.bool_inst, default=True, help="")
    aa("--eval_bits_memoriable", type=utils.bool_inst, default=True, help="")
    aa("--decode_only", type=utils.bool_inst, default=False, help="")
    aa("--key_str", type=str, default="111010110101000001010111010011010100010000100111")
    aa("--key_path", type=str, default="./key_path")
    aa("--msg_decoder_path", type=str, default= "/data/youngdong/Latte_watermark/output_epoch_1000/msg_decoder_48.pth")
    aa("--attack_mode", type=str, default= "all")
    aa("--num_bits", type=int, default=48)
    aa("--redundancy", type=int, default=1)
    aa("--decoder_depth", type=int, default=8)
    aa("--decoder_channels", type=int, default=64)
    aa("--img_size", type=int, default=512)
    aa("--nbits", type=int, default=48)
    aa("--batch_size", type=int, default=32)

    group = parser.add_argument_group('Experiments parameters')
    aa("--output_dir", type=str, default="output_epoch_1000/", help="Output directory for logs and images (Default: /output)")
    aa("--output_data_name", type=str, default="", help="")
    aa("--seed", type=int, default=0)
    aa("--debug", type=utils.bool_inst, default=False, help="Debug mode")
    aa("--key_seed", type=int, default=42)
    aa('--json_file', type=str, default="/home/jh/prompts_with_keys120_32bit.json")
    aa('--prompt_dir', type=str, default='/home/jh/Dynamic_Wouaf/Dynamic_Wouaf/Dynamic_Wouaf/DynamiCrafter/prompts/512_ours')
    parser.add_argument("--bit_capacity", type=int, default=32, help="seed for seed_everything")
    parser.add_argument("--bch_bits", type=int, default=14, help="seed for seed_everything")
    
    return parser


if __name__ == '__main__':

    # generate parser / parse parameters
    parser = get_parser()
    args = parser.parse_args()

    # run experiment
    main(args)