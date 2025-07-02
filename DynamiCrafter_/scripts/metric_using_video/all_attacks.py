import sys
import os

# 현재 파일 위치 기준으로 metric_for_video 폴더 경로 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
metric_for_video_path = os.path.join(current_dir, "..", "metric_for_video")

sys.path.append(metric_for_video_path)

# 이제 metric_for_video 폴더의 모듈들을 import 가능
import utils_img
from video_metrics.frechet_video_distance import fvd
from video_metrics.tLP import tLP
from video_metrics.tOF import tOF
import torch
import torch.nn as nn 
from utils_vid_attack import (gaussian_blur, Gaussian_noise, average_filtering, median_filtering, add_salt_and_pepper_noise,add_speckle_noise
                              ,frame_drop,frame_swap,frame_freezing,frame_averaging,h264_compression)
import pandas as pd 
import torch.nn.functional as F 
from BCH import bch_error_correction_batch


import json
import os

@torch.no_grad()
def get_bit_accs(device, watermarked_video, original_video, msg_decoder: nn.Module, key: torch.Tensor, batch_size: int = 16, attacks: dict = {}):
    key = (key > 0.5).int()
    # key = key.repeat(watermarked_video.shape[0], 1)
    log_stats = {}


    
    json_data = {}  # 🔹 JSON 저장용 딕셔너리

    for name, attack in attacks.items():
        imgs_aug = attack(watermarked_video.to(device,dtype = torch.float32)) 
        imgs_aug = imgs_aug.to(device,dtype=torch.bfloat16)  # [16, 3, 256, 256]
        all_decoded = []

        try:
            for frame_idx in range(imgs_aug.shape[0]):
                frame = imgs_aug[frame_idx].unsqueeze(0)
            
                
                decoded = msg_decoder(frame) 
                decoded = decoded.to(device)
                decoded = (torch.sigmoid(decoded)>0.5).int()
                
                all_decoded.append(decoded)
        except Exception as e: 
            print(f"⚠️ {name} 공격 중 오류 발생: {e}")

        decoded = torch.cat(all_decoded, dim=0)
        

        if 'frame_drop' in name or name in ['comb_h264_framedrop_crop', 'comb_h264_framedrop_rot']:
            print(name)
            _, _, bit_accs = bch_error_correction_batch(key[:decoded.shape[0]], decoded,bch_bits=15)
        else:
            print(name)
            _, _, bit_accs = bch_error_correction_batch(key, decoded,bch_bits=15)

        log_stats[f'bit_acc_{name}'] = bit_accs

        # 🔹 JSON 데이터 저장 (key 값도 포함)
        json_data[name] = decoded.cpu().numpy().tolist()  # 텐서를 리스트로 변환

    # 🔹 Ground Truth (GT) 데이터도 JSON에 저장
    # all_decoded_gt = []

    # for frame_idx in range(original_video.shape[0]):
    #     frame = original_video[frame_idx].unsqueeze(0)
    #     try:
    #         decoded_gt = msg_decoder.detect(frame.to(device))["preds"]
    #         decoded_gt = decoded_gt.to(device)

    #         mask_preds_gt = F.sigmoid(decoded_gt[:, 0, :, :]).unsqueeze(1)
    #         bit_preds_gt = decoded_gt[:, 1:, :, :]

    #         decoded_frame_gt = msg_predict_inference(bit_preds_gt, mask_preds_gt).cpu().float()
    #         all_decoded_gt.append(decoded_frame_gt)
    #     except Exception as e:
    #         print(f"⚠️ 프레임 {frame_idx} 처리 중 오류 발생: {e}")

    # if all_decoded_gt:
    #     decoded_gt = torch.cat(all_decoded_gt, dim=0)
    #     print(f"✅ 최종 decoded_gt shape: {decoded_gt.shape}")
    #     json_data["ground_truth"] = decoded_gt.cpu().numpy().tolist()  # GT 데이터 저장
    # else:
    #     decoded_gt = None

    # # ✅ JSON 파일 저장
    # json_filename = "sora_decoded_results.json"
    # with open(json_filename, "w") as f:
    #     json.dump({str(key.tolist()): json_data}, f, indent=4)

    # print(f"✅ JSON 저장 완료: {json_filename}")
    
    return log_stats ,json_data


all_attacks = {    
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
