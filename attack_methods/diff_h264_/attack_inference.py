import torch
import torch.nn as nn
import cv2
import numpy as np
import os
from diff_h264 import *
from utils import *
from attack import ImageAugmentor

# Set up the device
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

input_dir = 'input_video'
output_dir = 'attacked_video'
gt_dir = "gt_video"
attacked_out = "attacked_output"

video_paths = load_video_paths(input_dir)

batch_size = 1
max_frames=4
resize=256

augmentor = ImageAugmentor()

with torch.no_grad():
    for i in range(0, len(video_paths), batch_size):
        batched_path = video_paths[i:i+batch_size]

        if len(batched_path) < batch_size:
            break  # 부족한 배치를 건너뛰고 다음 루프로 진행
        
        videos = []
        for path in batched_path:
            video = load_video_frames(path, max_frames, resize)
            video = video.to(device)
            videos.append(video)

        batched_video = torch.stack(videos)
        bchw = augmentor.frame_to_batch(batched_video)
        augmented_tensor = augmentor.apply_random_augmentation(bchw)
        #augmented_tensor = augmentor.apply_all_augmentations(bchw)
        augmented_tensor = augmentor.batch_to_frame(augmented_tensor)
        

        for j in range(batch_size):
            attacked_dir = f"{attacked_out}/{output_dir}/sample_{i}_{j}.mp4"
            unattacked_dir = f"{attacked_out}/{gt_dir}/sample_{i}_{j}.mp4"
            save_video(augmented_tensor[j].cpu(), attacked_dir, fps=30)
            save_video(batched_video[j].cpu(), unattacked_dir, fps=30)