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

# Instantiate the model and load the trained weights
model = H264CompressorNet().to(device)
model.load_state_dict(torch.load('checkpoints/model_weights_epoch_20.pth', map_location=device))
model.eval()  # Set the model to evaluation mode

input_dir = 'input_video'
output_dir = 'compressed_video'
gt_dir = "uncompressed_video"
compressed_out = 'compressed_output'

video_paths = load_video_paths(input_dir)

batch_size = 1
max_frames=8
resize=256

#end to end 사이에 넣을려면 gradient flow 허용해야됨
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
        compressed_videos = model(batched_video)
        
        for j in range(batch_size):
            compressed_dir = f"{compressed_out}/{output_dir}/sample_{i}_{j}.mp4"
            uncompressed_dir = f"{compressed_out}/{gt_dir}/sample_{i}_{j}.mp4"
            save_video(compressed_videos[j].cpu(), compressed_dir, fps=30)
            save_video(batched_video[j].cpu(), uncompressed_dir, fps=30)