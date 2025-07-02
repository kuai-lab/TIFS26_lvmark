import os
import cv2
import torch
import ffmpeg
import subprocess
from utils import *
from tqdm import tqdm

def encode_video_to_h264(input_file, output_file):
    (
        ffmpeg
        .input(input_file)
        .filter('scale', 256, 256)  # 리사이징 필터 적용
        .output(output_file, vcodec='libx264', crf=23, preset='medium')
        .run()
    )

# h264학습용 데이터셋 만들기 위한 dir
input_path = "/ssd/dataset/panda70M/valid" 
output_path = "/ssd/dataset/panda70M/valid_h264"

# metric 비교를 위한 dir
"""input_path = "uncompressed_video" 
output_path = "ffmpeg_video"""

video_paths= load_video_paths(input_path)

for video_path in tqdm(video_paths, desc="Processing videos"):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # 출력 파일 경로 설정 (H.264로 인코딩된 파일)
    output_video = os.path.join(output_path, f"{video_name}.mp4")
    
    # H.264로 인코딩
    encode_video_to_h264(video_path, output_video)