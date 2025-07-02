import torch
import torch.nn as nn
import torch.optim as optim
import lpips
import numpy as np
import os
import cv2
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm 
import pdb

class H264CompressorNet(nn.Module):
    def __init__(self):
        super(H264CompressorNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=3, stride=1, padding=1),  # Conv2D 대신 Conv3D 사용
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(256, 128, kernel_size=3, stride=1, padding=1, output_padding=0),  # ConvTranspose2D 대신 ConvTranspose3D 사용
            nn.ReLU(),
            nn.ConvTranspose3d(128, 64, kernel_size=3, stride=1, padding=1, output_padding=0),
            nn.ReLU(),
            nn.ConvTranspose3d(64, 3, kernel_size=3, stride=1, padding=1, output_padding=0),
            nn.Sigmoid()  # 이미지 값을 0-1로 출력
        )

    def forward(self, x):
        x = self.encoder(x)  # 5D 입력: (batch_size, C, T, H, W)
        x = self.decoder(x)
        return x

# Loss 계산을 위한 MSE 및 LPIPS
class H264Loss(nn.Module):
    def __init__(self):
        super(H264Loss, self).__init__()
        self.mse_loss = nn.MSELoss()  # MSE (Mean Squared Error)
        self.lpips_loss = lpips.LPIPS(net='vgg')  # LPIPS (Learned Perceptual Image Patch Similarity)

    def forward(self, compressed, target):
        # compressed와 target은 [batch_size, C, T, H, W] 형태입니다.
        
        # 시간 축(T)을 따라 LPIPS 손실을 계산
        lpips_value = 0
        for t in range(compressed.size(2)):  # T만큼 반복
            lpips_value += self.lpips_loss(compressed[:, :, t, :, :], target[:, :, t, :, :])
        
        # LPIPS 손실의 평균을 사용
        lpips_value = lpips_value / compressed.size(2)
        
        # MSE 손실 계산
        mse = self.mse_loss(compressed, target)
        
        return mse + lpips_value.mean()  # MSE + LPIPS 손실의 조합 반환

# VideoDataset 정의
class VideoDataset(Dataset):
    def __init__(self, input_dir, gt_dir, transform=None, resize=(256, 256)):
        self.input_dir = input_dir
        self.gt_dir = gt_dir
        self.input_videos = sorted(os.listdir(input_dir))
        self.gt_videos = sorted(os.listdir(gt_dir))
        self.transform = transform
        self.resize = resize  # 리사이즈할 해상도 (256, 256)

    def __len__(self):
        return len(self.input_videos)

    def __getitem__(self, idx):
        # 오리지널 영상 로드
        input_video_path = os.path.join(self.input_dir, self.input_videos[idx])
        input_frames = self._load_video_frames(input_video_path)

        # h264로 압축된 타겟 영상 로드
        gt_video_path = os.path.join(self.gt_dir, self.gt_videos[idx])
        gt_frames = self._load_video_frames(gt_video_path)

        if self.transform:
            input_frames = self.transform(input_frames)
            gt_frames = self.transform(gt_frames)

        return input_frames, gt_frames


    def _load_video_frames(self, video_path, max_frames=8):
        """
        Load video frames from a file, resize them, and limit to a maximum number of frames.

        Args:
            video_path (str): Path to the video file.
            max_frames (int): Maximum number of frames to load.

        Returns:
            torch.Tensor: A tensor of shape (C, T, H, W) with the loaded frames.
        """
        # OpenCV를 사용하여 비디오를 프레임 단위로 로드
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened() and len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            # BGR을 RGB로 변환 및 float 타입으로 스케일링
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            # 프레임 해상도를 256x256으로 리사이즈
            frame = cv2.resize(frame, self.resize)
            frames.append(frame)
        cap.release()

        # 프레임이 최대 프레임 수에 도달하지 않은 경우, 부족한 프레임을 0으로 채움
        while len(frames) < max_frames:
            frames.append(np.zeros_like(frames[0]))

        # 프레임을 텐서로 변환하고 차원 변경 (T, H, W, C -> B, C, T, H, W)
        frames = np.array(frames)
        frames = np.transpose(frames, (3, 0, 1, 2))
        frames = torch.tensor(frames, dtype=torch.float32)
        return frames