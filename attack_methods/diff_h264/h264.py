import torch
import torch.nn as nn
import torch.optim as optim
import lpips
import numpy as np
import os
import cv2
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm 
from diff_h264 import *
from utils import *


# 학습 과정
def train(model, dataloader, optimizer, criterion, device, epochs=10):
    model.train()
    num_batches = len(dataloader)  # 전체 배치 수를 계산

    for epoch in range(epochs):
        running_loss = 0.0
        #이미지를 압축하면 -> 압축된 이미지
      
        # tqdm을 사용하여 진행 상황을 표시
        with tqdm(dataloader, unit="batch") as tepoch:
            for batch_idx, (inputs, targets) in enumerate(tepoch):
                tepoch.set_description(f"Epoch {epoch+1}/{epochs}")

                inputs = inputs.to(device)  # 입력 데이터를 GPU로 이동
                targets = targets.to(device)  # 타겟 데이터를 GPU로 이동

                optimizer.zero_grad()  # 이전의 기울기 초기화
                outputs = model(inputs)
                total_loss = criterion(outputs, targets)  # Loss 계산
                total_loss.backward()  # 역전파 실행
                optimizer.step()  #가중치 갱신

                running_loss += total_loss.item()  # 배치별 손실을 누적

                if batch_idx == num_batches - 1:
                    # 마지막 배치에서만 비디오 저장
                    save_video(inputs[0].cpu(), f'results/input_video_batch_{epoch}.mp4')
                    save_video(targets[0].cpu(), f'results/target_video_batch_{epoch}.mp4')
                    save_video(outputs[0].cpu(), f'results/output_video_batch_{epoch}.mp4')

                # tqdm 진행 막대에 현재 배치의 손실 값 표시
                tepoch.set_postfix(total_loss=total_loss.item())

        # 에포크가 끝날 때마다 평균 손실 출력
        print(f"Epoch [{epoch+1}/{epochs}], Average Loss: {running_loss / len(dataloader):.4f}")

        # 매 에포크마다 모델 웨이트 저장
        save_model_weights(model, f'checkpoints/model_weights_epoch_{epoch+1}.pth')

    print("Training Finished!")
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = H264CompressorNet().to(device)
criterion = H264Loss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 데이터셋 경로 설정
input_dir = 'dataset/webvid/train'  # 오리지널 영상 디렉토리
gt_dir = 'dataset/webvid/target'  # 전통적인 h264로 압축된 영상 디렉토리
batch_size = 8 # 예시 배치 사이즈

# 데이터셋 및 데이터로더 정의
video_dataset = VideoDataset(input_dir=input_dir, gt_dir=gt_dir)
train_dataloader = DataLoader(video_dataset, batch_size=batch_size, shuffle=True)

# 학습 시작
epochs = 20  # 학습할 에포크 수
train(model, train_dataloader, optimizer, criterion, device, epochs)
