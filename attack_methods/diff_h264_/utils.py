import os
import cv2
import torch
import numpy as np

def save_model_weights(model, path):
    torch.save(model.state_dict(), path)
    
def load_video(path):
    cap = cv2.VideoCapture(path)
    video = []
    for i in range(8):  # 8프레임만 처리 (필요에 따라 수정 가능)
        ret, frame = cap.read()
        if not ret:
            break
        
        # 프레임을 128x128로 리사이즈
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        frame_resized = cv2.resize(frame, (256, 256))
        
        # (H, W, C) -> (C, H, W)로 변환 후, 정규화 (0~1 범위)
        video.append(torch.from_numpy(frame_resized.transpose(2, 0, 1).astype('float32') / 255))

    return torch.stack(video, dim=1).unsqueeze(0)  # (1, C, F, H, W) 형태로 반환

def load_video_frames(video_path, max_frames=8, resize=256):
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
            frame = cv2.resize(frame, (resize, resize))
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

def make_batch(video_paths, batch_size=4):
    batched_videos = []
    for i in range(0, len(video_paths), batch_size):
        batched_path = video_paths[i:i+batch_size]

        if len(batched_path) < batch_size:
            break  # 부족한 배치를 건너뛰고 다음 루프로 진행
        
        videos = []
        for path in batched_path:
            video = load_video(path)
            videos.append(video)
        batched_video = torch.stack(videos)
        batched_videos.append(batched_video)

    return batched_videos

# 폴더 내 모든 비디오 파일 경로를 로드하는 함수
def load_video_paths(folder_path):
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv')  # 필요한 확장자 추가
    video_paths = []

    # 폴더 내 모든 파일 확인
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(video_extensions):
                video_paths.append(os.path.join(root, file))

    return video_paths

def save_video(frames, path, fps=30):
    """
    Save a sequence of frames to a video file.

    Args:
        frames (torch.Tensor): Tensor of shape (C, T, H, W).
        path (str): Path to save the video file.
        fps (int): Frames per second of the output video.
    """
    C, T, H, W = frames.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
    out = cv2.VideoWriter(path, fourcc, fps, (W, H))

    for t in range(T):
        frame = frames[:, t, :, :].permute(1, 2, 0).detach().numpy()  # Convert to HWC and detach
        frame = (frame * 255).astype(np.uint8)  # Convert to 8-bit image

        # Convert RGB to BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        out.write(frame)

    out.release()