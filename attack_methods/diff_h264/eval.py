import os
import cv2
import numpy as np
from tqdm import tqdm
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
import lpips
from PIL import Image

def get_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR (OpenCV format) to RGB (PIL format)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    cap.release()
    return frames

def get_video_metric(video_dir, video_dir_nw, num_videos=None, num_frames=None):
    filenames = os.listdir(video_dir)
    filenames_nw = os.listdir(video_dir_nw)

    filenames.sort()
    filenames_nw.sort()

    if num_videos is not None:
        filenames = filenames[:num_videos]
        filenames_nw = filenames_nw[:num_videos]

    # Ensure both directories have the same number of videos
    assert len(filenames) == len(filenames_nw), "Mismatch in the number of videos between the two directories."

    log_stats = []
    lpips_model = lpips.LPIPS(net='vgg')

    for ii, (filename, filename_nw) in enumerate(zip(tqdm(filenames), filenames_nw)):
        video_path = os.path.join(video_dir, filename)
        video_path_nw = os.path.join(video_dir_nw, filename_nw)

        # Print video paths for debugging
        print(f"Processing: {video_path} vs {video_path_nw}")

        frames = get_video_frames(video_path)
        frames_nw = get_video_frames(video_path_nw)

        # If num_frames is specified, use only the first num_frames frames
        if num_frames is not None:
            frames = frames[:num_frames]
            frames_nw = frames_nw[:num_frames]

        for frame_idx, (frame, frame_nw) in enumerate(zip(frames, frames_nw)):
            log_stat = {
                'filename': f"{filename}_{filename_nw}_frame_{frame_idx}",
                'ssim': structural_similarity(frame_nw, frame, channel_axis=2),
                'psnr': peak_signal_noise_ratio(frame_nw, frame),
                'lpips': lpips_model(lpips.im2tensor(frame_nw), lpips.im2tensor(frame)).item(),
                'linf': np.amax(np.abs(frame_nw.astype(int) - frame.astype(int)))
            }

            print(log_stat)
            log_stats.append(log_stat)

    return log_stats

#원본
gt_video_dir = "compressed_output/uncompressed_video"
#모방 h264
h264_video_dir = "compressed_output/compressed_video"
#ffmpeg h264
ffmpeg_video_dir = "ffmpeg_video"

#get_video_metric(gt_video_dir, ffmpeg_video_dir)
#get_video_metric(gt_video_dir, h264_video_dir)
get_video_metric(ffmpeg_video_dir, h264_video_dir)