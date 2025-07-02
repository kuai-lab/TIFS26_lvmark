from moviepy.editor import VideoFileClip
from utils import *

def mp4_to_gif(mp4_path, gif_path):
    # 동영상 파일 열기
    clip = VideoFileClip(mp4_path)

    # GIF로 저장 (전체 구간 변환)
    clip.write_gif(gif_path, fps=clip.fps)  # fps는 프레임 수를 조정할 수 있습니다


gt_paths = load_video_paths("uncompressed_video")
ffmpeg_paths = load_video_paths("ffmpeg_video")
h264_paths = load_video_paths("h264_video")



for idx, (path1, path2, path3) in enumerate(zip(gt_paths, ffmpeg_paths, h264_paths)):
    gt_gif_path = f"uncompressed_gif/{idx}.gif"
    ffmpeg_gif_path = f"ffmpeg_gif/{idx}.gif"
    h264_gif_path = f"h264_gif/{idx}.gif"


    mp4_to_gif(path1, gt_gif_path)
    mp4_to_gif(path2, ffmpeg_gif_path)
    mp4_to_gif(path3, h264_gif_path)
