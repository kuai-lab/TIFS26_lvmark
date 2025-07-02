import numpy as np
from augly.image import functional as aug_functional
import torch
from torchvision import transforms
from torchvision.transforms import functional
import cv2 
from PIL import Image, ImageEnhance, ImageOps 
import random  
import io 
import subprocess
import os
from moviepy.editor import ImageSequenceClip, VideoFileClip
import random
import string

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

default_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


normalize_vqgan = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Normalize (x - 0.5) / 0.5
unnormalize_vqgan = transforms.Normalize(mean=[-1, -1, -1], std=[1/0.5, 1/0.5, 1/0.5]) # Unnormalize (x * 0.5) + 0.5
normalize_img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Normalize (x - mean) / std
unnormalize_img = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225]) # Unnormalize (x * std) + mean

"""
### Video Attack Method
1. rotates / crop / resize / perspective_projection / temporal_loss_pass_filter / frame_rate_conversion
    Reference :
        Blind Robust Video Watermarking Based on Adaptive Region
        Selection and Channel Reference 
        https://arxiv.org/pdf/2209.13206


2. blurring / color_jitter / Gaussian_noise / frame_drop / frame swap
    Reference :
        DVMark: A Deep Multiscale Framework for Video Watermarking
        https://arxiv.org/pdf/2104.12734


3. gaussian_filtering / average_filtering / median_filtering / jpeg_compression
    Reference :
        A Robust Deep Learning-Based Video Watermarking Using Mosaic Generation
        https://www.scitepress.org/PublishedPapers/2023/116917/116917.pdf
"""

def images_to_video(image_list, output_file, codec='H264',fps=30):
    """
    Convert a list of PIL images to an MP4 video file.
    
    Args:
        image_list: List of PIL.Image objects
        output_file: String, path to the output video file
        fps: Integer, frames per second of the output video (default 30)
        
    Returns:
        None
    """
    # Convert the first image to a numpy array to get the size
    frame_array = np.array(image_list[0])
    height, width, layers = frame_array.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*codec)  # You can also use 'X264' for H.264 codec
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    # Write each frame to the video file
    for img in image_list:
        frame = np.array(img)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR
        out.write(frame)

    # Release the video writer object
    out.release()

    # return output_file



def rotates(x, angle):
    """
    Rotate frames to a specified angle without cropping 
    
    Args:
        x: List of PIL Image frames
        angle: Integer, the angle to rotate the frames

    Returns:
        rotated_frames: List of rotated frames


    Reference :
        Blind Robust Video Watermarking Based on Adaptive Region
        Selection and Channel Reference 
        https://arxiv.org/pdf/2209.13206
    """
    rotated_frames = []

    for frame in x : 
        frame = np.array(frame)
        h, w = frame.shape[:2]
        center = (w // 2, h // 2) 

        matrix = cv2.getRotationMatrix2D(center, angle, 1) 
        
        rotated_frame = cv2.warpAffine(frame, matrix, (w,h)) 
        rotated_frame = Image.fromarray(rotated_frame)
        rotated_frames.append(rotated_frame)
    


    return torch.stack(rotated_frames) 


def crop(x, crop_ratio=0.2):
    """
    Crop frames by specified ratio on all sides
    
    Args:
        x: List of PIL Image frames
        crop_ratio: Float, the ratio to crop from each side

    Returns:
        cropped_frames: List of cropped frames

    Reference :
        Blind Robust Video Watermarking Based on Adaptive Region
        Selection and Channel Reference 
        https://arxiv.org/pdf/2209.13206
    """
    cropped_frames = [] 
    for frame in x : 
        frame = np.array(frame)
        h, w = frame.shape[:2]
        crop_h = int(h* crop_ratio)
        crop_w = int(h* crop_ratio) 
        cropped_frame = Image.fromarray(frame[crop_h: h-crop_h, crop_w : w - crop_w])
        cropped_frames.append(cropped_frame) 

    return torch.stack(cropped_frames) 


def resize(x, scale_factor): 
    """
    Resize frames by specified scale factor
    
    Args:
        x: List of PIL Image frames
        scale_factor: Float, the factor by which to scale the frames

    Returns:
        resized_frames: List of resized frames


    Reference :
        Blind Robust Video Watermarking Based on Adaptive Region
        Selection and Channel Reference 
        https://arxiv.org/pdf/2209.13206
    """
    resized_frames = [] 

    for frame in x:
        frame = np.array(frame)
        h, w = frame.shape[:2] 
        new_w, new_h = int(w * scale_factor), int(h *scale_factor) 
        resized_frame = Image.fromarray(cv2.resize(frame, (new_w, new_h)))
        resized_frames.append(resized_frame) 
    
    return torch.stack(resized_frames) 


def perspective_projection(x, scale):
    """
    Apply Perspective Projection to frames with different scales for each corner
    
    Args:
        x: List of PIL Image frames
        scale: Tuple or list of 4 float values representing the scale for each corner
               (scale_tl, scale_tr, scale_bl, scale_br)

    Returns:
        projected_frames: List of perspective projected frames

    Reference :
        Blind Robust Video Watermarking Based on Adaptive Region
        Selection and Channel Reference 
        https://arxiv.org/pdf/2209.13206
    """
    scale_tl, scale_tr, scale_bl, scale_br = scale 

    projected_frames = [] 
    for frame in x : 
        frame = np.array(frame) 
        h, w = frame.shape[:2] 

        pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        pts2 = np.float32([
            [w * scale_tl, h * scale_tl],  # 상단 좌측 모서리를 scale_tl 비율만큼 이동
            [w * (1 - scale_tr), h * scale_tr],  # 상단 우측 모서리를 scale_tr 비율만큼 이동
            [w * scale_bl, h * (1 - scale_bl)],  # 하단 좌측 모서리를 scale_bl 비율만큼 이동
            [w * (1 - scale_br), h * (1 - scale_br)]  # 하단 우측 모서리를 scale_br 비율만큼 이동
        ])


        matrix = cv2.getPerspectiveTransform(pts1, pts2) 
        projected_frame = cv2.warpPerspective(frame, matrix, (w, h))

        projected_frames.append(Image.fromarray(projected_frame))
    
    return torch.stack(projected_frames)


def temporal_loss_pass_filter(x, kernel_size=4):
    """
    Apply temporal low-pass filtering using consecutive frames
    
    Args:
        x: List of PIL Image frames
        kernel_size: Integer, the number of consecutive frames to average

    Returns:
        filtered_frames: List of temporally filtered frames

    Reference :
        Blind Robust Video Watermarking Based on Adaptive Region
        Selection and Channel Reference 
        https://arxiv.org/pdf/2209.13206
    """
    filtered_frames = []   
    frames = [np.array(frame) for frame in x] 
    for i in range(len(frames) - kernel_size + 1):
        avg_frame = np.mean(frames[i: i+ kernel_size], axis=0).astype(np.uint8) 
        filtered_frames.append(Image.fromarray(avg_frame)) 
    return torch.stack(filtered_frames) 


def frame_rate_conversion(frames, original_fps, target_fps):
    """
    Convert frame rate and then back to original frame rate
    
    Args:
        frames: List of PIL Image frames
        original_fps: Integer, original frames per second of the video
        target_fps: Integer, target frames per second for conversion

    Returns:
        converted_frames: List of frames with the converted frame rate

    Reference :
        Blind Robust Video Watermarking Based on Adaptive Region
        Selection and Channel Reference 
        https://arxiv.org/pdf/2209.13206
    """
    if original_fps == target_fps: 
        return frames 
    
    conversion_ratio = target_fps / original_fps 
    converted_frames = [] 
    for i in range(int(frames) * conversion_ratio) :
        converted_frames.append(frames[int(i / conversion_ratio)])

    return torch.stack(converted_frames)


def blurring(x, sigma=2.0):
    """
    Apply Gaussian blurring to frames
    
    Args:s
        frames: List of PIL Image frames
        sigma: Float, standard deviation for Gaussian kernel
    
    Returns:
        blurred_frames: List of blurred frames


    Reference :
        DVMark: A Deep Multiscale Framework for Video Watermarking
        https://arxiv.org/pdf/2104.12734
    """

    blurred_frames = []
    for frame in x:
        frame = np.array(frame)
        blurred_frame = cv2.GaussianBlur(frame, (0, 0), sigma)
        blurred_frames.append(Image.fromarray(blurred_frame))
    return torch.stack(blurred_frames) 


def color_jitter(frames, brightness=0.5, contrast=0.5, saturation=0.5):
    """
    Apply color jitter to frames
    
    Args:
        frames: List of PIL Image frames
        brightness: Float, brightness factor
        contrast: Float, contrast factor
        saturation: Float, saturation factor
    
    Reference :
        DVMark: A Deep Multiscale Framework for Video Watermarking
        https://arxiv.org/pdf/2104.12734
    """
    jittered_frames = []
    for frame in frames:
        enhancer = ImageEnhance.Brightness(frame)
        frame = enhancer.enhance(1 + (random.random() - 0.5) * brightness)
        enhancer = ImageEnhance.Contrast(frame)
        frame = enhancer.enhance(1 + (random.random() - 0.5) * contrast)
        enhancer = ImageEnhance.Color(frame)
        frame = enhancer.enhance(1 + (random.random() - 0.5) * saturation)
        jittered_frames.append(frame)
    return torch.stack(jittered_frames)


def Gaussian_noise(x, std=0.05):
    """
    Add Gaussian noise to frames
    
    Args:
        frames: List of PIL Image frames
        std: Float, standard deviation of Gaussian noise
    
    Reference :
        DVMark: A Deep Multiscale Framework for Video Watermarking
        https://arxiv.org/pdf/2104.12734
    """
    noisy_frames = []
    for frame in x:
        frame = np.array(frame.cpu())
        noise = np.random.normal(0, std, frame.shape)

        noisy_frames.append(torch.from_numpy(frame + noise))
    
    return torch.stack(noisy_frames).to(torch.float32)

def frame_average(frames):
    """
    Calculate the average of a list of PIL Image frames.
    
    Args:
        frames: List of PIL Image frames
    
    Returns:
        Tensor: Average of frames as a PyTorch tensor
    
    Reference :
        Adapted from standard averaging techniques.
    """
    # Check if frames list is empty
    if not frames:
        raise ValueError("The frames list is empty.")

    # Convert frames to NumPy arrays and accumulate their sum
    sum_array = np.zeros(np.array(frames[0]).shape, dtype=np.float32)
    for frame in frames:
        sum_array += np.array(frame, dtype=np.float32)

    # Calculate the average
    average_array = sum_array / len(frames)

    # Convert the average back to a PyTorch tensor
    average_tensor = torch.tensor(average_array)

    return average_tensor


def frame_drop(frames, drop_ratio=0.2):
    """
    Drop frames by a specified ratio
    
    Args:
        frames: List of PIL Image frames
        drop_ratio: Float, the ratio of frames to drop
    
    Reference :
        DVMark: A Deep Multiscale Framework for Video Watermarking
        https://arxiv.org/pdf/2104.12734
    """
    num_frames = len(frames)
    num_drop = int(num_frames * drop_ratio)
    if num_drop % 2 ==1 and num_drop != 0 and num_drop != num_frames: num_drop += 1
    drop_indices = sorted(random.sample(range(num_frames), num_drop))
    remaining_frames = [frame for i, frame in enumerate(frames) if i not in drop_indices]
    return torch.stack(remaining_frames)



def frame_swap(frames, num_swaps=5):
    """
    Swap pairs of frames in the video
    
    Args:
        frames: List of PIL Image frames
        num_swaps: Integer, number of frame pairs to swap
    
    Reference :
        DVMark: A Deep Multiscale Framework for Video Watermarking
        https://arxiv.org/pdf/2104.12734
    """
    num_frames = len(frames)
    swapped_frames = frames.clone()
    for _ in range(num_swaps):
        idx1, idx2 = random.sample(range(num_frames), 2)
        swapped_frames[idx1], swapped_frames[idx2] = swapped_frames[idx2], swapped_frames[idx1]
    return swapped_frames


# # Gaussian Filtering
# def gaussian_blur(frames, std):
#     """s
#     Apply Gaussian filtering to frames
    
#     Args:
#         frames: List of PIL Image frames
#         kernel_size: Integer, size of the Gaussian kernel
    
#     Returns:
#         filtered_frames: List of Gaussian filtered frames
#     """
#     filtered_frames = []
#     for frame in frames:
#         # Convert PIL Image to NumPy array
#         frame_np = np.array(frame.cpu().detach().numpy())
#         # Apply Gaussian blur
#         filtered_frame_np = cv2.GaussianBlur(frame_np, (0,0), (std,std))
#         # Convert NumPy array back to PIL Image
#         filtered_frame = torch.from_numpy(filtered_frame_np)
#         filtered_frames.append(filtered_frame)
#     return torch.stack(filtered_frames)


# Gaussian Filtering
def gaussian_blur(frames, std):
    """
    Apply Gaussian filtering to frames
    
    Args:
        frames: List of PIL Image frames
        kernel_size: Integer, size of the Gaussian kernel
    
    Returns:
        filtered_frames: List of Gaussian filtered frames
    """
        
        
    frames_np = frames.permute(0, 2, 3, 1).cpu().numpy()  # (16, 640, 640, 3)로 변환 (cv2에서 처리하기 쉬운 형식)

    # GaussianBlur 적용된 결과를 저장할 배열 생성
    blurred_np = np.empty_like(frames_np)

    # 각 배치와 채널에 대해 GaussianBlur 적용
    for i in range(frames_np.shape[0]):  # 배치 차원
        for c in range(frames_np.shape[3]):  # 채널 차원 (3채널)
            blurred_np[i, :, :, c] = cv2.GaussianBlur(frames_np[i, :, :, c], (0,0), std)

    # 다시 PyTorch 텐서로 변환
    filtered_frames = torch.from_numpy(blurred_np).permute(0, 3, 1, 2)
    
    
    return filtered_frames

# Average Filtering
def average_filtering(frames, kernel_size=5):
    """
    Apply average filtering to frames
    
    Args:
        frames: List of PIL Image frames
        kernel_size: Integer, size of the average kernel
    
    Returns:
        filtered_frames: List of average filtered frames
    """
    
    frames_np = frames.permute(0, 2, 3, 1).cpu().numpy()  # (16, 640, 640, 3)로 변환 (cv2에서 처리하기 쉬운 형식)

    # GaussianBlur 적용된 결과를 저장할 배열 생성
    blurred_np = np.empty_like(frames_np)

    # 각 배치와 채널에 대해 GaussianBlur 적용
    for i in range(frames_np.shape[0]):  # 배치 차원
        for c in range(frames_np.shape[3]):  # 채널 차원 (3채널)
            blurred_np[i, :, :, c] = cv2.blur(frames_np[i, :, :, c], (kernel_size, kernel_size))

    # 다시 PyTorch 텐서로 변환
    filtered_frames = torch.from_numpy(blurred_np).permute(0, 3, 1, 2)
    
    return filtered_frames


# Median Filtering
def median_filtering(frames, kernel_size=5):
    """
    Apply median filtering to frames
    
    Args:
        frames: List of PIL Image frames
        kernel_size: Integer, size of the median kernel
    
    Returns:
        filtered_frames: List of median filtered frames
    """
    
    
    frames_np = frames.permute(0, 2, 3, 1).cpu().numpy()  # (16, 640, 640, 3)로 변환 (cv2에서 처리하기 쉬운 형식)

    # GaussianBlur 적용된 결과를 저장할 배열 생성
    blurred_np = np.empty_like(frames_np)

    # 각 배치와 채널에 대해 GaussianBlur 적용
    for i in range(frames_np.shape[0]):  # 배치 차원
        for c in range(frames_np.shape[3]):  # 채널 차원 (3채널)
            blurred_np[i, :, :, c] = cv2.medianBlur(frames_np[i, :, :, c], kernel_size)

    # 다시 PyTorch 텐서로 변환
    filtered_frames = torch.from_numpy(blurred_np).permute(0, 3, 1, 2)
    
    
    return filtered_frames

def h264_compression(tensor, fps=16, crf=23):
    """
    Compress a tensor of shape (frames, channels, height, width) using H.264 codec with MoviePy,
    load it back as a tensor, and delete the saved video file.

    Args:
        tensor: Tensor of shape (frames, channels, height, width).
        fps: Frames per second for the output video.
        crf: Constant Rate Factor for controlling quality.

    Returns:
        loaded_tensor: Loaded tensor of shape (frames, channels, height, width).
    """

    unnormalize_img = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])
    tensor = torch.clamp(unnormalize_img(tensor), 0,1)
    # Check if tensor is on GPU and move to CPU
    if tensor.is_cuda:
        tensor = tensor.cpu()

    # Get dimensions
    frames, channels, height, width = tensor.shape

    # Convert tensor to NumPy array and rearrange dimensions for MoviePy
    video_array = tensor.permute(0, 2, 3, 1).numpy()  # (frames, height, width, channels)
    
    # np_img = (video_array * 255).astype(np.uint8)
    # images = [np_img[i] for i in range(frames)]
    
    

    # Convert to a list of images (NumPy arrays)
    images = [video_array[i] * 255 for i in range(frames)]
    images = [image.astype(np.uint8) for image in images]  # Ensure uint8 type

    # Create an ImageSequenceClip from the list of images
    clip = ImageSequenceClip(images, fps=fps)

    # Define the output file path
    output_file = 'compressed_video' + ''.join(random.choices(string.ascii_letters + string.digits, k=8)) + '.mp4'

    # Write the video file with specified codec and CRF
    clip.write_videofile(output_file, codec='libx264', ffmpeg_params=['-crf', str(crf)], audio=False)

    # Load the saved video file back as a tensor
    with VideoFileClip(output_file) as video:
        loaded_images = []
        for frame in video.iter_frames(fps=fps, dtype='uint8'):
            loaded_images.append(frame)

    # Convert loaded images back to tensor and rearrange dimensions
    loaded_tensor = torch.tensor(np.array(loaded_images)).permute(0, 3, 1, 2)  # (frames, channels, height, width)

    # Delete the saved video file
    os.remove(output_file)
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    
    return normalize_img(loaded_tensor / 255.0) 

# def compress_h264_ffmpeg(tensor, fps=16, crf=23):
#     """
#     Compress a tensor of shape (frames, channels, height, width) using H.264 codec with FFmpeg and return the compressed tensor.

#     Args:
#         tensor: Tensor of shape (frames, channels, height, width).
#         fps: Frames per second for the output video.
#         crf: Constant Rate Factor for controlling quality.

#     Returns:
#         compressed_tensor: Tensor of shape (frames, channels, height, width) after compression.
#     """
#     # Check if tensor is on GPU and move to CPU
#     if tensor.is_cuda:
#         tensor = tensor.cpu()

#     # Get dimensions
#     frames, channels, height, width = tensor.shape

#     # Convert tensor to NumPy array and rearrange dimensions for OpenCV
#     video_array = tensor.permute(0, 2, 3, 1).numpy()  # (frames, height, width, channels)

#     # Save frames to temporary images
#     temp_images_dir = 'temp_images'
#     os.makedirs(temp_images_dir, exist_ok=True)

#     for i in range(frames):
#         frame = video_array[i]
#         if frame.dtype != np.uint8:
#             frame = np.clip(frame, 0, 1) * 255
#             frame = frame.astype(np.uint8)
#         cv2.imwrite(os.path.join(temp_images_dir, f'frame_{i:04d}.png'), frame)

#     # Use FFmpeg to compress images to video
#     output_file = 'temp_video.mp4'
#     ffmpeg_command = [
#         'ffmpeg',
#         '-framerate', str(fps),
#         '-i', os.path.join(temp_images_dir, 'frame_%04d.png'),
#         '-c:v', 'libx264',
#         '-crf', str(crf),
#         '-pix_fmt', 'yuv420p',
#         output_file
#     ]
#     subprocess.run(ffmpeg_command)

#     # Read the video back to extract frames
#     cap = cv2.VideoCapture(output_file)

#     # Create a list to store the frames
#     compressed_frames = []
    
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         # Convert frame back to tensor
#         frame_tensor = torch.from_numpy(frame).permute(2, 0, 1)  # (height, width, channels) to (channels, height, width)
#         compressed_frames.append(frame_tensor)

#     # Release the VideoCapture
#     cap.release()

#     # Remove temporary images and video file
#     for i in range(frames):
#         os.remove(os.path.join(temp_images_dir, f'frame_{i:04d}.png'))
#     os.rmdir(temp_images_dir)
#     os.remove(output_file)

#     # Stack the frames to create a tensor
#     compressed_tensor = torch.stack(compressed_frames)

#     return compressed_tensor


# def Video_H264Compression(x, output_file="output_video.mp4", codec='H264', crf=23, fps=30):
#     """
#     Convert a list of PIL images to an H.264 encoded MP4 video file.
    
#     Args:
#         image_list: List of PIL.Image objects
#         output_file: String, path to the output video file (default 'output_video.mp4')
#         fps: Integer, frames per second of the output video (default 30)
#         crf: Integer, constant rate factor for H.264 compression (default 25)
        
#     Returns:
#         output_file: String, path to the generated video file
#     """
#     temp_video_path = 'temp_' + output_file
#     images_to_video(x, temp_video_path, fps=30, codec='mp4v') 

#     import subprocess 
    
#     compressed_output_file = 'compressed_' + output_file
#     try:
#         result = subprocess.run([
#             'ffmpeg', '-y', '-i', temp_video_path,
#             '-vcodec', 'libopenh264', '-crf', str(crf),
#             '-preset', 'slow', '-pix_fmt', 'yuv420p',
#             compressed_output_file
#         ], check=True, capture_output=True, text=True)
#         print(result.stdout)
#         print(result.stderr)
#     except subprocess.CalledProcessError as e:
#         print("Error during ffmpeg execution:")
#         print(e.stdout)
#         print(e.stderr)
#         raise

#     return compressed_output_file

##


def random_hue(x, p=1.0):
    """
    Apply random hue adjustment to frames

    Args:
        x: List of PIL Image frames
        p: Float, probability of applying the adjustment

    Returns:
        adjusted_frames: List of hue adjusted frames
    """
    adjusted_frames = []
    
    color_jitter = transforms.ColorJitter(brightness=0, contrast=0, saturation=(0.5, 1.5), hue=0)
    
    for frame in x:
        if random.random() < p:
            adjusted_frame = color_jitter(frame)  # Apply random saturation adjustment
            adjusted_frames.append(adjusted_frame)
        else:
            adjusted_frames.append(frame)
    return torch.stack(adjusted_frames)


def histogram_equalization(x):
    """
    Apply histogram equalization to frames

    Args:
        x: List of PIL Image frames

    Returns:
        equalized_frames: List of histogram equalized frames
    """
    equalized_frames = []
    for frame in x:
        img_np = np.array(frame.convert("L"))  # Convert to grayscale for histogram equalization
        img_eq = cv2.equalizeHist(img_np)
        img_eq = cv2.cvtColor(img_eq, cv2.COLOR_GRAY2RGB)  # Convert back to RGB
        equalized_frames.append(Image.fromarray(img_eq))
    return torch.stack(equalized_frames)


def add_salt_and_pepper_noise(x, salt_prob=0.1, pepper_prob=0.1):
    """
    Add salt & pepper noise to frames

    Args:
        x: List of PIL Image frames
        salt_prob: Float, probability of salt noise
        pepper_prob: Float, probability of pepper noise

    Returns:
        noisy_frames: List of frames with added salt & pepper noise
    """
    noisy_frames = []
    
    for frame in x:
        salt = torch.rand_like(frame[0]) < salt_prob  # Random noise mask for salt
        pepper = torch.rand_like(frame[0]) < pepper_prob  # Random noise mask for pepper

        # Apply noise (255 for salt, 0 for pepper) - scaling frame to [0, 255] for image
        noisy_frame = frame.clone()  # Clone to avoid modifying original frame
        noisy_frame[:, salt] = 1  # Salt (255 in [0, 1] scaled space is 1)
        noisy_frame[:, pepper] = 0  # Pepper (0)

        noisy_frames.append(noisy_frame)
    return torch.stack(noisy_frames)


def add_speckle_noise(x, mean=0, std=0.1):
    """
    Add speckle noise to frames

    Args:
        x: List of PIL Image frames
        mean: Float, mean of the speckle noise
        std: Float, standard deviation of the speckle noise

    Returns:
        noisy_frames: List of frames with added speckle noise
    """
    noisy_frames = []
    for frame in x:
        noise = torch.randn_like(frame) * std + mean  # Gaussian noise
        noisy_frame = frame + frame * noise  # Speckle noise = original + original * noise
        
        # Clip the values to be in [0, 1]
        noisy_frame = torch.clamp(noisy_frame, 0, 1)
        
        noisy_frames.append(noisy_frame)
    return torch.stack(noisy_frames)


def color_quantization(x, num_colors=64):
    """
    Apply color quantization to frames

    Args:
        x: List of PIL Image frames
        num_colors: Integer, number of colors to quantize to

    Returns:
        quantized_frames: List of color quantized frames
    """
    quantized_frames = []

    def kmeans_quantize(img_tensor, k):
        pixels = img_tensor.view(-1, 3)
        pixels = pixels.to(device)

        # Initialize cluster centers
        indices = torch.randperm(pixels.size(0))[:k]
        centers = pixels[indices]

        for _ in range(10):  # Number of iterations
            # Assign pixels to nearest cluster center
            distances = torch.cdist(pixels, centers)
            labels = torch.argmin(distances, dim=1)

            # Update cluster centers
            new_centers = torch.stack([pixels[labels == i].mean(0) for i in range(k)])
            valid_clusters = ~torch.isnan(new_centers[:, 0])
            centers[valid_clusters] = new_centers[valid_clusters]

        # Replace each pixel with its cluster center
        quantized_pixels = centers[labels].view(img_tensor.size())
        return torch.stack(quantized_pixels)

    for frame in x:
        quantized_tensor = kmeans_quantize(frame, num_colors)
        quantized_img = transforms.ToPILImage()(quantized_tensor.permute(2, 0, 1).cpu())
        quantized_frames.append(quantized_img)

    return torch.stack(quantized_frames)

def floyd_steinberg_dithering(x, num_colors=64):
    """
    Apply Floyd-Steinberg dithering to frames

    Args:
        x: List of PIL Image frames
        num_colors: Integer, number of colors to quantize to

    Returns:
        dithered_frames: List of dithered frames
    """
    dithered_frames = []

    def floyd_steinberg(img_array, num_colors):
        h, w, _ = img_array.shape
        for y in range(h):
            for x in range(w):
                old_pixel = img_array[y, x].copy()
                new_pixel = np.round(old_pixel * (num_colors / 255.0)) * (255.0 / num_colors)
                img_array[y, x] = new_pixel
                quant_error = old_pixel - new_pixel
                if x + 1 < w:
                    img_array[y, x + 1] += quant_error * 7 / 16
                if y + 1 < h:
                    if x > 0:
                        img_array[y + 1, x - 1] += quant_error * 3 / 16
                    img_array[y + 1, x] += quant_error * 5 / 16
                    if x + 1 < w:
                        img_array[y + 1, x + 1] += quant_error * 1 / 16
        return img_array

    for frame in x:
        img = np.array(frame).astype(np.float32)
        dithered_img_array = floyd_steinberg(img, num_colors)
        dithered_img = Image.fromarray(dithered_img_array.astype(np.uint8))
        dithered_frames.append(dithered_img)

    return torch.stack(dithered_frames)


def frame_freezing(x, freeze_duration=5):
    """
    Apply frame freezing to frames

    Args:
        x: List of PIL Image frames
        freeze_duration: Integer, number of frames to freeze

    Returns:
        frozen_frames: List of frames with applied frame freezing
    """
    # import pdb
    # pdb.set_trace()
    num_frames = len(x)
    freeze_frame = random.randint(0, num_frames - freeze_duration)
    frozen_frames = torch.stack([x[freeze_frame]]).repeat(freeze_duration, 1,1,1)
    return torch.cat([x[:freeze_frame], frozen_frames, x[freeze_frame+freeze_duration:]])

def frame_averaging(x, average_duration=5):

    # import pdb
    # pdb.set_trace()
    num_frames = len(x)
    average_frame_number = random.randint(0, num_frames - average_duration)
    average_frame = 0
    for i in range(average_frame_number, average_frame_number + average_duration):
        average_frame += x[i,:,:,:]
    average_frame /= average_duration
    average_frames = average_frame.unsqueeze(0).repeat(average_duration, 1,1,1)
    return torch.cat([x[:average_frame_number], average_frames, x[average_frame_number+average_duration:]])


def frame_skipping(x, skip_interval=5):
    """
    Apply frame skipping to frames

    Args:
        x: List of PIL Image frames
        skip_interval: Integer, interval at which frames are skipped

    Returns:
        skipped_frames: List of frames with applied frame skipping
    """
    return torch.stack(x[::skip_interval])


def box_blur(x, kernel_size):
    """
    Apply box blur to frames

    Args:
        x: List of PIL Image frames
        kernel_size: Integer, size of the box kernel

    Returns:
        blurred_frames: List of box blurred frames
    """
    blurred_frames = []
    for frame in x:
        frame_np = np.array(frame)
        blurred_frame_np = cv2.blur(frame_np, (kernel_size, kernel_size))
        blurred_frame = Image.fromarray(blurred_frame_np)
        blurred_frames.append(blurred_frame)
    return torch.stack(blurred_frames)


def motion_blur(x, kernel_size):
    """
    Apply motion blur to frames

    Args:
        x: List of PIL Image frames
        kernel_size: Integer, size of the motion blur kernel

    Returns:
        blurred_frames: List of motion blurred frames
    """
    blurred_frames = []
    for frame in x:
        frame_np = np.array(frame)
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[kernel_size // 2, :] = np.ones(kernel_size)
        kernel = kernel / kernel_size
        blurred_frame_np = cv2.filter2D(frame_np, -1, kernel)
        blurred_frame = Image.fromarray(blurred_frame_np)
        blurred_frames.append(blurred_frame)
    return torch.stack(blurred_frames)


def random_pixel_corruption(x, corruption_rate=0.05):
    """
    Randomly corrupt pixels in frames

    Args:
        x: List of PIL Image frames
        corruption_rate: Float, proportion of pixels to corrupt

    Returns:
        corrupted_frames: List of frames with random pixel corruption
    """
    corrupted_frames = []
    for frame in x:
        img_np = np.array(frame)
        num_corrupt = int(corruption_rate * img_np.size)
        coords = [np.random.randint(0, i, num_corrupt) for i in img_np.shape]
        img_np[tuple(coords)] = np.random.randint(0, 256, num_corrupt)
        corrupted_frames.append(Image.fromarray(img_np))
    return torch.stack(corrupted_frames)


def bicubic_interpolation(x, scale_factor):
    """
    Apply bicubic interpolation to resize frames

    Args:
        x: List of PIL Image frames
        scale_factor: Float, scale factor for resizing

    Returns:
        resized_frames: List of resized frames
    """
    resized_frames = []
    for frame in x:
        resized_frame = frame.resize((int(frame.width * scale_factor), int(frame.height * scale_factor)), Image.BICUBIC)
        resized_frames.append(resized_frame)
    return torch.stack(resized_frames)