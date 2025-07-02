import numpy as np
import torch
from video_metrics.util_fvd import open_url
from typing import Tuple
import scipy
from PIL import Image
import os
import pdb


def compute_fvd(feats_fake: np.ndarray, feats_real: np.ndarray) -> float:
    # pdb.set_trace()
    mu_gen, sigma_gen = compute_stats(feats_fake)
    mu_real, sigma_real = compute_stats(feats_real)

    m = np.square(mu_gen - mu_real).sum()
    # pdb.set_trace()
    # 1개 video
    s = np.sqrt(np.dot(sigma_gen, sigma_real)) # pylint: disable=no-member
    fid = np.real(m + sigma_gen + sigma_real - s * 2)
    
    
    # 여러 video
    # s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen, sigma_real), disp=False) # pylint: disable=no-member
    # fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))
    # pdb.set_trace()
    return float(fid)


def compute_stats(feats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = feats.mean(axis=0) # [d]
    sigma = np.cov(feats, rowvar=False) # [d, d]

    return mu, sigma

@torch.no_grad()
def compute_our_fvd(videos_fake: np.ndarray, videos_real: np.ndarray, device: str='cuda') -> float:
    detector_url = 'https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1'
    detector_kwargs = dict(rescale=False, resize=False, return_features=True) # Return raw features before the softmax layer.

    with open_url(detector_url, verbose=False) as f:
        detector = torch.jit.load(f).eval().to(device)

    videos_fake = torch.from_numpy(videos_fake).permute(0, 4, 1, 2, 3).to(device)
    videos_real = torch.from_numpy(videos_real).permute(0, 4, 1, 2, 3).to(device)
    feats_fake = detector(videos_fake, **detector_kwargs).cpu().numpy()
    feats_real = detector(videos_real, **detector_kwargs).cpu().numpy()

    return compute_fvd(feats_fake, feats_real)


def load_images_to_array(folder_path):
    image_files = sorted([os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(('png', 'jpg', 'jpeg'))])
    
    frames = len(image_files)
    images = []

    for file in image_files:
        img = Image.open(file).resize((224, 224))
        img_array = np.array(img, dtype=np.float32) / 255.0

        if img_array.shape[-1] == 3:
            images.append(img_array)
        else:
            raise ValueError(f"Image at {file} does not have 3 channels")

    images_np = np.array(images)
    images_np = np.expand_dims(images_np, axis=0)  # (frame 수, 224, 224, 3) -> (1, frame 수, 224, 224, 3)

    return images_np


def load_images_from_folders(parent_folder_path):
    video_folders = [os.path.join(parent_folder_path, folder) for folder in os.listdir(parent_folder_path) if os.path.isdir(os.path.join(parent_folder_path, folder))]
    
    all_videos = []
    max_frames = 0
    
    # Find the maximum number of frames
    for video_folder in video_folders:
        image_files = sorted([os.path.join(video_folder, file) for file in os.listdir(video_folder) if file.endswith(('png', 'jpg', 'jpeg'))])
        if len(image_files) > max_frames:
            max_frames = len(image_files)
    
    for video_folder in video_folders:
        image_files = sorted([os.path.join(video_folder, file) for file in os.listdir(video_folder) if file.endswith(('png', 'jpg', 'jpeg'))])
        
        images = []
        for file in image_files:
            img = Image.open(file).resize((224, 224))
            img_array = np.array(img, dtype=np.float32) / 255.0

            if img_array.shape[-1] == 3:
                images.append(img_array)
            else:
                raise ValueError(f"Image at {file} does not have 3 channels")
        
        # Pad the sequence to have the same number of frames
        while len(images) < max_frames:
            images.append(np.zeros((224, 224, 3), dtype=np.float32))
        
        images_np = np.array(images)
        all_videos.append(images_np)
    
    all_videos_np = np.array(all_videos)  # (video 수, frame 수, 224, 224, 3)

    return all_videos_np

def fvd(img_dir, img_dir_nw):
    gt = load_images_to_array(img_dir_nw)
    watermarked = load_images_to_array(img_dir)
    fvd = compute_our_fvd(watermarked, gt, 'cuda')
    return fvd

if __name__ == '__main__':

    ## 1개 video
    gt_path = '/ssd/CVPR2025/diffusion/Latte/sample_videos_nw/t2v-a_cat_wearing_sunglasses_and_working_as_a_lifeguard_at_pool'
    watermarked_path = '/ssd/CVPR2025/diffusion/Latte/sample_videos_w/t2v-a_cat_wearing_sunglasses_and_working_as_a_lifeguard_at_pool'
    gt = load_images_to_array(gt_path)
    watermarked = load_images_to_array(watermarked_path)

    ## 여러 video
    # gt_path = './data/original_video/crafter_output_1'
    # watermarked_path = './data/watermark_video/crafter_output_1'
    # gt = load_images_from_folders(gt_path)
    # watermarked = load_images_from_folders(watermarked_path)

    fvd = compute_our_fvd(watermarked, gt, 'cuda')
    print(fvd)


# seed_fake = 1
# seed_real = 2
# num_videos = 1
# video_len = 16

# videos_fake = np.random.RandomState(seed_fake).rand(num_videos, video_len, 224, 224, 3).astype(np.float32)
# videos_real = np.random.RandomState(seed_real).rand(num_videos, video_len, 224, 224, 3).astype(np.float32)
# print(compute_our_fvd(videos_fake, videos_real, 'cuda'))