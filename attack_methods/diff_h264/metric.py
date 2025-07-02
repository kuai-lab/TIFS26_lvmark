import argparse
import json
import os
import shutil
import tqdm
from pathlib import Path
from PIL import Image

import numpy as np

from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import lpips
from video_metrics.frechet_video_distance import fvd
from video_metrics.tLP import tLP
from video_metrics.tOF import tOF

def get_img_metric(img_dir, img_dir_nw, num_imgs=None):
    filenames = os.listdir(img_dir)
    filenames.sort()
    if num_imgs is not None:
        filenames = filenames[:num_imgs]
    log_stats = []
    lpips_model = lpips.LPIPS(net='vgg')
    for ii, filename in enumerate(tqdm.tqdm(filenames)):
        pil_img_ori = Image.open(os.path.join(img_dir_nw, filename))
        pil_img = Image.open(os.path.join(img_dir, filename))
        img_ori = np.asarray(pil_img_ori)
        img = np.asarray(pil_img)
        log_stat = {
            'filename': filename,
            'ssim': structural_similarity(img_ori, img, channel_axis=2),
            'psnr': peak_signal_noise_ratio(img_ori, img),
            'lpips': lpips_model(lpips.im2tensor(img_ori), lpips.im2tensor(img)).item(),
            'linf': np.amax(np.abs(img_ori.astype(int)-img.astype(int)))
        }
        log_stats.append(log_stat)
    return log_stats

def get_video_metric(img_dir, img_dir_nw):
    fvd_val = fvd(img_dir, img_dir_nw)
    tLP_val = tLP(img_dir, img_dir_nw)
    tOF_val = tOF(img_dir, img_dir_nw)
    
    vid_metrics = {
    "fvd": fvd_val,
    "tLP": tLP_val,
    "tOF": tOF_val
}
    return vid_metrics