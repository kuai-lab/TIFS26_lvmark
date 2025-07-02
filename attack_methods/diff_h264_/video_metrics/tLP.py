import numpy as np
import cv2
import os, sys
import pandas as pd
from video_metrics.LPIPSmodels import util
import video_metrics.LPIPSmodels.dist_model as dm
import pdb
from torchvision import transforms
from PIL import Image
import torch

def im2tensor(image, imtype=np.uint8, cent=1., factor=255./2.):
# def im2tensor(image, imtype=np.uint8, cent=1., factor=1.):
    return torch.Tensor((image / factor - cent)
                        [:, :, :, np.newaxis].transpose((3, 2, 0, 1)))

def crop_8x8( img ):
    
    ori_h = img.shape[0]
    ori_w = img.shape[1]
    
    h = (ori_h//32) * 32
    w = (ori_w//32) * 32
    
    while(h > ori_h - 16):
        h = h - 32
    while(w > ori_w - 16):
        w = w - 32
    
    y = (ori_h - h) // 2
    x = (ori_w - w) // 2
    crop_img = img[y:y+h, x:x+w]
    return crop_img, y, x


def load_images_to_array(folder_path): # 0~255
    image_arrays = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            file_path = os.path.join(folder_path, filename)
            img = Image.open(file_path).convert('RGB')
            image_arrays.append(np.array(img))
    
    images_np = np.stack(image_arrays, axis=0)  # (N, H, W, C)
    
    return images_np
def tLP(img_dir, img_dir_nw):
    model = dm.DistModel()
    model.initialize(model='net-lin',net='alex',use_gpu=True)
    gt = load_images_to_array(img_dir_nw)
    watermarked = load_images_to_array(img_dir)
    i = 0
    tLPs = []
    for gt_, watermarked_ in zip(gt, watermarked):
        watermarked_, ofy, ofx = crop_8x8(watermarked_)
        gt_, ofy, ofx = crop_8x8(gt_)
        
        gt_ = im2tensor(gt_)
        watermarked_ = im2tensor(watermarked_)
        if i != 0:
            dist0t = model.forward(gt_pre, gt_)
            dist1t = model.forward(watermarked_pre, watermarked_)
            dist01t = np.absolute(dist0t - dist1t) * 100.0
            tLPs.append( dist01t[0] )
        gt_pre = gt_
        watermarked_pre = watermarked_
        i += 1
    return np.mean(np.array(tLPs))


if __name__ == '__main__':

    model = dm.DistModel()
    model.initialize(model='net-lin',net='alex',use_gpu=True)

    gt_path = '/ssd/CVPR2025/diffusion/Latte/sample_videos_nw/t2v-a_cat_wearing_sunglasses_and_working_as_a_lifeguard_at_pool'
    watermarked_path = '/ssd/CVPR2025/diffusion/Latte/sample_videos_w/t2v-a_cat_wearing_sunglasses_and_working_as_a_lifeguard_at_pool'
    gt = load_images_to_array(gt_path)
    watermarked = load_images_to_array(watermarked_path)
    i = 0
    tLPs = []
    for gt_, watermarked_ in zip(gt, watermarked):
        watermarked_, ofy, ofx = crop_8x8(watermarked_)
        gt_, ofy, ofx = crop_8x8(gt_)
        
        gt_ = im2tensor(gt_)
        watermarked_ = im2tensor(watermarked_)
        if i != 0:
            dist0t = model.forward(gt_pre, gt_)
            dist1t = model.forward(watermarked_pre, watermarked_)
            dist01t = np.absolute(dist0t - dist1t) * 100.0
            tLPs.append( dist01t[0] )
        gt_pre = gt_
        watermarked_pre = watermarked_
        i += 1

    print(np.mean(np.array(tLPs)))

