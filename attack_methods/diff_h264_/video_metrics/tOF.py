import numpy as np
import cv2
import os, sys
import pandas as pd
from video_metrics.LPIPSmodels import util
import video_metrics.LPIPSmodels.dist_model as dm
import pdb
from torchvision import transforms
from PIL import Image
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

def tOF(img_dir, img_dir_nw):
    model = dm.DistModel()
    model.initialize(model='net-lin',net='alex',use_gpu=True)
    gt = load_images_to_array(img_dir_nw)
    watermarked = load_images_to_array(img_dir)
    i = 0
    tOFs = []
    for gt_, watermarked_ in zip(gt, watermarked):
        watermarked_grey = cv2.cvtColor(watermarked_, cv2.COLOR_RGB2GRAY)
        gt_grey = cv2.cvtColor(gt_, cv2.COLOR_RGB2GRAY)
        if i != 0:
            gt_OF=cv2.calcOpticalFlowFarneback(pre_gt_grey, gt_grey, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            watermarked_OF=cv2.calcOpticalFlowFarneback(pre_watermarked_grey, watermarked_grey, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            gt_OF, ofy, ofx = crop_8x8(gt_OF)
            watermarked_OF, ofy, ofx = crop_8x8(watermarked_OF)
            OF_diff = np.absolute(gt_OF - watermarked_OF)
            OF_diff = np.sqrt(np.sum(OF_diff * OF_diff, axis = -1)) # l1 vector norm
            # OF_diff, ofy, ofx = crop_8x8(OF_diff)
            tOFs.append( OF_diff.mean() )
        pre_watermarked_grey = watermarked_grey
        pre_gt_grey = gt_grey
        i += 1
    return np.mean(np.array(tOFs))

if __name__ == '__main__':

    model = dm.DistModel()
    model.initialize(model='net-lin',net='alex',use_gpu=True)

    gt_path = '/ssd/CVPR2025/diffusion/Latte/sample_videos_nw/t2v-a_cat_wearing_sunglasses_and_working_as_a_lifeguard_at_pool'
    watermarked_path = '/ssd/CVPR2025/diffusion/Latte/sample_videos_w/t2v-a_cat_wearing_sunglasses_and_working_as_a_lifeguard_at_pool'
    gt = load_images_to_array(gt_path)
    watermarked = load_images_to_array(watermarked_path)

    i = 0
    tOFs = []
    for gt_, watermarked_ in zip(gt, watermarked):
        watermarked_grey = cv2.cvtColor(watermarked_, cv2.COLOR_RGB2GRAY)
        gt_grey = cv2.cvtColor(gt_, cv2.COLOR_RGB2GRAY)
        if i != 0:
            gt_OF=cv2.calcOpticalFlowFarneback(pre_gt_grey, gt_grey, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            watermarked_OF=cv2.calcOpticalFlowFarneback(pre_watermarked_grey, watermarked_grey, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            gt_OF, ofy, ofx = crop_8x8(gt_OF)
            watermarked_OF, ofy, ofx = crop_8x8(watermarked_OF)
            OF_diff = np.absolute(gt_OF - watermarked_OF)
            OF_diff = np.sqrt(np.sum(OF_diff * OF_diff, axis = -1)) # l1 vector norm
            # OF_diff, ofy, ofx = crop_8x8(OF_diff)
            tOFs.append( OF_diff.mean() )
        pre_watermarked_grey = watermarked_grey
        pre_gt_grey = gt_grey
        i += 1

    print(np.mean(np.array(tOFs)))

