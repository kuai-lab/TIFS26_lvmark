from utils_vid_attack import * 
import os 
import cv2
from tqdm import tqdm 

def save_all_frames(frame_list, path, fps): 
    if not os.path.exists(path):
        os.makedirs(path, exist_ok= True) 
    
    for idx,frame in enumerate(frame_list): 
        image_name = f'{idx}.png' 
        frame.save(os.path.join(path, image_name)) 


attacks = {
    'rotates' : lambda x: rotates(x,30),
    'crop' : lambda x : crop(x, 0.2),
    'resize' : lambda x :resize(x, 1.5),
    'perspective_projection' :lambda x: perspective_projection(x, [0.33, 0.25, 0.15, 0.85 ]),
    'temporal_loss_pass_filter' : lambda x: temporal_loss_pass_filter(x, kernel_size=4),
    'frame_rate_conversion' : lambda x: frame_rate_conversion(x, 30, 30),
    'blurring' : lambda x: blurring(x, 2.0),
    'color_jitter_brightness': lambda x: color_jitter(x, brightness=0.5, contrast=0.0, saturation=0.0),
    'color_jitter_contrast': lambda x: color_jitter(x, brightness=0.0, contrast=0.5, saturation=0.0),
    'color_jitter_saturation': lambda x: color_jitter(x, brightness=0.0, contrast=0.0, saturation=0.5),
    'Gaussian_noise': lambda x: Gaussian_noise(x, std=0.05),
    'frame_drop' : lambda x: frame_drop(x, drop_ratio=0.2),
    'frame_swap' : lambda x: frame_swap(x, num_swaps=5), 
    'gaussian_filtering' :lambda x :gaussian_filtering(x, kernel_size=5),
    'average_filtering': lambda x :average_filtering(x, kernel_size=5),
    'median_filtering' :lambda x: median_filtering(x, kernel_size=5),
    #
    'random_hue': lambda x: random_hue(x, p=1.0),
    'histogram_equalization': lambda x: histogram_equalization(x),
    'add_salt_and_pepper_noise': lambda x: add_salt_and_pepper_noise(x, salt_prob=0.1, pepper_prob=0.1),
    'add_speckle_noise': lambda x: add_speckle_noise(x, mean=0, std=0.1),
    'color_quantization': lambda x: color_quantization(x, num_colors=64),
    'floyd_steinberg_dithering': lambda x: floyd_steinberg_dithering(x, num_colors=8),
    'frame_freezing': lambda x: frame_freezing(x, freeze_duration=5),
    'frame_skipping': lambda x: frame_skipping(x, skip_interval=5),
    'box_blur': lambda x: box_blur(x, kernel_size=5),
    'motion_blur': lambda x: motion_blur(x, kernel_size=5),
    'random_pixel_corruption': lambda x: random_pixel_corruption(x, corruption_rate=0.05),
    'bicubic_interpolation': lambda x: bicubic_interpolation(x, scale_factor=1.5)
    # 'Video_H264Compression' : lambda x : Video_H264Compression(x)
}


# attack_method = attacks.keys() 

data_path = 'data/gt' 
video_name = sorted(os.listdir(data_path)) 



video_frames = [Image.open(os.path.join(data_path,img)) for img in video_name] 
# print(video_frames, sep='\n')
# codec = mp4v 
flag = 1
fps = 30
for k,v in attacks.items():
    attack_name, attack_method = k, v
    save_folder = os.path.join("Attack", k+f"_frame_{str(fps)}")

    current_attack_frames = attacks[attack_name](video_frames)
    save_all_frames(current_attack_frames, save_folder, fps) 
    

    if flag: 
        images_to_video(current_attack_frames,os.path.join(save_folder,f'{k}_fps_{fps}.mp4'), codec='mp4v', fps=fps)
    


