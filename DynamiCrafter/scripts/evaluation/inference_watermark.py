import argparse, os, sys, glob
import datetime, time
from omegaconf import OmegaConf
from tqdm import tqdm
from einops import rearrange, repeat
from collections import OrderedDict

import torch
import torchvision
import torchvision.transforms as transforms
from pytorch_lightning import seed_everything
from PIL import Image
sys.path.insert(1, os.path.join(sys.path[0], '..', '..'))
from lvdm.models.samplers.ddim import DDIMSampler
from lvdm.models.samplers.ddim_multiplecond import DDIMSampler as DDIMSampler_multicond
from utils.utils import instantiate_from_config
import random
import bchlib
from customization import customize_vae_decoder
from attribution import MappingNetwork
import numpy as np
import cv2
# from torchvision.models import resnet18, ResNet18_Weights
import configparser as cfg
from torchvision.models import resnet18, ResNet18_Weights
# from decoder.hvdm_decoder import HVDM_with_Resnet_v12_resnet18 ,HVDM_with_Resnet50,HVDM_with_Resnet50_high_freq
# from decoder.utils import DWT_3Ds
import json
from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights, resnet34, ResNet34_Weights
import pdb
import pickle

def load_json_as_list(data_dir, json_file="prompts_with_keys120_32bit.json"):
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Convert each key-value pair to [key, value] and store in a list
    # key_value_list = [[key, value] for key, value in data.items()]
    prompt_file = []
    message_file = []
        
    # Iterate over key-value pairs and check if the corresponding image file exists
    for key, value in data.items():
        file_path = os.path.join(data_dir, f"prompt_imgs/{key}_.png")
        
        # Add to lists only if the file exists
        if os.path.exists(file_path):
            prompt_file.append(key)   # image, key pair list 
            message_file.append(value)
    
    return prompt_file, message_file
    

def get_filelist(data_dir, postfixes):
    patterns = [os.path.join(data_dir,f"*.{postfix}") for postfix in postfixes]
    file_list = []
    for pattern in patterns:
        file_list.extend(glob.glob(pattern))
    file_list.sort()
    return file_list

def load_model_checkpoint(model, ckpt):
    state_dict = torch.load(ckpt, map_location="cpu")
    if "state_dict" in list(state_dict.keys()):
        state_dict = state_dict["state_dict"]
        try:
            model.load_state_dict(state_dict, strict=True)
        except:
            ## rename the keys for 256x256 model
            new_pl_sd = OrderedDict()
            for k,v in state_dict.items():
                new_pl_sd[k] = v

            for k in list(new_pl_sd.keys()):
                if "framestride_embed" in k:
                    new_key = k.replace("framestride_embed", "fps_embedding")
                    new_pl_sd[new_key] = new_pl_sd[k]
                    del new_pl_sd[k]
            model.load_state_dict(new_pl_sd, strict=True)
    else:
        # deepspeed
        new_pl_sd = OrderedDict()
        for key in state_dict['module'].keys():
            new_pl_sd[key[16:]]=state_dict['module'][key]
        model.load_state_dict(new_pl_sd)
    print('>>> model checkpoint loaded.')
    return model

def load_prompts(prompt_file):
    f = open(prompt_file, 'r')
    prompt_list = []
    for idx, line in enumerate(f.readlines()):
        l = line.strip()
        if len(l) != 0:
            prompt_list.append(l)
        f.close()
    return prompt_list


def get_existing_file_paths(data_dir, prompt_file):
    # Generate paths and filter out any that don't exist
    file_list = [
        os.path.join(data_dir, f"prompt_imgs/{prompt}_.png")
        for prompt in prompt_file
        if os.path.exists(os.path.join(data_dir, f"prompt_imgs/{prompt}_.png"))
    ]
    
    return file_list


def load_data_prompts(data_dir, json_file, video_size=(256,256), video_frames=16, interp=False):
    transform = transforms.Compose([
        transforms.Resize(min(video_size)),
        transforms.CenterCrop(video_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

    prompt_file, message_file = load_json_as_list(data_dir,json_file) # .png 
    
    default_idx = 0
    default_idx = min(default_idx, len(prompt_file)-1)
    if len(prompt_file) > 1:
        print(f"Warning: multiple prompt files exist. The one {os.path.split(prompt_file[default_idx])[1]} is used.")
    ## only use the first one (sorted by name) if multiple exist
    
    ## load video
    # file_list = get_filelist(os.path.join(data_dir, 'prompt_imgs'), ['jpg', 'png', 'jpeg', 'JPEG', 'PNG'])
    file_list = [os.path.join(data_dir, os.path.join("prompt_imgs",f"{prompt}_.png")) for prompt in prompt_file] 

    print(file_list)
    data_list = []
    filename_list = []
    message_list = []
    # prompt_list = load_prompts(prompt_file[default_idx])
    prompt_list = prompt_file
    n_samples = len(prompt_list)

    
    for idx in range(n_samples):

        if interp:
            image1 = Image.open(file_list[2*idx]).convert('RGB')
            image_tensor1 = transform(image1).unsqueeze(1) # [c,1,h,w]
            image2 = Image.open(file_list[2*idx+1]).convert('RGB')
            image_tensor2 = transform(image2).unsqueeze(1) # [c,1,h,w]
            frame_tensor1 = repeat(image_tensor1, 'c t h w -> c (repeat t) h w', repeat=video_frames//2)
            frame_tensor2 = repeat(image_tensor2, 'c t h w -> c (repeat t) h w', repeat=video_frames//2)
            frame_tensor = torch.cat([frame_tensor1, frame_tensor2], dim=1)
            _, filename = os.path.split(file_list[idx*2])
            message = message_file[idx*2]
        else:
            image = Image.open(file_list[idx]).convert('RGB')
            image_tensor = transform(image).unsqueeze(1) # [c,1,h,w]
            frame_tensor = repeat(image_tensor, 'c t h w -> c (repeat t) h w', repeat=video_frames)
            _, filename = os.path.split(file_list[idx])
            message = message_file[idx]
        data_list.append(frame_tensor)
        filename_list.append(filename)
        message_list.append(message)
    prompt_list = [elem for item in filename_list for elem in prompt_list if item.startswith(elem)]
    return filename_list, data_list, prompt_list, message_list

def save_results(prompt, samples, filename, fakedir, fps=8, loop=False):
    filename = filename.split('.')[0]+'.mp4'
    prompt = prompt[0] if isinstance(prompt, list) else prompt

    ## save video
    videos = [samples]
    savedirs = [fakedir]
    for idx, video in enumerate(videos):
        if video is None:
            continue
        # b,c,t,h,w
        video = video.detach().cpu()
        video = torch.clamp(video.float(), -1., 1.)
        n = video.shape[0]
        video = video.permute(2, 0, 1, 3, 4) # t,n,c,h,w
        if loop:
            video = video[:-1,...]
        
        frame_grids = [torchvision.utils.make_grid(framesheet, nrow=int(n), padding=0) for framesheet in video] #[3, 1*h, n*w]
        grid = torch.stack(frame_grids, dim=0) # stack in temporal dim [t, 3, h, n*w]
        grid = (grid + 1.0) / 2.0
        grid = (grid * 255).to(torch.uint8).permute(0, 2, 3, 1)
        path = os.path.join(savedirs[idx], filename)
        torchvision.io.write_video(path, grid, fps=fps, video_codec='h264', options={'crf': '10'}) ## crf indicates the quality


def save_results_seperate(prompt, samples, filename,phis ,fakedir ,fps=10, loop=False):
    prompt = prompt[0] if isinstance(prompt, list) else prompt

    ## save video
    videos = [samples]
    savedirs = [fakedir]
    for idx, video in enumerate(videos):
        if video is None:
            continue

        video = video.detach().cpu()
        if loop: 
            video = video[:,:,:-1,...]
        video = torch.clamp(video.float(), -1., 1.).squeeze(0)

        path = os.path.join(savedirs[idx].replace('samples', f"key_{args.key_seed}"), f'{filename.split(".")[0][:30]}.avi')
        fourcc = cv2.VideoWriter_fourcc(*'FFV1') 
        frame_size = (video.shape[3], video.shape[2]) 
        out = cv2.VideoWriter(path, fourcc, fps, frame_size)

        video = rearrange(video, 'c t h w -> t h w c')
        for i in range(video.shape[0]):
            frame = video[i].numpy()  
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  
            frame = (frame + 1.0) / 2.0
            out.write((frame*255).astype(np.uint8)) 
        out.release()  

def get_latent_z(model, videos):
    b, c, t, h, w = videos.shape
    x = rearrange(videos, 'b c t h w -> (b t) c h w')
    z = model.encode_first_stage(x)
    z = rearrange(z, '(b t) c h w -> b c t h w', b=b, t=t)
    return z


def image_guided_synthesis(model, prompts,encoded_phis,affine_list, videos, noise_shape, n_samples=1, ddim_steps=50, ddim_eta=1., \
                        unconditional_guidance_scale=1.0, cfg_img=None, fs=None, text_input=False, multiple_cond_cfg=False, loop=False, interp=False, timestep_spacing='uniform', guidance_rescale=0.0, **kwargs):
    from torch.cuda.amp import autocast
    
    ddim_sampler = DDIMSampler(model) if not multiple_cond_cfg else DDIMSampler_multicond(model)
    batch_size = noise_shape[0]
    fs = torch.tensor([fs] * batch_size, dtype=torch.long, device=model.device)

    if not text_input:
        prompts = [""]*batch_size

    img = videos[:,:,0] #bchw
    img_emb = model.embedder(img) ## blc
    img_emb = model.image_proj_model(img_emb)

    cond_emb = model.get_learned_conditioning(prompts)
    cond = {"c_crossattn": [torch.cat([cond_emb,img_emb], dim=1)]}
    if model.model.conditioning_key == 'hybrid':
        z = get_latent_z(model, videos) # b c t h w
        if loop or interp:
            img_cat_cond = torch.zeros_like(z)
            img_cat_cond[:,:,0,:,:] = z[:,:,0,:,:]
            img_cat_cond[:,:,-1,:,:] = z[:,:,-1,:,:]
        else:
            img_cat_cond = z[:,:,:1,:,:]
            img_cat_cond = repeat(img_cat_cond, 'b c t h w -> b c (repeat t) h w', repeat=z.shape[2])
        cond["c_concat"] = [img_cat_cond] # b c 1 h w
    
    if unconditional_guidance_scale != 1.0:
        if model.uncond_type == "empty_seq":
            prompts = batch_size * [""]
            uc_emb = model.get_learned_conditioning(prompts)
        elif model.uncond_type == "zero_embed":
            uc_emb = torch.zeros_like(cond_emb)
        uc_img_emb = model.embedder(torch.zeros_like(img)) ## b l c
        uc_img_emb = model.image_proj_model(uc_img_emb)
        uc = {"c_crossattn": [torch.cat([uc_emb,uc_img_emb],dim=1)]}
        if model.model.conditioning_key == 'hybrid':
            uc["c_concat"] = [img_cat_cond]
    else:
        uc = None

    ## we need one more unconditioning image=yes, text=""
    if multiple_cond_cfg and cfg_img != 1.0:
        uc_2 = {"c_crossattn": [torch.cat([uc_emb,img_emb],dim=1)]}
        if model.model.conditioning_key == 'hybrid':
            uc_2["c_concat"] = [img_cat_cond]
        kwargs.update({"unconditional_conditioning_img_nonetext": uc_2})
    else:
        kwargs.update({"unconditional_conditioning_img_nonetext": None})

    z0 = None
    cond_mask = None

    batch_variants = []
    for _ in range(n_samples):

        if z0 is not None:
            cond_z0 = z0.clone()
            kwargs.update({"clean_cond": True})
        else:
            cond_z0 = None
        if ddim_sampler is not None:
            # pdb.set_trace()
            with autocast(dtype=torch.float16):
                samples, _ = ddim_sampler.sample(S=ddim_steps,
                                            conditioning=cond,
                                            batch_size=batch_size,
                                            shape=noise_shape[1:],
                                            verbose=False,
                                            unconditional_guidance_scale=unconditional_guidance_scale,
                                            unconditional_conditioning=uc,
                                            eta=ddim_eta,
                                            cfg_img=cfg_img, 
                                            mask=cond_mask,
                                            x0=cond_z0,
                                            fs=fs,
                                            timestep_spacing=timestep_spacing,
                                            guidance_rescale=guidance_rescale,
                                            **kwargs
                                            )


        samples = samples.to(torch.float16)
        batch_images = model.decode_first_stage(samples, encoded_phis, affine_list=affine_list)
        batch_variants.append(batch_images)
    ## variants, batch, c, t, h, w
    batch_variants = torch.stack(batch_variants)
    return batch_variants.permute(1, 0, 2, 3, 4, 5)


def get_phis_and_save(args, fakedir, filenames):
    print('aa')
    
    BCH_POLYNOMIAL = 8219 
    BCH_BITS = args.bch_bits  
    
    message_byte = []
    encoded_byte = []
    assert args.bit_capacity % 32 == 0
    for i in range(int(args.bit_capacity/32)):

    
        key = np.random.randint(2, size=32).tolist()

        bch = bchlib.BCH(BCH_BITS, BCH_POLYNOMIAL, swap_bits=False)
        message_byte_ = bytes(np.packbits(key, bitorder='little'))
        encoded_byte_ = bch.encode(message_byte_)
        key = torch.tensor(np.array(key), dtype=torch.float32)
        message_byte.append(message_byte_)
        encoded_byte.append(encoded_byte_)
        if i==0:
            phis = key.unsqueeze(0)
            
        else:
            phis = torch.cat((phis, key.unsqueeze(0)))
            
    
    
    data_BCH = {
        'BCH_message_byte' : message_byte,
        'BCH_encoded_byte' : encoded_byte,
        'keys': phis
    }
    phis = phis.repeat(int(16/(args.bit_capacity/32)),1) # (1234,1234,1234,1234)
    path_BCH = os.path.join(fakedir.replace('samples', f"key_{args.key_seed}"), f'{filenames[0].split(".")[0][:30]}')
    os.makedirs(path_BCH, exist_ok=True)
    # pdb.set_trace()
    with open(path_BCH+'/bch.pkl', 'wb') as f:
        pickle.dump(data_BCH, f)
        
    return phis
    



def run_inference(args, gpu_num, gpu_no):
    ## model config
    config = OmegaConf.load(args.config)
    model_config = config.pop("model", OmegaConf.create())
    
    ## set use_checkpoint as False as when using deepspeed, it encounters an error "deepspeed backend not set"
    model_config['params']['unet_config']['params']['use_checkpoint'] = False
    model = instantiate_from_config(model_config)
    model = model.cuda(gpu_no)
    model.perframe_ae = args.perframe_ae
    assert os.path.exists(args.ckpt_path), "Error: checkpoint Not Found!"
    model = load_model_checkpoint(model, args.ckpt_path)
    weight_dtype = torch.float16
    print('')
    print(f'Load watermark vae decoder : {args.vae_decoder_path}')
    model.first_stage_model = customize_vae_decoder(model.first_stage_model, args.int_dimension, args.lr_mult).cuda(gpu_no)
    vae_state_dict = torch.load(args.vae_decoder_path, map_location='cuda')
    model.first_stage_model.load_state_dict(vae_state_dict,strict=False)
    model.eval()
    print('Success!!!')
    
    print('')
    print(f'Load watermark Mapping Network : {args.mapping_path}')
    mapping_network = MappingNetwork(args.phi_dimension, args.int_dimension, num_layers=args.mapping_layer)
    mapping_network = mapping_network.cuda(gpu_no)
    mapping_state_dict = torch.load(args.mapping_path, map_location='cuda')
    
    # matching mapping network state dict keys
    parameter_names = dict(zip(mapping_state_dict.keys(), mapping_network.state_dict().keys()))
    state_dict = OrderedDict(dict((parameter_names[key], value) for (key, value) in mapping_state_dict.items()))
    mapping_network.load_state_dict(state_dict)
    mapping_network.eval()
    print('Success!!!')
    
    affine_list = []
    with open(args.affine_list, 'r') as file:
        for line in file:
            affine_list.append(line.strip())
    model = model.to(weight_dtype)
    mapping_network = mapping_network.to(weight_dtype)

    ## run over data
    assert (args.height % 16 == 0) and (args.width % 16 == 0), "Error: image size [h,w] should be multiples of 16!"
    assert args.bs == 1, "Current implementation only support [batch size = 1]!"
    ## latent noise shape
    h, w = args.height // 8, args.width // 8
    channels = model.model.diffusion_model.out_channels
    n_frames = args.video_length
    print(f'Inference with {n_frames} frames')
    noise_shape = [args.bs, channels, n_frames, h, w]

    fakedir = os.path.join(args.savedir, "samples")
    fakedir_separate = os.path.join(args.savedir, f"key_{args.key_seed}")

    # os.makedirs(fakedir, exist_ok=True)
    os.makedirs(fakedir_separate, exist_ok=True)
    print("save dir : ", fakedir_separate)
    ## prompt file setting
    assert os.path.exists(args.prompt_dir), "Error: prompt file Not Found!"
    filename_list, data_list, prompt_list, message_list = load_data_prompts(args.prompt_dir, json_file = args.json_file
                                                              ,video_size=(args.height, args.width), video_frames=n_frames, interp=args.interp)
    
    # filename_list = [next(item for item in filename_list if item.startswith(elem)) for elem in prompt_list]    
    # a = 1/0
    num_samples = len(prompt_list)
    samples_split = num_samples // gpu_num
    print('Prompts testing [rank:%d] %d/%d samples loaded.'%(gpu_no, samples_split, num_samples))
    #indices = random.choices(list(range(0, num_samples)), k=samples_per_device)
    indices = list(range(samples_split*gpu_no, samples_split*(gpu_no+1)))
    prompt_list_rank = [prompt_list[i] for i in indices]
    data_list_rank = [data_list[i] for i in indices]
    filename_list_rank = [filename_list[i] for i in indices]
    message_list_rank = [message_list[i] for i in indices]

    start = time.time()
 
    for idx, indice in tqdm(enumerate(range(0, len(prompt_list_rank), args.bs)), desc='Sample Batch'):
        prompts = prompt_list_rank[indice:indice+args.bs]
        videos = data_list_rank[indice:indice+args.bs]
        filenames = filename_list_rank[indice:indice+args.bs]
        messages = message_list_rank[indice:indice+args.bs][0] 
        print(messages)
        if isinstance(videos, list):
            videos = torch.stack(videos, dim=0).to("cuda")
        else:
            videos = videos.unsqueeze(0).to("cuda") # 1.png, 0100101....0000 <-> 1.png 0100101..0000 32 bits 
            
        # phis = torch.tensor([int(x) for x in messages], dtype=torch.float32).repeat(16, 1)
        
        phis = get_phis_and_save(args, fakedir, filenames)
        # np.random.randint(2, size=k)

        print(videos.dtype, phis.dtype, next(mapping_network.parameters()).dtype)
        phis = phis.cuda(gpu_no)
        encoded_phis = mapping_network(phis)
        
        videos = videos.to(weight_dtype)
        encoded_phis = encoded_phis.to(weight_dtype)
        a = time.time()
        # import pdb; pdb.set_trace()
        print('**************',filenames, messages,'*****************')
        batch_samples = image_guided_synthesis(model, prompts, encoded_phis, affine_list, videos, noise_shape, args.n_samples, args.ddim_steps, args.ddim_eta, \
                            args.unconditional_guidance_scale, args.cfg_img, args.frame_stride, args.text_input, args.multiple_cond_cfg, args.loop, args.interp, args.timestep_spacing, args.guidance_rescale)
        print(batch_samples.shape)
        b = time.time() 
        print("Spented Time : ", b-a, 's' )
        ## save each example individuall
        for nn, samples in enumerate(batch_samples):
            ## samples : [n_samples,c,t,h,w]
            prompt = prompts[nn]
            filename = filenames[nn]
            # save_results(prompt, samples, filename, fakedir, fps=8, loop=args.loop)
            save_results_seperate(prompt, samples, filename,phis ,fakedir,fps=8, loop=args.loop)

    print(f"Saved in {args.savedir}. Time used: {(time.time() - start):.2f} seconds")


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--savedir", type=str, default=None, help="results saving path")
    parser.add_argument("--ckpt_path", type=str, default=None, help="checkpoint path")
    parser.add_argument("--vae_decoder_path", type=str, default=None, help="vae checkpoint path")
    parser.add_argument("--mapping_path", type=str, default=None, help="mapping checkpoint path")
    parser.add_argument("--config", type=str, help="config (yaml) path")
    parser.add_argument("--prompt_dir", type=str, default=None, help="a data dir containing videos and prompts")
    parser.add_argument("--n_samples", type=int, default=1, help="num of samples per prompt",)
    parser.add_argument("--ddim_steps", type=int, default=50, help="steps of ddim if positive, otherwise use DDPM",)
    parser.add_argument("--ddim_eta", type=float, default=1.0, help="eta for ddim sampling (0.0 yields deterministic sampling)",)
    parser.add_argument("--bs", type=int, default=1, help="batch size for inference, should be one")
    parser.add_argument("--height", type=int, default=512, help="image height, in pixel space")
    parser.add_argument("--width", type=int, default=512, help="image width, in pixel space")
    parser.add_argument("--frame_stride", type=int, default=3, help="frame stride control for 256 model (larger->larger motion), FPS control for 512 or 1024 model (smaller->larger motion)")
    parser.add_argument("--unconditional_guidance_scale", type=float, default=1.0, help="prompt classifier-free guidance")
    parser.add_argument("--seed", type=int, default=123, help="seed for seed_everything")
    parser.add_argument("--video_length", type=int, default=16, help="inference video length")
    parser.add_argument("--negative_prompt", action='store_true', default=False, help="negative prompt")
    parser.add_argument("--text_input", action='store_true', default=False, help="input text to I2V model or not")
    parser.add_argument("--multiple_cond_cfg", action='store_true', default=False, help="use multi-condition cfg or not")
    parser.add_argument("--cfg_img", type=float, default=None, help="guidance scale for image conditioning")
    parser.add_argument("--timestep_spacing", type=str, default="uniform", help="The way the timesteps should be scaled. Refer to Table 2 of the [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://huggingface.co/papers/2305.08891) for more information.")
    parser.add_argument("--guidance_rescale", type=float, default=0.0, help="guidance rescale in [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://huggingface.co/papers/2305.08891)")
    parser.add_argument("--perframe_ae", action='store_true', default=False, help="if we use per-frame AE decoding, set it to True to save GPU memory, especially for the model of 576x1024")
    parser.add_argument("--int_dimension", type=int, default=128, help="message bits")
    parser.add_argument("--lr_mult", type=float, default=1.0, help="")
    parser.add_argument("--phi_dimension", type=int, default=32, help="message bits")
    parser.add_argument("--mapping_layer", type=int, default=2, help="message bits")
    parser.add_argument("--key_seed", type=int, default=42, help="message bits")
    parser.add_argument("--msg_decoder_path", type=str, default="", help="message bits")
    parser.add_argument("--affine_list", type=str, default="", help="affine list")
    parser.add_argument('--json_file', type=str, default="/home/jh/prompts_with_keys120_32bit.json")
    ## currently not support looping video and generative frame interpolation
    parser.add_argument("--loop", action='store_true', default=False, help="generate looping videos or not")
    parser.add_argument("--interp", action='store_true', default=False, help="generate generative frame interpolation or not")
    parser.add_argument("--bit_capacity", type=int, default=32, help="seed for seed_everything")
    parser.add_argument("--bch_bits", type=int, default=14, help="seed for seed_everything")
    return parser


if __name__ == '__main__':
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    print("@DynamiCrafter cond-Inference: %s"%now)
    parser = get_parser()
    args = parser.parse_args()

    seed = args.seed
    if seed < 0:
        seed = random.randint(0, 2 ** 31)
    seed_everything(seed)
    rank, gpu_num = 0, 1
    run_inference(args, gpu_num, rank)