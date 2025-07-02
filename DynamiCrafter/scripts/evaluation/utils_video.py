# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import time
import datetime
import os
import subprocess
import functools
from collections import defaultdict, deque

import numpy as np
from PIL import Image
import pdb
import cv2
from torchvision import transforms

import torch
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision.datasets.folder import is_image_file, default_loader
from torchvision.transforms.functional import to_tensor
import random
### Optimizer building

def parse_params(s):
    """
    Parse parameters into a dictionary, used for optimizer and scheduler parsing.
    Example: 
        "SGD,lr=0.01" -> {"name": "SGD", "lr": 0.01}
    """
    s = s.replace(' ', '').split(',')
    params = {}
    params['name'] = s[0]
    for x in s[1:]:
        x = x.split('=')
        params[x[0]]=float(x[1])
    return params

def build_optimizer(name, model_params, **optim_params):
    """ Build optimizer from a dictionary of parameters """
    torch_optimizers = sorted(name for name in torch.optim.__dict__
        if name[0].isupper() and not name.startswith("__")
        and callable(torch.optim.__dict__[name]))
    if hasattr(torch.optim, name):
        return getattr(torch.optim, name)(model_params, **optim_params)
    raise ValueError(f'Unknown optimizer "{name}", choose among {str(torch_optimizers)}')

def adjust_learning_rate(optimizer, step, steps, warmup_steps, blr, min_lr=1e-6):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if step < warmup_steps:
        lr = blr * step / warmup_steps 
    else:
        lr = min_lr + (blr - min_lr) * 0.5 * (1. + math.cos(math.pi * (step - warmup_steps) / (steps - warmup_steps)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


def is_video_file(filename):
    return filename.lower().endswith(('.mp4', '.avi', '.mov'))

@functools.lru_cache()
def get_video_paths(path):
    paths = []
    for root, _, files in os.walk(path):
        for filename in files:
            if is_video_file(filename):
                paths.append(os.path.join(root, filename))
    return sorted(paths)

class VideoFolder(Dataset):
    """A video folder dataset intended for self-supervised learning."""

    def __init__(self, path, img_size, loader=cv2.VideoCapture, n_frames=16):
        self.samples = get_video_paths(path)
        self.loader = loader
        self.n_frames = n_frames
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # transforms.Lambda(lambda x: x.half())
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        video_path = self.samples[idx]
        cap = self.loader(video_path)

        frames = []
        success, frame = cap.read()
        while success and len(frames) < self.n_frames:
            if self.transform:
                frame = self.transform(frame)
            else:
                frame = to_tensor(frame)
            frames.append(frame)
            success, frame = cap.read()

        cap.release()

        # If the video has less than n_frames, repeat the last frame
        while len(frames) < self.n_frames:
            frames.append(frames[-1].clone())

        return torch.stack(frames)
    
class VideoFolder_bfchw(Dataset):
    """A video folder dataset intended for self-supervised learning, returning videos in (channel, frame, height, width) format."""

    def __init__(self, path, img_size, loader=cv2.VideoCapture, n_frames=16):
        self.samples = get_video_paths(path)
        self.loader = loader
        self.n_frames = n_frames
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            # transforms.Lambda(lambda x: x.half())
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        video_path = self.samples[idx]
        cap = self.loader(video_path)

        frames = []
        success, frame = cap.read()
        while success and len(frames) < self.n_frames:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if self.transform:
                frame = self.transform(frame)
            frames.append(frame)
            success, frame = cap.read()

        cap.release()

        # If the video has fewer than n_frames, repeat the last frame
        while len(frames) < self.n_frames:
            frames.append(frames[-1].clone())

        # Stack frames along the frame dimension (channel, frame, height, width)
        frames = torch.stack(frames, dim=0)

        return frames

class VideoFolder_bfchw_ours(Dataset):
    """A video folder dataset intended for self-supervised learning, returning videos in (channel, frame, height, width) format."""

    def __init__(self, transforms, path, loader=cv2.VideoCapture, n_frames=16):
        self.samples = get_video_paths(path)
        self.loader = loader
        self.n_frames = n_frames
        self.transform = transforms

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        video_path = self.samples[idx]
        cap = self.loader(video_path)

        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Randomly select a starting frame index
        if total_frames < self.n_frames:
            start_frame = 0  # Not enough frames, start from the beginning
        else:
            start_frame = random.randint(0, total_frames - self.n_frames)

        # Set the video position to the start_frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        for _ in range(self.n_frames):
            success, frame = cap.read()

            if not success:
                break
            if self.transform:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = self.transform(frame)
            frames.append(frame)

        cap.release()

        # If the video has fewer than n_frames, repeat the last frame
        while len(frames) < self.n_frames:
            frames.append(frames[-1].clone())

        # Stack frames along the frame dimension (channel, frame, height, width)
        frames = torch.stack(frames, dim=1)

        return frames

def video_collate_fn(batch):
    """ Collate function for data loader. Allows to have video frames of different sizes"""
    return torch.stack(batch)

def get_video_dataloader(data_dir, img_size, batch_size=128, n_frames=16, num_imgs=None, shuffle=False, num_workers=4):
    """ Get dataloader for the videos in the data_dir."""
    dataset = VideoFolder_bfchw(data_dir, img_size, n_frames=n_frames)
    
    if num_imgs is not None:
        dataset = torch.utils.data.Subset(dataset, np.random.choice(len(dataset), num_imgs, replace=False))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True, drop_last=False, collate_fn=video_collate_fn)



def get_video_dataloader_ours(data_dir, transforms, batch_size=128, n_frames=16, num_imgs=None, shuffle=False, num_workers=4):
    """ Get dataloader for the videos in the data_dir."""
    dataset = VideoFolder_bfchw_ours(transforms, data_dir, n_frames=n_frames)
    
    if num_imgs is not None:
        dataset = torch.utils.data.Subset(dataset, np.random.choice(len(dataset), num_imgs, replace=False))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True, drop_last=False, collate_fn=video_collate_fn)


# ### Data loading

# @functools.lru_cache()
# def get_image_paths(path):
#     paths = []
#     for path, _, files in os.walk(path):
#         for filename in files:
#             paths.append(os.path.join(path, filename))
#     return sorted([fn for fn in paths if is_image_file(fn)])

# class ImageFolder:
#     """An image folder dataset intended for self-supervised learning."""

#     def __init__(self, path, transform=None, loader=default_loader):
#         self.samples = get_image_paths(path)
#         self.loader = loader
#         self.transform = transform

#     def __getitem__(self, idx: int):
#         assert 0 <= idx < len(self)
#         img = self.loader(self.samples[idx])
#         if self.transform:
#             return self.transform(img)
#         return img

#     def __len__(self):
#         return len(self.samples)

# def collate_fn(batch):
#     """ Collate function for data loader. Allows to have img of different size"""
#     return batch

# def get_dataloader(data_dir, transform, batch_size=128, num_imgs=None, shuffle=False, num_workers=4, collate_fn=collate_fn):
#     """ Get dataloader for the images in the data_dir. The data_dir must be of the form: input/0/... """
#     dataset = ImageFolder(data_dir, transform=transform)
#     if num_imgs is not None:
#         dataset = Subset(dataset, np.random.choice(len(dataset), num_imgs, replace=False))
#     return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True, drop_last=False, collate_fn=collate_fn)

def pil_imgs_from_folder(folder):
    """ Get all images in the folder as PIL images """
    images = []
    filenames = []
    for filename in os.listdir(folder):
        try:
            img = Image.open(os.path.join(folder,filename))
            if img is not None:
                filenames.append(filename)
                images.append(img)
        except:
            print("Error opening image: ", filename)
    return images, filenames

### Metric logging

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.6f} ({global_avg:.6f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)

class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.6f}')
        data_time = SmoothedValue(fmt='{avg:.6f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.6f} s / it)'.format(header, total_time_str, total_time / (len(iterable)+1)))

### Misc 

def bool_inst(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('Boolean value expected in args')

def get_sha():
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode('ascii').strip()
    sha = 'N/A'
    diff = "clean"
    branch = 'N/A'
    try:
        sha = _run(['git', 'rev-parse', 'HEAD'])
        subprocess.check_output(['git', 'diff'], cwd=cwd)
        diff = _run(['git', 'diff-index', 'HEAD'])
        diff = "has uncommited changes" if diff else "clean"
        branch = _run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
    except Exception:
        pass
    message = f"sha: {sha}, status: {diff}, branch: {branch}"
    return message

if __name__ =="__main__":
    train_dataloader = get_video_dataloader()