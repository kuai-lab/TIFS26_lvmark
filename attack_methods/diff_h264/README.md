## H264CompressorNet Overview

H264CompressorNet is an autoencoder-based video compression neural network consisting of three Conv3d layers in the encoder and three ConvTranspose3d layers in the decoder.

- **H264CompressorNet**: A neural network composed of Conv3d layers for the encoder and ConvTranspose3d layers for the decoder.
- **H264Loss**: Combination of mse (mean squared error) and lpips (Learned Perceptual Image Patch Similarity).
- **VideoDataset**: A custom DataLoader class for loading video datasets.

## H264 Dataset Preparation

- **Input dataset**: WebVid
- **Target dataset**: WebVid compressed using ffmpeg

```bash
python original_h264.py
```

- Running this script will compress all videos in the `webvid/train` directory using ffmpeg's H264 compression and store the results in the `webvid/target` directory as the target dataset.

## H264 Training

```bash
python h264_train.py
```

### Training Example Results:

|  | **Batch Size** | **Frame Size** | **Train Dataset** | **Training Time per Epoch** | **Total Loss (lpips + mse)** | **VRAM Usage** |
| --- | --- | --- | --- | --- | --- | --- |
| Ours | 8 | 8 | WebVid 10000 | 1h | 0.01 (took only 10min) | 22GB |

### Training Configuration:

```python
# Default configuration
input_dir = 'dataset/webvid/train'
gt_dir = 'dataset/webvid/target'
criterion = H264Loss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
batch_size = 8
epochs = 20 
```

- **input_dir**: Path to the input video directory.
- **gt_dir**: Path to the ffmpeg-compressed target video directory.
- **loss**: mse + lpips
- **batch_size**: Batch size for training.
- **epochs**: Number of epochs.
- **resize**: Resolution (adjustable in the `VideoDataset` class in `diff_h264.py`).

## H264 Inference

```bash
python h264_inference.py
```

### Inference Configuration:

```python
input_dir = 'input_video'
output_dir = 'compressed_video'
gt_dir = "uncompressed_video"
compressed_out = 'compressed_output'

batch_size = 1
max_frames = 16
resize = 256
```

- **input_dir**: Path to the input video directory.
- **output_dir**: Path to the compressed output video directory.
- **gt_dir**: Path to the Ground Truth (uncompressed) video directory.

## Evaluation

```bash
python eval.py
```

### Evaluation Configuration:

```python
gt_video_dir = "compressed_output/uncompressed_video"
compare_video_dir = "compressed_output/compressed_video"
get_video_metric(gt_video_dir, compare_video_dir)
```

- **gt_video_dir**: Path to the Ground Truth video directory.
- **compare_video_dir**: Path to the video to be compared.

## Comparison of Metrics

| Metric         | SSIM  | PSNR  | LPIPS | L-inf |
|----------------|-------|-------|-------|-------|
| GT vs ffmpeg   | 0.96  | 38.52 | 0.10  | 41.56 |
| GT vs Ours     | 0.94  | 38.83 | 0.09  | 36.19 |
| ffmpeg vs Ours | 0.93  | 37.20 | 0.12  | 44.31 |

## Attack Types

```bash
python attack.py
```

- In the **ImageAugmentor** class, you can add, remove, or modify attack methods in the augmentations pipeline.
- The trained H264 model has also been added to the pipeline for attacks.

### Attack Configuration Example:

```python
K.ColorJiggle(brightness=0.0, contrast=0.0, saturation=0.0, hue=0.0, same_on_batch=False, p=1.0, keepdim=False),
K.RandomGaussianNoise(mean=0.0, std=1.0, same_on_batch=False, p=0.5, keepdim=False),
K.RandomHorizontalFlip(p=0.5, p_batch=1.0, same_on_batch=False, keepdim=False),
H264CompressorNet()
```

- **self.apply_augmentation**: Apply all attacks (each attack’s probability can be adjusted via the `p` parameter).
- **self.apply_random_augmentation**: Apply one randomly selected attack from the list (set `p=1` for all to ensure randomness).
- **self.apply_all_augmentations**: Apply each attack one by one and save the results to the `all_attacked` folder (for visualization).
- **self.check_differentitable**: Check if the gradient still flows after applying the augmentations.

## Attack Inference

```bash
python attack_inference.py
```

### Attack Inference Configuration:

```python
# Default configuration
input_dir = 'input_video'
output_dir = 'attacked_video'
gt_dir = "gt_video"
attacked_out = "attacked_output"

batch_size = 1
max_frames = 16
resize = 256
```

- **input_dir**: Path to the input video directory.
- **output_dir**: Path to the attacked video output directory.
- **gt_dir**: Path to the Ground Truth video directory.
```
