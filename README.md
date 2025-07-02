# LVMark: Robust Watermark for Latent Video Diffusion Models

## Overview
<img src="assets/LVMark.png" width="450">
Rapid advancements in video diffusion models have enabled the creation of realistic videos, raising concerns about unauthorized use and driving the demand for techniques to protect model ownership. Existing watermarking methods, while effective for image diffusion models, do not account for temporal consistency, leading to degraded video quality and reduced robustness against video distortions. To address this issue, we introduce LVMark, a novel watermarking method for video diffusion models. We propose a new watermark decoder tailored for generated videos by learning the consistency between adjacent frames. It ensures accurate message decoding, even under malicious attacks, by combining the low-frequency components of the 3D wavelet domain with the RGB features of the video. Additionally, our approach minimizes video quality degradation by embedding watermark messages in layers with minimal impact on visual appearance using an importance-based weight modulation strategy. We optimize both the watermark decoder and the latent decoder of diffusion model, effectively balancing the trade-off between visual quality and bit accuracy. Our experiments show that our method embeds invisible watermarks into video diffusion models, ensuring robust decoding accuracy with 512-bit capacity, even under video distortions.

## Requirements
Preliminary requirements:
- Python>=3.8
- PyTorch==2.4.1

Run the following command:
```
pip3 install -r requirements.txt
```

## Datasets
The training dataset can be downloaded from:
[Panda-70M](https://github.com/snap-research/Panda-70M)

## Code Usage


To evaluate watermark robustness against video compression, we include a pretrained H.264 distortion network as part of our Attack Distortion Layer.  
This module simulates real-world H.264 encoding effects, ensuring that watermarks can be reliably detected even after compression.

Checkpoint path:  
`attack_methods/diff_h264/checkpoints/model_weights_epoch_20.pth`


Use `trainval_LVMark.py` to train and evaluate the model:
```
CUDA_VISIBLE_DEVICES=0 accelerate launch ./trainval_LVMark.py \
    --pretrained_model_name_or_path ./DynamiCrafter/checkpoints/dynamicrafter_512_v1/model.ckpt \
    --model_config ./DynamiCrafter/configs/inference_1024_v1.0.yaml \
    --center_crop \
    --dataloader_num_workers 8 \
    --train_batch_size 1 \
    --exp_name {exp name} \
    --train_dir {train dataset path} \
    --val_dir {valid dataset path}\
    --n_frames 8 \
    --resolution 256 \
    --train_steps_per_epoch 250 \
    --max_train_steps 15000 \
    --checkpointing_steps 250 \
    --phi_dimension 32 \
    --attack all \
    --output_dir output_1024 \
    --seed 2777 --dim 2048 \
    --learning_rate 1e-4 \
    --lambda_w 0.8 \
    --lambda_lpips 0.7 \
    --lr_mult 1 \
    --affine_list_path ./layer_noise_0.5.txt \
    --lambda_patch_l1 10
```

Use `inference_watermark.sh` to generate watermarked videos:
```
cd ./DynamiCrafter/scripts
. ./inference_watermark.sh
```

Use `metrics_for_videos.sh` to compute evaluation metrics:
```
cd ./DynamiCrafter/scripts/metric_using_video
. ./metrics_for_videos.sh
```

## Integration with Open-Sora

To embed watermarks into Open-Sora, please refer to the official repository:  
[Open-Sora](https://github.com/hpcaitech/Open-Sora)

You can train Open-Sora using the provided `customization_sora.py` and `layer_noise_0.5_sora.txt` files.


## Citation
```bibtex
@misc{jang2025lvmarkrobustwatermarklatent,
      title={LVMark: Robust Watermark for Latent Video Diffusion Models}, 
      author={MinHyuk Jang and Youngdong Jang and JaeHyeok Lee and Feng Yang and Gyeongrok Oh and Jongheon Jeong and Sangpil Kim},
      year={2025},
      eprint={2412.09122},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2412.09122}, 
}
```
