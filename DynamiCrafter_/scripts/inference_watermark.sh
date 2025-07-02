#!/bin/bash


ERROR_MESSAGE=""




model_path=(PATH_TO_MODEL)
name=DIRECTORY_NAME

bit_capacities=(32 128 256 512)
bch_bits=(14)


version=512 ##1024, 512, 256
# seed=123
gpu=1
ckpt=MODEL_CKPT_PATH
config=../configs/inference_512_v1.0_wm.yaml

prompt_dir=../prompts/512_ours
res_dir="results_watermark"

for bch_bit in "${bch_bits[@]}"; do
    for bit_capacity in "${bit_capacities[@]}"; do

        CUDA_VISIBLE_DEVICES=$gpu python3 scripts/evaluation/inference_watermark.py \
        --seed 42 \
        --ckpt_path $ckpt \
        --config $config \
        --savedir $res_dir/$name/$bch_bit/$bit_capacity \
        --n_samples 1 \
        --bs 1 --height 320 --width 512 \
        --unconditional_guidance_scale 7.5 \
        --ddim_steps 50 \
        --ddim_eta 1.0 \
        --prompt_dir $prompt_dir \
        --text_input \
        --video_length 16 \
        --frame_stride 24 \
        --timestep_spacing 'uniform_trailing' --guidance_rescale 0.7 \
        --int_dimension 128 --lr_mult 1 \
        --phi_dimension 32 \
        --vae_decoder_path $model_path/vae_decoder.pth \
        --mapping_path $model_path/mapping_network.pth \
        --msg_decoder_path $model_path/decoding_network.pth \
        --json_file PATH_TO_PROMPT_JSON_FILE \
        --affine_list ../../layer_noise_0.5.txt \
        --bit_capacity $bit_capacity \
        --bch_bits $bch_bit

        if [ $? -ne 0 ]; then
            ERROR_MESSAGE="${ERROR_MESSAGE} $res_dir/$name/$bch_bit/$bit_capacity;"
        fi

    done
done



if [ -n "$ERROR_MESSAGE" ]; then
    echo "$ERROR_MESSAGE" > error_log.txt
fi