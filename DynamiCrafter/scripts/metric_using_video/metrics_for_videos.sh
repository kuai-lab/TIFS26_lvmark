NBITS=(32)
SEEDS=(42)

# ./prompts_sora 파일 읽기
mapfile -t lines < #path of prompts_sora.txt

# PROMPTS 변수에 저장
PROMPTS=()

for SEED in "${SEEDS[@]}"; 
do
    for NBIT in "${NBITS[@]}"; 
    do
        WM_VIDEO= #path of watermarked video
        ORIGIN_VIDEO= #path of original video

        MODEL_PATH= # path of model path
        KEY_PATH= # path of key path

        for PROMPT in "${lines[@]}"; 
        do
            # echo ${PROMPT}
            CUDA_VISIBLE_DEVICES=0 python metrics_for_videos.py --eval_imgs True --eval_bits True --eval_vid True --eval_bits_memoriable True --output_data_name "" \
                --watermarked_video "PATH_TO_WATERMARKED_VIDEO" --original_video "PATH_TO_ORIGINAL_VIDEO" --output_dir "PATH_TO_OUTPUT_DIR"  --nbits ${NBIT} --key_path ${KEY_PATH} \
                --msg_decoder_path "${MODEL_PATH}/decoding_network.pth"
        done
    done
done


