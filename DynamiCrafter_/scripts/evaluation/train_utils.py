import torch
import cv2
import itertools
import argparse
import matplotlib.pyplot as plt
import numpy as np
import math
import torch.nn.functional as F
import logging
import os
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20230211, help="seed for seed_everything")
    parser.add_argument("--height", type=int, default=224, help='Video hieght')
    parser.add_argument("--width", type=int, default=224, help='Video Width')
    parser.add_argument("--mode", default="base", type=str, help="which kind of inference mode: {'base', 'i2v'}")
    parser.add_argument("--ckpt_path", type=str, default=None, help="checkpoint path")
    parser.add_argument("--config", type=str, help="config (yaml) path")
    parser.add_argument("--prompt_file", type=str, default=None, help="a text file containing many prompts")
    parser.add_argument("--savedir", type=str, default=None, help="results saving path")
    parser.add_argument("--savefps", type=str, default=10, help="video fps to generate")
    parser.add_argument("--n_samples", type=int, default=1, help="num of samples per pㅠrompt",)
    parser.add_argument("--ddim_steps", type=int, default=50, help="steps of ddim if positive, otherwise use DDPM",)
    parser.add_argument("--ddim_eta", type=float, default=1.0, help="eta for ddim sampling (0.0 yields deterministic sampling)",)
    parser.add_argument("--bs", type=int, default=1, help="batch size for inference")
    parser.add_argument("--frames", type=int, default=-1, help="frames num to inference")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--frame_stride", type=int, default=3, help="frame stride control for 256 model (larger->larger motion), FPS control for 512 or 1024 model (smaller->larger motion)")
    parser.add_argument("--unconditional_guidance_scale", type=float, default=1.0, help="prompt classifier-free guidance")
    parser.add_argument("--unconditional_guidance_scale_temporal", type=float, default=None, help="temporal consistency guidance")
    parser.add_argument("--resolution", type=int, default=256, help='resolution size')
    parser.add_argument("--cond_input", type=str, default=None, help="data dir of conditional input")
    parser.add_argument("--data_path", type=str, default='/data/minhyuk/cvpr2025/Train_Dataset/panda70m_preprocessed')
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Initial learning rate (after the potential warmup period) to use.")
    parser.add_argument('--lr_scheduler', type=str, default='cosine_with_restarts', help='scheduler type to use')
    parser.add_argument('--cosine_cycle', type=int, default=1000, help='cosine_with _restarts option')
    parser.add_argument('--lr_warmup_steps', type=int, default=0, help='Number of steps for the warmyup in the lr scheduler')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--max_train_steps', type=int, default=1000, help="Total number of training steps to perform.  If provided, overrides num_train_epochs.")
    parser.add_argument('--num_epochs', type =int, default =300, help='epoch')
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-6, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--random_flip", action='store_true', help='whether to randomly flip images horizontally')
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--attack",type=str,default='',help=(
            "which attack methods to apply ('c' | 'r' | 'g' | 'b' | 'n' | 'e' | 'j' | ... | 'crgbnej' or 'all' || 'AE_b_1' | 'AE_c_6' | ...)"
            "e.g. 'cr' denotes random cropping ('c') and rotation ('r')"
            "Use 'crgbnej' or 'all' for the combined attack in the paper"
        )
    )
    parser.add_argument("--perframe_ae", action='store_true', default=False, help="if we use per-frame AE decoding, set it to True to save GPU memory, especially for the model of 576x1024")
    parser.add_argument('--phi_dimension', type=int, default=32, help='phi_dimension')
    parser.add_argument('--int_dimension', type=int, default=64, help='intermediate dimension')
    parser.add_argument('--mapping_layer', type=int, default=2, help='FC layers of mapping network')
    parser.add_argument('--lr_mult', type=float, default=1, help='Learning rate multiplier for the affine layers.')
    parser.add_argument("--output_dir", type=str, default='model_pth')
    parser.add_argument('--csv_name', type=str, default='val_results.txt')
    return parser
def visualize_two_videos_frames(video_1, video_2, num_frames=8, save_path='output_videos_frames_our.png'):
    """
    두 개의 비디오(CUDA 텐서)를 입력받아 각 비디오에서 8개의 프레임을 선택해 하나의 이미지로 저장합니다.
    Args:
        video_1, video_2 (torch.Tensor): 두 개의 비디오 텐서. T, C, H, W 형태.
        num_frames (int): 각 비디오에서 시각화할 프레임 수.
        save_path (str): 결과 이미지를 저장할 경로.
    """
    # 파일 경로와 디렉터리 경로 분리
    save_dir = os.path.dirname(save_path)
    print(save_dir)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    # 두 비디오에서 사용할 프레임 수를 정함 (min(T, num_frames)으로 제한)
    video_1_frames = min(video_1.shape[0], num_frames)
    video_2_frames = min(video_2.shape[0], num_frames)
    # 이미지 크기 및 subplot 생성 (2행 num_frames열로 배치)
    fig, axs = plt.subplots(2, num_frames, figsize=(16, 4))
    # 첫 번째 비디오의 프레임 배치 (첫 번째 행)
    for i in range(video_1_frames):
        frame = video_1[i].permute(1, 2, 0).cpu().detach().numpy()  # T, C, H, W -> H, W, C
        frame = (frame * 0.5) + 0.5  # [-1, 1] -> [0, 1]
        frame = np.clip(frame, 0, 1)  # 범위를 [0, 1]로 클램핑
        axs[0, i].imshow(frame)
        axs[0, i].axis('off')  # 축 제거
    # 두 번째 비디오의 프레임 배치 (두 번째 행)
    for i in range(video_2_frames):
        frame = video_2[i].permute(1, 2, 0).cpu().detach().numpy()  # T, C, H, W -> H, W, C
        frame = (frame * 0.5) + 0.5  # [-1, 1] -> [0, 1]
        frame = np.clip(frame, 0, 1)  # 범위를 [0, 1]로 클램핑
        axs[1, i].imshow(frame)
        axs[1, i].axis('off')  # 축 제거
    # 레이아웃 조정 및 이미지 저장
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
def normalize_tensor(tensor):
    return ((tensor + 1.0) / 2.0).clamp(0, 1)
def check_tensor_range(tensor, tensor_name):
    """
    텐서의 min, max 값을 출력하는 함수.
    Args:
        tensor (torch.Tensor): 확인할 텐서.
        tensor_name (str): 텐서의 이름 (출력용).
    """
    print(f"{tensor_name} - Min value: {tensor.min().item()}, Max value: {tensor.max().item()}")
def save_video(tensor, filename, fps=24):
    """
    PyTorch 텐서(1, 3, t, h, w)를 비디오로 저장하는 함수.
    Args:
        tensor (torch.Tensor): (1, 3, t, h, w)의 크기를 가진 텐서 (배치 크기는 1이어야 함).
        filename (str): 저장할 비디오 파일의 이름.
        fps (int): 비디오의 프레임 속도.
    """
    # 텐서를 정규화하고 클램프
    # check_tensor_range(tensor, '1')
    tensor = normalize_tensor(tensor)
    # tensor = tensor.clamp(0,1)
    # 텐서를 NumPy로 변환하고, 필요한 경우 텐서를 CPU로 옮김
    video_np = tensor.detach().squeeze(0).permute(1, 2, 3, 0).cpu().numpy()  # (t, h, w, 3)
    # 영상 크기 가져오기
    t, h, w, _ = video_np.shape
    # 비디오 저장 설정
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 코덱 설정
    out = cv2.VideoWriter(filename, fourcc, fps, (w, h))
    for i in range(t):
        # 프레임을 [0, 255] 범위로 변환
        frame = (video_np[i] * 255).astype(np.uint8)
        # OpenCV는 BGR 형식을 사용하므로, RGB에서 BGR로 변환
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # 프레임을 비디오에 기록
        out.write(frame)
    out.release()  # 비디오 파일 저장 완료
def get_phis(phi_dimension, batch_size ,eps = 1e-8):
    phi_length = phi_dimension
    b = batch_size
    phi = torch.empty(b,phi_length).uniform_(0,1)
    return torch.bernoulli(phi) + eps

def get_params_optimize(vaed, mapping_network, decoding_network):
    params_to_optimize = itertools.chain(vaed.parameters(), mapping_network.parameters(), decoding_network.parameters())
    return params_to_optimize
def check_model_gradient_flow(model, mapping_network, decoding_network) :
        # VAE Decoder의 그라디언트 확인
    print("Checking VAE Decoder gradients...")
    for name, param in model.module.first_stage_model.decoder.named_parameters():
        if param.requires_grad and param.grad is None:
            print(f"Parameter: {name} has no gradient but requires_grad=True")
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                print(f"Parameter: {name} has NaN gradients!")
            # print(f"{name} gradient min: {param.grad.min()}, max: {param.grad.max()}")
    # Mapping Network의 그라디언트 확인
    print("Checking Mapping Network gradients...")
    for name, param in mapping_network.named_parameters():
        if param.requires_grad and param.grad is None:
            print(f"Parameter: {name} has no gradient but requires_grad=True")
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                print(f"Parameter: {name} has NaN gradients!")
            # print(f"{name} gradient min: {param.grad.min()}, max: {param.grad.max()}")
    # Decoding Network의 그라디언트 확인
    print("Checking Decoding Network gradients...")
    for name, param in decoding_network.named_parameters():
        if param.requires_grad and param.grad is None:
            print(f"Parameter: {name} has no gradient but requires_grad=True")
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                print(f"Parameter: {name} has NaN gradients!")
            # print(f"{name} gradient min: {param.grad.min()}, max: {param.grad.max()}")
def calculate_psnr(image_tensor, generated_video, max_pixel_value=1.0):
    """
    image_tensor: [T, C, H, W] 형태의 원본 비디오 텐서
    generated_video: [T, C, H, W] 형태의 생성된 비디오 텐서
    max_pixel_value: 픽셀의 최대 값 (8비트 이미지의 경우 255.0)
    return: PSNR 값
    """
    # 두 텐서 간의 MSE 계산
    mse = F.mse_loss(image_tensor, generated_video)
    # MSE가 0이면 PSNR은 무한대가 되므로, 이를 방지
    if mse == 0:
        return float('inf')
    # PSNR 계산
    psnr = 20 * math.log10(max_pixel_value) - 10 * math.log10(mse.item())
    return psnr
def log_metrics(loss_key, loss_lpips_reg, bit_acc, psnr, iteration):
    log_message = (f"Iteration: {iteration}, "
                   f"Key Loss: {loss_key:.4f}, "
                   f"LPIPS Loss: {loss_lpips_reg:.4f}, "
                   f"Bit Accuracy: {bit_acc.mean().item():.4f}, "
                   f"PSNR: {psnr:.4f}")
    # 로그에 출력
    logging.info(log_message)
    print(log_message)  # 콘솔에 동시에 출력하고 싶다면 추가
def remove_module_prefix(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("module.", "")  # "module." 접두어 제거
        new_state_dict[new_key] = value
    return new_state_dict
