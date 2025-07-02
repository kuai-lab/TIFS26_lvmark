import torch
import kornia.augmentation as K
import random
import torchvision.transforms as transforms
import os
from PIL import Image
import matplotlib.pyplot as plt
from diff_h264 import H264CompressorNet

class ImageAugmentor:
    def __init__(self, augmentations=None):
        if augmentations is None:
            augmentations = [
                K.ColorJiggle(brightness=1.0, contrast=1.0, saturation=0.0, hue=0.0, same_on_batch=False, p=1.0, keepdim=False),
                K.ColorJitter(brightness=0.0, contrast=0.0, saturation=1.0, hue=1.0, same_on_batch=False, p=1.0, keepdim=False),
                K.RandomAutoContrast(clip_output=True, same_on_batch=False, p=1.0, keepdim=False),
                K.RandomBoxBlur(kernel_size=(3, 3), border_type='reflect', normalized=True, same_on_batch=False, p=0.5, keepdim=False),
                K.RandomBrightness(brightness=(1.0, 1.0), clip_output=True, same_on_batch=False, p=1.0, keepdim=False),
                K.RandomChannelDropout(num_drop_channels=1, fill_value=0.0, same_on_batch=False, p=0.5, keepdim=False),
                K.RandomChannelShuffle(same_on_batch=False, p=0.5, keepdim=False),
                K.RandomClahe(clip_limit=(40.0, 40.0), grid_size=(8, 8), slow_and_differentiable=False, same_on_batch=False, p=0.5, keepdim=False),
                K.RandomContrast(contrast=(1.0, 1.0), clip_output=True, same_on_batch=False, p=1.0, keepdim=False),
                K.RandomEqualize(same_on_batch=False, p=0.5, keepdim=False),
                K.RandomGamma(gamma=(1.0, 1.0), gain=(1.0, 1.0), same_on_batch=False, p=1.0, keepdim=False),
                K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0), border_type='reflect', separable=True, same_on_batch=False, p=0.5, keepdim=False),
                K.RandomGaussianIllumination(gain=(0.01, 0.15), center=(0.1, 0.9), sigma=(0.2, 1.0), sign=(-1.0, 1.0), p=0.5, same_on_batch=False, keepdim=False),
                K.RandomGaussianNoise(mean=0.0, std=1.0, same_on_batch=False, p=0.5, keepdim=False),
                K.RandomGrayscale(rgb_weights=None, same_on_batch=False, p=0.1, keepdim=False),
                K.RandomHue(hue=(0.0, 0.0), same_on_batch=False, p=1.0, keepdim=False),
                K.RandomInvert(max_val=torch.tensor(1.0), same_on_batch=False, p=0.5, keepdim=False),
                K.RandomJPEG(jpeg_quality=50.0, same_on_batch=False, p=1.0, keepdim=False),
                K.RandomLinearCornerIllumination(gain=(0.01, 0.2), sign=(-1.0, 1.0), p=0.5, same_on_batch=False, keepdim=False),
                K.RandomLinearIllumination(gain=(0.01, 0.2), sign=(-1.0, 1.0), p=0.5, same_on_batch=False, keepdim=False),
                K.RandomMedianBlur(kernel_size=(3, 3), same_on_batch=False, p=0.5, keepdim=False),
                K.RandomMotionBlur(kernel_size=3, angle=35., direction=0.5, same_on_batch=False, p=0.5, keepdim=False),
                K.RandomPlanckianJitter(mode='blackbody', select_from=None, same_on_batch=False, p=0.5, keepdim=False),
                K.RandomPlasmaBrightness(roughness=(0.1, 0.7), intensity=(0.0, 1.0), same_on_batch=False, p=0.5, keepdim=False),
                K.RandomPlasmaContrast(roughness=(0.1, 0.7), same_on_batch=False, p=0.5, keepdim=False),
                K.RandomPlasmaShadow(roughness=(0.1, 0.7), shade_intensity=(-1.0, 0.0), shade_quantity=(0.0, 1.0), same_on_batch=False, p=0.5, keepdim=False),
                K.RandomPosterize(bits=3, same_on_batch=False, p=0.5, keepdim=False),
                K.RandomRain(number_of_drops=(1000, 2000), drop_height=(5, 20), drop_width=(-5, 5), same_on_batch=False, p=0.5, keepdim=False),
                K.RandomRGBShift(r_shift_limit=0.5, g_shift_limit=0.5, b_shift_limit=0.5, same_on_batch=False, p=0.5, keepdim=False),
                K.RandomSaltAndPepperNoise(amount=(0.01, 0.06), salt_vs_pepper=(0.4, 0.6), p=0.5, same_on_batch=False, keepdim=False),
                K.RandomSaturation(saturation=(1.0, 1.0), same_on_batch=False, p=1.0, keepdim=False),
                K.RandomSharpness(sharpness=0.5, same_on_batch=False, p=0.5, keepdim=False),
                K.RandomSnow(snow_coefficient=(0.5, 0.5), brightness=(2, 2), same_on_batch=False, p=1.0, keepdim=False),
                K.RandomSolarize(thresholds=0.1, additions=0.1, same_on_batch=False, p=0.5, keepdim=False),
                K.CenterCrop(size=(128,128), align_corners=True, p=1.0, keepdim=False, cropping_mode='slice'),
                K.PadTo((4,5), pad_mode='constant', pad_value=0, keepdim=False),
                K.RandomAffine(degrees=(-15.,20.), translate=None, scale=None, shear=None, same_on_batch=False, align_corners=False, p=0.5, keepdim=False),
                K.RandomCrop(size=(2,2), padding=None, pad_if_needed=False, fill=0, padding_mode='constant', same_on_batch=False, align_corners=True, p=1.0, keepdim=False, cropping_mode='slice'),
                K.RandomElasticTransform(kernel_size=(63, 63), sigma=(32.0, 32.0), alpha=(1.0, 1.0), align_corners=False, padding_mode='zeros', same_on_batch=False, p=0.5, keepdim=False),
                K.RandomErasing(scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0.0, same_on_batch=False, p=0.5, keepdim=False),
                K.RandomFisheye(center_x=torch.tensor([-.3, .3]), center_y=torch.tensor([-.3, .3]), gamma=torch.tensor([.9, 1.]), same_on_batch=False, p=0.5, keepdim=False),
                K.RandomHorizontalFlip(p=0.5, p_batch=1.0, same_on_batch=False, keepdim=False),
                K.RandomPerspective(distortion_scale=0.5, same_on_batch=False, align_corners=False, p=0.5, keepdim=False, sampling_method='basic'),
                K.RandomResizedCrop(size=(128,128), scale=(0.8, 1.2), ratio=(0.75, 1.33), same_on_batch=False, align_corners=True, p=1.0, keepdim=False, cropping_mode='resample'),
                K.RandomRotation(degrees=45.0, same_on_batch=False, align_corners=True, p=0.5, keepdim=False),
                K.RandomShear(shear=(-5., 2., 5., 10.), same_on_batch=False, align_corners=False, p=0.5, keepdim=False),
                K.RandomThinPlateSpline(scale=0.2, align_corners=False, same_on_batch=False, p=0.5, keepdim=False),
                K.RandomVerticalFlip(p=0.5, p_batch=1.0, same_on_batch=False, keepdim=False),
                H264CompressorNet()
            ]
        self.augment_pipeline = torch.nn.Sequential(*augmentations)
        self.augmentations = augmentations
        

    def check_differentiable(self, images_tensor):
        """
        증강 적용 후 텐서가 미분 가능한지 확인하는 함수
        Args:
            images_tensor: 이미지 텐서
        """
        images_tensor.requires_grad_(True)
        augmented_tensor = self.augment_pipeline(images_tensor)
        augmented_tensor.sum().backward()  # 역전파 계산
        return images_tensor.grad is not None

    def frame_to_batch(self, tensor):
        """
        B, C, T, H, W -> T, C, H, W로 변환
        Args:
            tensor: B, C, T, H, W 형태의 텐서
        Returns:
            변환된 텐서: T, C, H, W
        """
        tensor_tchw = tensor.squeeze(0).permute(1, 0, 2, 3)  # (T, C, H, W)
        return tensor_tchw

    def batch_to_frame(self, tensor):
        """
        T, C, H, W -> 1, C, T, H, W로 변환
        Args:
            tensor: T, C, H, W 형태의 텐서
        Returns:
            변환된 텐서: 1, C, T, H, W
        """
        tensor_bcthw = tensor.unsqueeze(0).permute(0, 2, 1, 3, 4)
        return tensor_bcthw
    
    def save_tensor_images(self, tensor, folder_path):
        """
        4차원 이미지 텐서(B, C, H, W)를 배치별로 폴더에 저장하는 함수
        Args:
            tensor: (B, C, H, W) 형태의 이미지 텐서
            folder_path: 이미지를 저장할 폴더 경로
        """
        B, C, H, W = tensor.shape
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        transform = transforms.ToPILImage()

        for i in range(B):
            img_tensor = tensor[i]
            img = transform(img_tensor)
            img_path = os.path.join(folder_path, f"image_{i}.png")
            img.save(img_path)

    def load_images_as_tensor(self, folder_path):
        """
        폴더에 있는 이미지들을 텐서로 로드하는 함수
        Args:
            folder_path: 이미지들이 저장된 폴더 경로
        Returns:
            텐서: (B, C, H, W) 형태의 텐서
        """
        image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
        transform = transforms.Compose([transforms.ToTensor()])
        image_tensors = [transform(Image.open(os.path.join(folder_path, img)).convert("RGB")) for img in image_files]
        return torch.stack(image_tensors)

    def apply_augmentation(self, images_tensor, folder_path=None):
        """
        이미지 증강을 적용하고 저장하는 함수
        Args:
            images_tensor: 입력 이미지 텐서
            folder_path: 증강된 이미지를 저장할 폴더 경로
        """
        augmented_img_tensor = self.augment_pipeline(images_tensor)
        if folder_path is not None:
            self.save_tensor_images(augmented_img_tensor, folder_path)
        
        return augmented_img_tensor

    def apply_random_augmentation(self, images_tensor, folder_path=None):
        """
        랜덤 이미지 증강을 적용하고 저장하는 함수
        Args:
            images_tensor: 입력 이미지 텐서
            folder_path: 증강된 이미지를 저장할 폴더 경로
        """
        device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

        augment_random = random.choice(self.augmentations)

        if isinstance(augment_random, H264CompressorNet):
            augment_random.to(device)
            augment_random.load_state_dict(torch.load('checkpoints/model_weights_epoch_20.pth', map_location=device))
            augment_random.eval()
            frames_tensor = self.batch_to_frame(images_tensor)
            attacked_tensor = augment_random(frames_tensor.to(device))
            attacked_tensor = self.frame_to_batch(attacked_tensor)
            print("H264Compression")

        else:
            attacked_tensor = augment_random(images_tensor.cpu())
            print(augment_random)

       
        if folder_path is not None:
            self.save_tensor_images(attacked_tensor, folder_path)
        
        return attacked_tensor.to(device)

    def apply_all_augmentations(self, images_tensor, folder_path="all_attacked"):
        """
        모든 증강을 하나씩 적용하는 함수
        Args:
            images_tensor: 입력 이미지 텐서
            folder_path: 증강된 이미지를 저장할 폴더 경로
        """
        augmented_tensors = []
        device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

        for augment in self.augmentations:
            if isinstance(augment, H264CompressorNet):
                augment.to(device)
                augment.load_state_dict(torch.load('checkpoints/model_weights_epoch_20.pth', map_location=device))
                augment.eval()
                frames_tensor = self.batch_to_frame(images_tensor)
                attacked_tensor = augment(frames_tensor.to(device))
                attacked_tensor = self.frame_to_batch(attacked_tensor)
                print(f"Applied H264CompressorNet augmentation: {augment}")
            else:
                attacked_tensor = augment(images_tensor.cpu())
                print(f"Applied augmentation: {augment}")

            augmented_tensors.append(attacked_tensor)

            # 증강된 이미지를 저장하는 옵션이 있는 경우
            if folder_path is not None:
                aug_folder_path = os.path.join(folder_path, f"{type(augment).__name__}")
                self.save_tensor_images(attacked_tensor, aug_folder_path)
        exit()
        # 텐서를 모두 합쳐 반환
        return torch.stack(augmented_tensors, dim=0)

