import numpy as np
from PIL import Image
import torchvision.transforms as transforms

class RGBConverter:
    def __call__(self, img):
        if img.mode != 'RGB':
            return img.convert('RGB')
        return img

def get_transform(train=True):
    input_mean = [0.48145466, 0.45578275, 0.40821073]
    input_std = [0.26862954, 0.26130258, 0.27508549]
    input_size = 32

    if train:
        return transforms.Compose([
            RGBConverter(),
            transforms.RandomAffine(degrees=0, translate=(0.3, 0.3), scale=(0.5, 1.5), shear=10),
            transforms.RandomResizedCrop(input_size, scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomRotation(degrees=10),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.RandomSolarize(threshold=128.0, p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=input_mean, std=input_std)
        ])
    else:
        return transforms.Compose([
            RGBConverter(),
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=input_mean, std=input_std)
        ])
