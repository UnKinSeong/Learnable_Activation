import numpy as np
from PIL import Image
import torchvision.transforms as transforms


class RandomColorTransform:
    def __call__(self, img):
        img_array = np.array(img)

        if len(img_array.shape) == 2:
            color_shift = np.random.randint(0, 50)
            img_array = np.clip(img_array + color_shift, 0, 255)
        else:
            color_shift = np.random.randint(0, 50, size=3)
            img_array = np.clip(
                img_array + color_shift.reshape(1, 1, 3), 0, 255)

        return Image.fromarray(img_array.astype('uint8'))


def get_transform(train=True):
    if train:
        return transforms.Compose([
            RandomColorTransform(),
            transforms.ToTensor(),
            transforms.Resize((32, 32)),
            transforms.Lambda(lambda x: x.repeat(
                3, 1, 1) if x.size(0) == 1 else x)
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((32, 32)),
            transforms.Lambda(lambda x: x.repeat(
                3, 1, 1) if x.size(0) == 1 else x)
        ])
