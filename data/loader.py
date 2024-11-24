import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset, Subset
from torchvision import datasets
from .transforms import get_transform
import random
import os
import pandas as pd
from PIL import Image


class DynamicSamplingDataset(Dataset):
    def __init__(self, datasets, sample_size):
        self.datasets = datasets
        self.sample_size = sample_size
        self.total_samples = sample_size * len(datasets)
        self.dataset_lengths = [len(dataset) for dataset in datasets]
        self.current_indices = self._sample_indices()
        
        # Create label mapping for each dataset
        self.label_offsets = self._create_label_offsets()
        
    def _create_label_offsets(self):
        offsets = [0]  # First dataset starts at 0
        current_offset = 0
        
        # Define number of classes for each dataset
        classes_per_dataset = {
            'MNIST': 10,
            'CIFAR10': 10,
            'Caltech101': 101,
            'SVHN': 10,
            'FashionMNIST': 10,
            'KMNIST': 10,
            'STL10': 10,
            'ChestXray': 2,
            'HAM10000': 7
        }
        
        # Calculate offsets for each dataset
        for dataset in self.datasets:
            dataset_name = dataset.__class__.__name__.replace('Dataset', '')
            num_classes = classes_per_dataset.get(dataset_name, 0)
            current_offset += num_classes
            offsets.append(current_offset)
            
        return offsets

    def _sample_indices(self):
        all_indices = []
        for d_idx, dataset_length in enumerate(self.dataset_lengths):
            actual_sample_size = min(self.sample_size, dataset_length)
            if actual_sample_size < dataset_length:
                indices = random.sample(range(dataset_length), actual_sample_size)
            else:
                indices = random.choices(range(dataset_length), k=self.sample_size)
            
            all_indices.extend([(d_idx, s_idx) for s_idx in indices])
        
        random.shuffle(all_indices)
        return all_indices

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        dataset_idx, sample_idx = self.current_indices[idx]
        image, label = self.datasets[dataset_idx][sample_idx]
        
        # Adjust label based on dataset offset
        global_label = label + self.label_offsets[dataset_idx]
        
        return image, global_label

class ImageNetDETDatasetTrain(Dataset):
    def __init__(self, root_path, transform=None):
        self.root_path = root_path
        self.transform = transform
        self.image_paths = []
        
        # ILSVRC2013_train
        train_2013_path = os.path.join(root_path, 'ILSVRC2013_train')
        if os.path.exists(train_2013_path):
            for class_folder in os.listdir(train_2013_path):
                class_path = os.path.join(train_2013_path, class_folder)
                for img_name in os.listdir(class_path):
                    if img_name.endswith('.JPEG'):
                        self.image_paths.append(os.path.join(class_path, img_name))
        
        # Extras and 2014 train
        for folder in os.listdir(root_path):
            if folder != 'ILSVRC2013_train' and not folder.endswith('.tar'):
                folder_path = os.path.join(root_path, folder)
                if os.path.isdir(folder_path):
                    for img_name in os.listdir(folder_path):
                        if img_name.endswith('.JPEG'):
                            self.image_paths.append(os.path.join(folder_path, img_name))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, 0  # Return 0 as label since we're only using it for training

class ChestXrayDataset(Dataset):
    def __init__(self, root_path, transform=None, split='train'):
        self.root_path = root_path
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Load normal images (label 0)
        normal_path = os.path.join(root_path, split, 'NORMAL')
        if os.path.exists(normal_path):
            for img_name in os.listdir(normal_path):
                if img_name.endswith('.jpeg'):
                    self.image_paths.append(os.path.join(normal_path, img_name))
                    self.labels.append(0)

        # Load pneumonia images (label 1)
        pneumonia_path = os.path.join(root_path, split, 'PNEUMONIA')
        if os.path.exists(pneumonia_path):
            for img_name in os.listdir(pneumonia_path):
                if img_name.endswith('.jpeg'):
                    self.image_paths.append(os.path.join(pneumonia_path, img_name))
                    self.labels.append(1)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

class HAM10000Dataset(Dataset):
    def __init__(self, root_path, transform=None, split='train'):
        self.root_path = root_path
        self.transform = transform
        
        # Load the full metadata file first to get all possible classes
        full_metadata = pd.read_csv(os.path.join(root_path, 'HAM10000_metadata.csv'))
        self.dx_to_label = {dx: idx for idx, dx in enumerate(full_metadata['dx'].unique())}
        
        # Load the split-specific metadata file
        metadata_file = f'{split}_metadata.csv'
        self.metadata = pd.read_csv(os.path.join(root_path, metadata_file))
        
        # Setup image paths and corresponding labels
        self.image_paths = []
        self.labels = []
        
        images_dir = os.path.join(root_path, f'{split}_images')
        for _, row in self.metadata.iterrows():
            img_path = os.path.join(images_dir, f"{row['image_id']}.jpg")
            if os.path.exists(img_path):
                self.image_paths.append(img_path)
                self.labels.append(self.dx_to_label[row['dx']])

    def __len__(self):
        return len(self.image_paths)
    
    def get_label_names(self):
        return list(self.dx_to_label.keys())

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def load_train_datasets_autoencoder(sample_size=3000, batch_size=64, path='./datasets', imagenet_det_path=None, chest_xray_path=None, ham10000_path=None):
    transform = get_transform(train=True)

    # Load datasets
    datasets_list = [
        datasets.MNIST(root=path, train=True,
                       download=True, transform=transform),
        datasets.CIFAR10(root=path, train=True,
                         download=True, transform=transform),
        datasets.Caltech101(root=path, download=True, transform=transform),
        datasets.SVHN(root=path, split='train',
                      download=True, transform=transform),
        datasets.FashionMNIST(root=path, train=True,
                              download=True, transform=transform),
        datasets.KMNIST(root=path, train=True,
                        download=True, transform=transform),
        datasets.STL10(root=path, split='train',
                       download=True, transform=transform)
    ]

    # Add ImageNet DET
    if imagenet_det_path:
        imagenet_det = ImageNetDETDatasetTrain(
            root_path=imagenet_det_path,
            transform=transform
        )
        datasets_list.append(imagenet_det)
    else:
        path = "D:/BigData/datasets/ILSVRC2017_DET/ILSVRC/Data/DET/train"
        # Check if the path exists
        if os.path.exists(path):
            imagenet_det = ImageNetDETDatasetTrain(
                root_path=path,
            transform=transform
            )
            datasets_list.append(imagenet_det)

    # Add Chest X-ray
    if chest_xray_path:
        chest_xray = ChestXrayDataset(
            root_path=chest_xray_path,
            transform=transform,
            split='train'
        )
        datasets_list.append(chest_xray)
    else:
        path = "D:/Learnable_Activation/datasets/chest_xray"
        if os.path.exists(path):
            chest_xray = ChestXrayDataset(
                root_path=path,
                transform=transform,
                split='train'
            )
            datasets_list.append(chest_xray)
    
    # Add HAM10000
    if ham10000_path:
        ham10000 = HAM10000Dataset(
            root_path=ham10000_path,
            transform=transform,
            split='train'
        )
        datasets_list.append(ham10000)
    else:
        path = 'D:/Learnable_Activation/datasets/HAM10000'
        if os.path.exists(path):
            ham10000 = HAM10000Dataset(
                root_path=path,
                transform=transform,
                split='train'
            )
            datasets_list.append(ham10000)

    # Create dynamic sampling dataset
    dynamic_dataset = DynamicSamplingDataset(datasets_list, sample_size)
    return DataLoader(dynamic_dataset, batch_size=batch_size, shuffle=True)

def load_test_datasets_autoencoder(batch_size=100, path='./datasets'):
    transform = get_transform(train=False)
    chest_xray_path = "D:/Learnable_Activation/datasets/chest_xray"
    ham10000_path = "D:/Learnable_Activation/datasets/HAM10000"
    
    # Load test datasets
    test_datasets = {
        'MNIST': datasets.MNIST(root=path, train=False, download=True, transform=transform),
        'CIFAR10': datasets.CIFAR10(root=path, train=False, download=True, transform=transform),
        'Caltech101': datasets.Caltech101(root=path, download=True, transform=transform),
        'SVHN': datasets.SVHN(root=path, split='test', download=True, transform=transform),
        'FashionMNIST': datasets.FashionMNIST(root=path, train=False, download=True, transform=transform),
        'KMNIST': datasets.KMNIST(root=path, train=False, download=True, transform=transform),
        'STL10': datasets.STL10(root=path, split='test', download=True, transform=transform),
        "ChestXray": ChestXrayDataset(root_path=chest_xray_path, split='test', transform=transform),
        "HAM10000": HAM10000Dataset(root_path=ham10000_path, split='test', transform=transform),
    }

    # Create dataloaders
    test_loaders = {
        name: DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for name, dataset in test_datasets.items()
    }

    return test_loaders

DATASET_OFFSETS = {
    'MNIST': {
        'start': 0,
        'classes': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    },
    'CIFAR10': {
        'start': 10,
        'classes': ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    },
    'Caltech101': {
        'start': 20,
        'classes': [
        'accordion', 'airplanes', 'anchor', 'ant', 'background_google',
        'barrel', 'bass', 'beaver', 'binocular', 'bonsai',
        'brain', 'brontosaurus', 'buddha', 'butterfly', 'camera',
        'cannon', 'car_side', 'ceiling_fan', 'cellphone', 'chair',
        'chandelier', 'cougar_body', 'cougar_face', 'crab', 'crayfish',
        'crocodile', 'crocodile_head', 'cup', 'dalmatian', 'dollar_bill',
        'dolphin', 'dragonfly', 'electric_guitar', 'elephant', 'emu',
        'euphonium', 'ewer', 'faces', 'faces_easy', 'ferry',
        'flamingo', 'flamingo_head', 'garfield', 'gerenuk', 'gramophone',
        'grand_piano', 'hawksbill', 'headphone', 'hedgehog', 'helicopter',
        'ibis', 'inline_skate', 'joshua_tree', 'kangaroo', 'ketch',
        'lamp', 'laptop', 'leopards', 'llama', 'lobster',
        'lotus', 'mandolin', 'mayfly', 'menorah', 'metronome',
        'minaret', 'motorbikes', 'nautilus', 'octopus', 'okapi',
        'pagoda', 'panda', 'pigeon', 'pizza', 'platypus',
        'pyramid', 'revolver', 'rhino', 'rooster', 'saxophone',
        'schooner', 'scissors', 'scorpion', 'sea_horse', 'snoopy',
        'soccer_ball', 'stapler', 'starfish', 'stegosaurus', 'stop_sign',
        'strawberry', 'sunflower', 'tick', 'trilobite', 'umbrella',
        'watch', 'water_lilly', 'wheelchair', 'wild_cat', 'windsor_chair',
        'wrench', 'yin_yang'
    ]
    },
    'SVHN': {
        'start': 121,
        'classes': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    },
    'FashionMNIST': {
        'start': 131,
        'classes': ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    },
    'KMNIST': {
        'start': 141,
        'classes': ['お', 'き', 'す', 'つ', 'な', 'は', 'ま', 'や', 'れ', 'を']
    },
    'STL10': {
        'start': 151,
        'classes': ['airplane', 'bird', 'car', 'cat', 'deer', 
                   'dog', 'horse', 'monkey', 'ship', 'truck']
    },
    'ChestXray': {
        'start': 161,
        'classes': ['NORMAL', 'PNEUMONIA']
    },
    'HAM10000': {
        'start': 163,
        'classes': ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
    }
}
