import os
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
from torchvision import transforms

# Data Preparation
DATASET_PATH = "/kaggle/input/celebahq-resized-256x256/celeba_hq_256"
all_files = [os.path.join(DATASET_PATH, file) for file in os.listdir(DATASET_PATH) if file.endswith(".jpg")]
train_files, test_files = train_test_split(all_files, test_size=0.2, random_state=42)

def corrupt_image(image):
    image_np = np.array(image)
    noise = np.random.normal(0, 25, image_np.shape).astype(np.float32)
    noisy_image = np.clip(image_np + noise, 0, 255).astype(np.uint8)
    blurred_image = cv2.GaussianBlur(noisy_image, (5, 5), 1)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 30]
    _, compressed_image = cv2.imencode('.jpg', blurred_image, encode_param)
    decompressed_image = cv2.imdecode(compressed_image, cv2.IMREAD_COLOR)
    return Image.fromarray(decompressed_image)

transform_lr = transforms.Compose([
    transforms.Resize((32, 32)), 
    transforms.Lambda(corrupt_image),
    transforms.ToTensor(),
])
transform_hr = transforms.Compose([
    transforms.Resize((128, 128)), 
    transforms.ToTensor(),
])

class CelebADataset(Dataset):
    def __init__(self, file_paths, transform_lr, transform_hr):
        self.file_paths = file_paths
        self.transform_lr = transform_lr
        self.transform_hr = transform_hr

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img = Image.open(self.file_paths[idx]).convert("RGB")
        return self.transform_lr(img), self.transform_hr(img)

train_dataset = CelebADataset(train_files, transform_lr, transform_hr)
test_dataset = CelebADataset(test_files, transform_lr, transform_hr)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)