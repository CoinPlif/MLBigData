import pandas as pd
from tqdm import tqdm 
from PIL import Image
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader

import boto3
from botocore.client import Config

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    total_samples = 0
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Validation'):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            batch_size = images.size(0)
            total_samples += batch_size
            running_loss += loss.item() * batch_size

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / total_samples
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds)
    epoch_precision = precision_score(all_labels, all_preds)
    epoch_recall = recall_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc, epoch_f1, epoch_precision, epoch_recall


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    total_samples = 0
    
    for images, labels in tqdm(dataloader, desc='Training'):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        batch_size = images.size(0)
        total_samples += batch_size
        running_loss += loss.item() * batch_size

        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / total_samples
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc, epoch_f1


def setup_model(cfg):
    model = models.resnet18(pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg['learning_rate'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    return model, criterion, optimizer, scheduler


def setup_loaders(cfg):
    train_dir = cfg['train_dir']
    test_dir = cfg['test_dir']
    batch_size = cfg['batch_size']

    class ImageDataset(Dataset):
        def __init__(self, csv_file, root_dir, transform=None):
            self.data = pd.read_csv(csv_file)
            self.root_dir = Path(root_dir)
            self.transform = transform
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            img_path = self.root_dir / self.data.iloc[idx]['file_name']
            image = Image.open(img_path).convert('RGB')
            label = self.data.iloc[idx]['label']
            
            if self.transform:
                image = self.transform(image)
            
            return image, label

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = ImageDataset(
        csv_file=f'{train_dir}/train.csv',
        root_dir=f'{train_dir}',
        transform=train_transform
    )

    test_dataset = ImageDataset(
        csv_file=f'{test_dir}/test.csv',
        root_dir=f'{test_dir}',
        transform=test_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, test_loader

def upload_to_minio(file_path, bucket_name, object_name):
    s3 = boto3.client(
        's3',
        endpoint_url='http://localhost:6062',
        aws_access_key_id='minioadmin',
        aws_secret_access_key='minioadmin',
        region_name='us-east-1',
        config=Config(signature_version='s3v4')
    )
    try:
        s3.head_bucket(Bucket=bucket_name)
    except Exception:
        s3.create_bucket(Bucket=bucket_name)

    s3.upload_file(file_path, bucket_name, object_name)
    print(f"Загружено: s3://{bucket_name}/{object_name}")