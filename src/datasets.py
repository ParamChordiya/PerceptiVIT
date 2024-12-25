import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class RealFakeDataset(Dataset):
    def __init__(self, csv_path, max_samples=100, transform=None):
        """
        Args:
            csv_path (str): Path to CSV file (train.csv, valid.csv, or test.csv).
            max_samples (int): The maximum number of samples to keep from this dataset.
            transform (torchvision.transforms.Compose): Transform to apply on the image.
        """
        data_df = pd.read_csv(csv_path)
        if len(data_df) > max_samples:
            data_df = data_df[:max_samples]
        self.data_df = data_df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        row = self.data_df.iloc[idx]
        img_path = row['path']
        label = row['label']  # 0 => fake, 1 => real
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


def get_dataloaders(train_csv, valid_csv, test_csv, 
                    batch_size=8,   # smaller batch size for quick tests
                    num_workers=0,  # set to 0 to avoid potential deadlocks
                    img_size=256,
                    max_samples=100):
    """
    Returns train, valid, and test DataLoaders.
    Each dataset is restricted to the first `max_samples` rows from its CSV.
    """
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_dataset = RealFakeDataset(train_csv, max_samples=max_samples, transform=train_transform)
    valid_dataset = RealFakeDataset(valid_csv, max_samples=max_samples, transform=val_test_transform)
    test_dataset  = RealFakeDataset(test_csv,  max_samples=max_samples, transform=val_test_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers
    )

    return train_loader, valid_loader, test_loader
