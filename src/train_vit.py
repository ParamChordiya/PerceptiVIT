# src/train_vit.py
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from datasets import get_dataloaders
from model_vit import VisionTransformer

def train_one_epoch(model, device, dataloader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    all_labels = []
    all_preds = []

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc

def validate_one_epoch(model, device, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)

            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(probs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc, all_labels, all_preds, all_probs

def plot_curves(train_loss, val_loss, train_acc, val_acc, title='VisionTransformer'):
    epochs = range(1, len(train_loss)+1)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    # Loss curve
    axs[0].plot(epochs, train_loss, 'b-o', label='Train Loss')
    axs[0].plot(epochs, val_loss, 'r-o', label='Val Loss')
    axs[0].set_title(f'{title} - Loss')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].legend()

    # Accuracy curve
    axs[1].plot(epochs, train_acc, 'b-o', label='Train Acc')
    axs[1].plot(epochs, val_acc, 'r-o', label='Val Acc')
    axs[1].set_title(f'{title} - Accuracy')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend()

    plt.tight_layout()
    plt.show()

def plot_roc(labels, probs, title='ROC Curve'):
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='AUC = %0.2f' % roc_auc)
    plt.plot([0,1], [0,1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.show()

def plot_confusion(labels, preds, class_names=['Fake','Real'], title='Confusion Matrix'):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

def main():
    # CSV paths
    train_csv = 'data/train.csv'
    valid_csv = 'data/valid.csv'
    test_csv  = 'data/test.csv'

    # Hyperparams
    batch_size = 32
    lr = 1e-4
    epochs = 100
    img_size = 256
    max_samples = 1000

    # Dataloaders
    train_loader, _, _ = get_dataloaders(
        train_csv, valid_csv, test_csv, 
        batch_size=batch_size, 
        num_workers=0,    # 0 to avoid potential concurrency issues
        img_size=img_size,
        max_samples=100000
    )

    _, valid_loader, _ = get_dataloaders(
        train_csv, valid_csv, test_csv, 
        batch_size=batch_size, 
        num_workers=0,    # 0 to avoid potential concurrency issues
        img_size=img_size,
        max_samples=20000
    )

    _, _, test_loader = get_dataloaders(
        train_csv, valid_csv, test_csv, 
        batch_size=batch_size, 
        num_workers=0,    # 0 to avoid potential concurrency issues
        img_size=img_size,
        max_samples=20000
    )

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize Vision Transformer
    model = VisionTransformer(
        img_size=img_size,
        patch_size=16,
        in_channels=3,
        num_classes=2,
        embed_dim=192,    # smaller embedding
        depth=4,          # fewer blocks
        num_heads=4,      # fewer heads
        mlp_ratio=4.0,
        dropout=0.1
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    # Lists for plotting
    train_loss_list, val_loss_list = [], []
    train_acc_list, val_acc_list = [], []

    # Training loop
    for epoch in range(epochs):
        # print(f"\n=== EPOCH {epoch+1}/{epochs} ===")
        train_loss, train_acc = train_one_epoch(model, device, train_loader, criterion, optimizer)
        val_loss, val_acc, val_labels, val_preds, val_probs = validate_one_epoch(model, device, valid_loader, criterion)

        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)

        print(f"[EPOCH {epoch+1}] "
              f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f} | "
              f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")


    os.makedirs("plots", exist_ok=True)

    plot_curves(train_loss_list, val_loss_list, train_acc_list, val_acc_list, title='CustomViT')
    plt.savefig("plots/train_val_curves.png")
    plt.close()

    test_loss, test_acc, test_labels, test_preds, test_probs = validate_one_epoch(model, device, test_loader, criterion)
    print(f"\nTEST => Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")

    plot_roc(test_labels, test_probs, title='CustomViT - ROC')
    plt.savefig("plots/roc_curve.png") 
    plt.close()

    plot_confusion(test_labels, test_preds, ['Fake', 'Real'], title='CustomViT- Confusion Matrix')
    plt.savefig("plots/confusion_matrix.png") 
    plt.close()

    torch.save(model.state_dict(), "custom_vit_model2.pth")
    print("Model saved to custom_vit_model2.pth")

if __name__ == "__main__":
    main()
