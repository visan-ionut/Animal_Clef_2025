from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from transformers import get_scheduler
import timm
import csv
import torch.nn as nn
import torch
import torch.optim as optim
import pandas as pd
import os
import torchvision.transforms as T
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AnimalDataset(Dataset):
    def __init__(self, csv_file, root_dir, label_encoder, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.label_encoder = label_encoder

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Build the full path to the image
        img_path = os.path.join(self.root_dir, self.data.iloc[idx]['path'])

        # Open the image
        image = Image.open(img_path).convert('RGB')

        # Apply transformations (if any)
        if self.transform:
            image = self.transform(image)

        # Label is the value from "identity"
        label_str = self.data.iloc[idx]['identity']
        label = self.label_encoder[label_str]

        return image, label

def train_one_epoch(model, dataloader, optimizer, scheduler, criterion, device, epoch):
    model.train()
    running_loss = 0.0

    for images, labels in tqdm(dataloader, desc=f"Training (Epoch {epoch+1})"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()

    return running_loss / len(dataloader)

def validate(model, dataloader, criterion, device, epoch):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc=f"Validating (Epoch {epoch+1})"):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    return running_loss / len(dataloader), accuracy

# Root directory where images are located
root = './animal-clef-2025'

# Simple resize for displaying images
transform_display = T.Compose([
    T.Resize([384, 384]),
])

# Transform for training / evaluation (resize + normalization)
transform = T.Compose([
    *transform_display.transforms,   # reuse resize step
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

train_csv = "train_database_metadata.csv"
test_csv = "test_database_metadata.csv"

train_identities = pd.read_csv(train_csv)['identity'].unique()
label_encoder = {identity: idx for idx, identity in enumerate(sorted(train_identities))}

# Datasets
train_dataset = AnimalDataset(
    csv_file=train_csv,
    root_dir=root,
    label_encoder=label_encoder,
    transform=transform
)

test_dataset = AnimalDataset(
    csv_file=test_csv,
    root_dir=root,
    label_encoder=label_encoder,
    transform=transform
)

batch_size = 16

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

model = timm.create_model(
    'vit_large_patch16_384', 
    pretrained=True, 
    num_classes=len(label_encoder)
)
model = model.to(device)

num_epochs = 10
lr = 5e-5
weight_decay = 0.01

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

num_training_steps = len(train_loader) * num_epochs
num_warmup_steps = int(0.1 * num_training_steps) 

scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
)

metrics_csv = "epoch_metrics.csv"

if not os.path.exists(metrics_csv):
    with open(metrics_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_acc", "learning_rate"])

train_losses = []
val_losses = []
val_accuracies = []
learning_rates = []

best_val_loss = float('inf')
best_val_acc = 0.0
best_model_path = "best_model.pth"

for epoch in range(num_epochs):
    train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, criterion, device, epoch)
    val_loss, val_acc = validate(model, test_loader, criterion, device, epoch)

    current_lr = scheduler.get_last_lr()[0]
    learning_rates.append(current_lr)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    print(f"Epoch {epoch+1}/{num_epochs} | "
          f"Train Loss: {train_loss:.4f} | "
          f"Val Loss: {val_loss:.4f} | "
          f"Val Acc: {val_acc:.4f} | "
          f"LR: {current_lr:.8f}")

    with open(metrics_csv, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([epoch+1, train_loss, val_loss, val_acc, current_lr])

    # Save best model if val_loss decreased and val_acc increased
    if val_loss < best_val_loss and val_acc > best_val_acc:
        best_val_loss = val_loss
        best_val_acc = val_acc
        torch.save({
            'model_state_dict': model.state_dict(),
            'label_encoder': label_encoder,
        }, best_model_path)
        print(f"✅ Best model saved at epoch {epoch+1} with Val Loss {val_loss:.4f} and Val Acc {val_acc:.4f}")

# Plot Train and Validation Loss
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.legend()
plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid()
plt.savefig("loss_plot.png")
plt.close()

# Plot Validation Accuracy
plt.plot(val_accuracies, label="Validation Accuracy")
plt.legend()
plt.title("Validation Accuracy Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid()
plt.savefig("accuracy_plot.png")
plt.close()

# Plot Learning Rate over Epochs
plt.plot(learning_rates, label="Learning Rate")
plt.legend()
plt.title("Learning Rate Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Learning Rate")
plt.grid()
plt.savefig("learning_rate_plot.png")
plt.close()
