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

# Early stopping class
class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.0005):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0

    def should_stop(self, current_loss):
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience

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

# Transforms for train and validation
train_transform = T.Compose([
    T.RandomResizedCrop(384, scale=(0.8, 1.0)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(degrees=10),
    T.ColorJitter(0.4, 0.4, 0.4, 0.1),
    T.AutoAugment(policy=T.AutoAugmentPolicy.IMAGENET),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    T.RandomErasing(p=0.25, scale=(0.02, 0.33))
])

test_transform = T.Compose([
    T.Resize([384, 384]),
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
    transform=train_transform
)

test_dataset = AnimalDataset(
    csv_file=test_csv,
    root_dir=root,
    label_encoder=label_encoder,
    transform=test_transform
)

batch_size = 8

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

model = timm.create_model(
    'convnext_xlarge.fb_in22k_ft_in1k_384',
    pretrained=False,
    num_classes=10772
)

checkpoint = torch.load("best_model.pth", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

if hasattr(model, 'head') and isinstance(model.head, nn.Linear):
    in_features = model.head.in_features
    model.head = nn.Linear(in_features, len(label_encoder))
elif hasattr(model.head, 'fc'):
    in_features = model.head.fc.in_features
    model.head.fc = nn.Linear(in_features, len(label_encoder))
else:
    raise ValueError("Could not find classification head in model.")

model.to(device)

num_epochs = 30
lr = 8e-5
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
best_model_path = "best_model_animal_clef_2025.pth"

early_stopping = EarlyStopping(patience=8, min_delta=0.0005)

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

    # Early stopping check
    if early_stopping.should_stop(val_loss):
        print("🛑 Early stopping triggered!")
        break

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
