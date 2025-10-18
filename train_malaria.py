import torch, torchvision
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import time
import os

# GPU check
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Config
DATA_PATH = r"D:\MalariaDetection\cell_images"  # Update if needed
BATCH_SIZE = 32
IMG_SIZE = 224
EPOCHS = 20
LEARNING_RATE = 1e-3
SAVE_MODEL_PATH = r"D:\MalariaDetection\model\best_blood_model.pth"

# Dataset
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])
full_data = datasets.ImageFolder(root=DATA_PATH, transform=transform)
train_size = int(0.8*len(full_data))
val_size = len(full_data) - train_size
train_data, val_data = random_split(full_data, [train_size, val_size])
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

# Model
model = models.resnet18(pretrained=True)
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, len(full_data.classes))
model = model.to(device)

# Loss & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training
best_val_acc = 0
start_time = time.time()
for epoch in range(1, EPOCHS+1):
    model.train()
    running_correct = 0
    total = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_correct += (outputs.argmax(1)==labels).sum().item()
        total += labels.size(0)
    train_acc = 100*running_correct/total

    # Validation
    model.eval()
    val_correct, total_val = 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            val_correct += (outputs.argmax(1)==labels).sum().item()
            total_val += labels.size(0)
    val_acc = 100*val_correct/total_val
    print(f"Epoch {epoch}/{EPOCHS} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

    if val_acc>best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), SAVE_MODEL_PATH)

total_time = time.time()-start_time
print(f"Training finished in {total_time/60:.2f} min. Best Val Acc: {best_val_acc:.2f}%")
