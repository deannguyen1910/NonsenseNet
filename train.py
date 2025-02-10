import torch
import torch.optim as optim
import torch.nn as nn
import pickle
import numpy as np
from model import BaseNet
from torch.utils.data import DataLoader, TensorDataset
import subprocess  # To run eval.py
import os

# Select device (CUDA, MPS, or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Load and shuffle all data batches together
def load_all_batches():
    data = []
    labels = []
    for i in range(1, 6):  # Assuming 5 data batches
        batch_path = f"data/cifar-10-batches-py/data_batch_{i}"
        with open(batch_path, 'rb') as file:
            batch = pickle.load(file, encoding='bytes')
            data.append(batch[b'data'])
            labels.append(batch[b'labels'])
    data = np.concatenate(data, axis=0)
    labels = np.concatenate(labels, axis=0)
    indices = np.arange(len(data))
    np.random.shuffle(indices)
    return torch.tensor(data[indices], dtype=torch.float32), torch.tensor(labels[indices], dtype=torch.long)

# Load dataset
train_data, train_labels = load_all_batches()
train_data = train_data.view(-1, 3, 32, 32).to(device)  # Reshape for CNN input
train_labels = train_labels.to(device)

# Set batch size and DataLoader
batch_size = 64
dataset_size = len(train_data)
max_epochs = max(10, dataset_size // batch_size)  # Ensure at least 10 epochs

train_dataset = TensorDataset(train_data, train_labels)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Initialize model
model = BaseNet().to(device)

# Kaiming He Initialization Function
def kaiming_init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):  # Apply only to Conv2d and Linear layers
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)  # Set bias to zero

# model.apply(kaiming_init)  # Apply initialization

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(1, max_epochs + 1):
    model.train()
    epoch_loss = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_train_loss = epoch_loss / len(train_loader)
    print(f"Epoch [{epoch}/{max_epochs}], Training Loss: {avg_train_loss:.4f}")

    # Save model predictions and run evaluation every 10 epochs
    if epoch % 10 == 0:
        pred_file = "data/output.pth"
        torch.save(model.state_dict(), pred_file)  # Save model state
        print(f"Saved model predictions to {pred_file}")

        # Run evaluation
        print("Evaluating test loss...")
        test_loss = subprocess.run(["python", "eval.py", pred_file], capture_output=True, text=True)
        print(f"Test Loss: {test_loss.stdout.strip()}")

print("Training complete!")