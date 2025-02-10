import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class NonsenseNet(nn.Module):
    def __init__(self):
        super(NonsenseNet, self).__init__()
        
        # Convolutional Layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        
        # Pooling Layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully Connected Layers
        self.fc1 = nn.Linear(16 * 5 * 5, 400)
        self.fc2 = nn.Linear(400, 120)
        self.fc3 = nn.Linear(120, 84)
        self.fc4 = nn.Linear(84, 10)  # Final output layer

        # Load weights if file exists
        weight_file = "data/output.pth"  # Change to .pth file
        if os.path.exists(weight_file):
            self.load(weight_file)

    def load(self, file_path):
        """
        Loads the model weights from a .pth file.
        """
        try:
            self.load_state_dict(torch.load(file_path, map_location=torch.device("cpu")))
            print(f"Model weights loaded successfully from {file_path}")
        except Exception as e:
            print(f"Error loading model weights: {e}")

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Conv1 -> ReLU -> Pool
        x = self.pool(F.relu(self.conv2(x)))  # Conv2 -> ReLU -> Pool
        
        x = x.view(-1, 16 * 5 * 5)  # Flattening
        x = F.relu(self.fc1(x))  # FC1 -> ReLU
        x = F.relu(self.fc2(x))  # FC2 -> ReLU
        x = F.relu(self.fc3(x))  # FC3 -> ReLU
        x = self.fc4(x)  # FC4 (Final Output Layer)

        return x