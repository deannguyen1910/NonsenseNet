import sys
import torch
import torch.nn as nn
from model import NonsenseNet
import pickle
import numpy as np
import matplotlib.pyplot as plt
import random

# Select device
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# Function to load test data
def load_test_data():
    test_path = "data/cifar-10-batches-py/test_batch"  # Adjust if needed
    with open(test_path, 'rb') as file:
        batch = pickle.load(file, encoding='bytes')
        test_data = torch.tensor(batch[b'data'], dtype=torch.float32)
        test_labels = torch.tensor(batch[b'labels'], dtype=torch.long)
    
    test_data = test_data.view(-1, 3, 32, 32) / 255.0  # Rescale pixel values
    return test_data.to(device), test_labels.to(device)

# Function to visualize a random subset of test cases
def visualize_test_cases(test_data, test_labels, predicted_labels, num_samples=5):
    indices = random.sample(range(len(test_data)), num_samples)
    fig, axes = plt.subplots(1, num_samples, figsize=(10, 2))
    for i, idx in enumerate(indices):
        image = test_data[idx].cpu().numpy().transpose((1, 2, 0))
        actual_label = test_labels[idx].item()
        predicted_label = predicted_labels[idx].item()
        
        axes[i].imshow(image)
        axes[i].set_title(f"Pred: {predicted_label}\nActual: {actual_label}")
        axes[i].axis("off")
    plt.show()

# Function to evaluate the model
def evaluate_model(pred_file="data/output.pth", num_samples=5):
    # Load model and weights
    model = NonsenseNet().to(device)
    model.load_state_dict(torch.load(pred_file, map_location=device))
    model.eval()  # Set model to evaluation mode

    # Load test data
    test_data, test_labels = load_test_data()

    # Compute accuracy
    with torch.no_grad():  # No gradients needed for evaluation
        outputs = model(test_data)
        _, predicted = torch.max(outputs, 1)  # Get the class with the highest probability
        correct = (predicted == test_labels).sum().item()
        total = test_labels.size(0)

    print(f"Correct Predictions: {correct} / {total}")  # Show correct out of total
    print(f"Accuracy: {correct / total * 100:.2f}%")  # Also print percentage accuracy

    # Visualize random test cases
    visualize_test_cases(test_data, test_labels, predicted, num_samples)

# Run evaluation if script is called directly
if __name__ == "__main__":
    pred_file = "data/output.pth" if len(sys.argv) < 2 else sys.argv[1]  # Default to data/output.pth
    num_samples = 5 if len(sys.argv) < 3 else int(sys.argv[2])  # Default number of samples to 5
    evaluate_model(pred_file, num_samples)
