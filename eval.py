import sys
import torch
import torch.nn as nn
from model import NonsenseNet
import pickle
import numpy as np

# Select device
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# Function to load test data
def load_test_data():
    test_path = "data/test_batch"  # Adjust if needed
    with open(test_path, 'rb') as file:
        batch = pickle.load(file, encoding='bytes')
        test_data = torch.tensor(batch[b'data'], dtype=torch.float32)
        test_labels = torch.tensor(batch[b'labels'], dtype=torch.long)
    
    test_data = test_data.view(-1, 3, 32, 32)  # Reshape for CNN input
    return test_data.to(device), test_labels.to(device)

# Function to evaluate the model
def evaluate_model(pred_file):
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

    accuracy = (correct / total) * 100  # Convert to percentage
    print(f"Accuracy: {accuracy:.2f}%")  # Print accuracy as percentage

# Run evaluation if script is called directly
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python eval.py <predictions.pth>")
        sys.exit(1)

    pred_file = sys.argv[1]  # Get predictions file from command line
    evaluate_model(pred_file)