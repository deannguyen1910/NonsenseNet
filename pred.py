import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from model import NonsenseNet

# Initialize model
model = NonsenseNet()
model.load("data/output.pth")  # Load weights from output.pth
model.eval()

# Define data transformations
data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def predict():
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=data_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)
    
    classes = testset.classes
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    with torch.no_grad():
        for i, (images, labels) in enumerate(testloader):
            images = images.to(device)  # Move to the same device as the model
            
            # Ensure images are correctly shaped (should already be 3x32x32 from dataset)
            if images.dim() == 3:
                images = images.unsqueeze(0)  # Add batch dimension if missing
            
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            print(f'Predicted: {classes[predicted[0]]}, Actual: {classes[labels[0]]}')
            
            if i == 9:  
                break

if __name__ == "__main__":
    predict()
