import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.device_count())
print("device is:", device)


from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# Define a transform to normalize the data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


# Download and load the training data
trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)


# Download and load the test data
testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=64, shuffle=False)



import torch.nn as nn
import torch.nn.functional as F


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)


    def forward(self, x):
        x = x.view(x.shape[0], -1)  # Flatten the input
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = SimpleNet()
model.to(device)


# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


# Training loop
epochs = 5
for epoch in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        # Move input and label tensors to the GPU
        images, labels = images.to(device), labels.to(device)


        # Zero the gradients
        optimizer.zero_grad()


        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)


        # Backward pass and optimize
        loss.backward()
        optimizer.step()


        running_loss += loss.item()


    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(trainloader)}")


correct = 0
total = 0
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()


print(f'Accuracy of the network on the 10000 test images: {100 * correct / total}%')
