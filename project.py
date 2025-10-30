import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Define transformations for the training and validation sets
transform = transforms.Compose([
    transforms.Resize((224, 224)),        # Resize the image to 224x224 for ResNet input
    transforms.ToTensor(),                # Convert the image to a tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

# Load the dataset
data_dir = 'path_to_your_dataset'  # Replace with the path to your dataset directory
dataset = datasets.ImageFolder(root=data_dir, transform=transform)

# Split dataset into train and validation sets
train_size = int(0.8 * len(dataset))  # 80% training data
val_size = len(dataset) - train_size  # 20% validation data
train_data, val_data = random_split(dataset, [train_size, val_size])

# DataLoader for batching
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)


# Load a pre-trained ResNet model and modify the final layer
model = models.resnet18(pretrained=True)

# Freeze the weights of the pre-trained layers
for param in model.parameters():
    param.requires_grad = False

# Modify the fully connected layer to match the number of classes in your dataset
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)  # Assuming binary classification (early-stage disease vs healthy)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)  # Only optimize the final layer


num_epochs = 10  # Set the number of epochs for training

# Training Loop
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")


model.eval()  # Set the model to evaluation mode
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Print classification report and confusion matrix
print(classification_report(all_labels, all_preds))
conf_matrix = confusion_matrix(all_labels, all_preds)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Healthy', 'Disease'], yticklabels=['Healthy', 'Disease'])
plt.show()


torch.save(model.state_dict(), 'early_disease_detection_model.pth')
print("Model saved as early_disease_detection_model.pth")# Save the trained model
