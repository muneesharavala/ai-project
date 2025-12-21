import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from preprocessing import load_data

# 🔹 Config
DATA_DIR = "data"
EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🔹 Load data
train_loader, test_loader, classes = load_data(DATA_DIR, BATCH_SIZE)

# 🔹 Define Model (Transfer Learning - ResNet18)
model = models.resnet18(weights='IMAGENET1K_V1')
model.fc = nn.Linear(model.fc.in_features, len(classes))  # output for class count
model = model.to(DEVICE)

# 🔹 Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 🔹 Training Loop
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {running_loss/len(train_loader):.4f}")

# 🔹 Save Model
torch.save(model.state_dict(), "models/disease_detector.pt")
print("✅ Model training complete and saved!")
