import torch
import torch.nn as nn
import torch.optim as optim
from model import ImprovedSelfPruningNN
from utils import sparsity_loss

def train_model(train_loader, lambda_val, device, epochs=30):
    
    model = ImprovedSelfPruningNN().to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)

            ce_loss = criterion(outputs, labels)
            sp_loss = sparsity_loss(model)

            loss = ce_loss + lambda_val * sp_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    return model
