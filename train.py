import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from dataset import HumanFigureDataset
from model import CombinedModel

def train_model(base_dir, num_epochs=10, batch_size=32, learning_rate=0.0001):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.98, 0.98, 0.98],
                            std=[0.065, 0.065, 0.065]),
    ])
    
    dataset = HumanFigureDataset(base_dir=base_dir)
    body_parts = dataset.body_parts
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CombinedModel(body_parts=len(body_parts)).to(device)
    #
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, proportions, labels in train_loader:
            images = images.to(device)
            proportions = proportions.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images, proportions)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            print(loss.item())
            # running_loss += loss.item() * images.size(0)
        
        # epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')
        
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, proportions, labels in val_loader:
                images = images.to(device)
                proportions = proportions.to(device)
                labels = labels.to(device)
                
                outputs = model(images, proportions)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader.dataset)
        accuracy = 100 * correct / total
        print(f'Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%')
