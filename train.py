import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import pandas as pd
from torchvision import transforms
from dataset import HumanFigureDataset
from models import ImageModel, PartModel, TransformerClassifier

def train(data_root, csv_path, epochs, batch_size, learning_rate, image_model_name, part_model_name, validation_split=0.2):
    
    if image_model_name == 'resnet18':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.98, 0.98, 0.98],
                            std=[0.065, 0.065, 0.065])
        ])
    elif image_model_name == 'efficientnetb0':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.98, 0.98, 0.98],
                            std=[0.065, 0.065, 0.065])
        ])
    
    dataset = HumanFigureDataset(data_root, csv_path, transform=transform)
    
    val_size = int(validation_split * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_model = ImageModel(image_model_name).to(device)
    csv_data = pd.read_csv(csv_path)
    csv_data = csv_data.iloc[1:, :]
    csv_input_dim = csv_data.shape[1] - 2
    csv_model = PartModel(part_model_name, csv_input_dim).to(device)
    transformer_classifier = TransformerClassifier(image_model.output_dim, csv_model.output_dim).to(device)

    #####
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(transformer_classifier.parameters(), lr=learning_rate,weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    for epoch in range(epochs):
        transformer_classifier.train()
        training_loss = 0.0
        step = 0
        for images, csv_data, labels in train_loader:
            images, csv_data, labels = images.to(device), csv_data.to(device), labels.to(device)
            
            csv_data = csv_data.float()
            
            # outputs = image_model(images)
            image_embeddings = image_model(images)
            csv_embeddings = csv_model(csv_data)
            outputs = transformer_classifier(image_embeddings, csv_embeddings)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            print(f'Batch {step}, Loss: {loss.item():.4f}')
            training_loss += loss.item() * images.size(0)
        scheduler.step()
        epoch_loss = training_loss / train_size
        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}')
        
        transformer_classifier.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, csv_data, labels in val_loader:
                images, csv_data, labels = images.to(device), csv_data.to(device), labels.to(device)
                
                csv_data = csv_data.float()
                
                image_embeddings = image_model(images)
                csv_embeddings = csv_model(csv_data)
                outputs = transformer_classifier(image_embeddings, csv_embeddings)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss /= val_size
        accuracy = 100 * correct / total
        print(f'Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%')
