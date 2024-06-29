#model.py
import torch
import torch.nn as nn
import torchvision.models as models
from efficientnet_pytorch import EfficientNet

class ImageModel(nn.Module):
    def __init__(self, model_name):
        super(ImageModel, self).__init__()
        if model_name == 'resnet18':
            self.model = models.resnet18(pretrained=True)
            self.model.fc = nn.Identity()
            self.output_dim = 512
        elif model_name == 'efficientnetb0':
            self.model = EfficientNet.from_pretrained('efficientnet-b0')
            self.model._fc = nn.Identity() 
            self.output_dim = 1280
            # self.head = nn.Linear(self.output_dim, 2)
    def forward(self, x):
        # with torch.no_grad():
        image_embeddings = self.model(x)
        
        return image_embeddings

class PartModel(nn.Module):
    def __init__(self, model_name, input_dim):
        super(PartModel, self).__init__()
        if model_name == 'lstm':
            self.model = nn.LSTM(input_dim, 64, batch_first=True)
            self.output_dim = 64
        elif model_name == 'linear':
            self.model = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 64)
            )
            self.output_dim = 64

    def forward(self, x):
        if isinstance(self.model, nn.LSTM):
            x, _ = self.model(x)
            x = x[:, -1, :]
        else:
            x = self.model(x)
        return x

class TransformerClassifier(nn.Module):
    def __init__(self, image_output_dim, part_output_dim, transformer_dim=256):
        super(TransformerClassifier, self).__init__()
        self.linear_layer = nn.Linear(image_output_dim + part_output_dim, transformer_dim)
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=transformer_dim, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=1)
        self.fc = nn.Linear(transformer_dim, 2)

    def forward(self, image_embedding, part_embedding):
        combined_embedding = torch.cat((image_embedding, part_embedding), dim=1)
        reduced_embedding = self.linear_layer(combined_embedding)
        transformer_output = self.transformer_encoder(reduced_embedding.unsqueeze(1))
        transformer_output = transformer_output.squeeze(1)
        output = self.fc(transformer_output)
        return output