import torch
import torch.nn as nn
import torchvision.models as models

class CombinedModel(nn.Module):
    def __init__(self, body_parts=16):
        super(CombinedModel, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.fc = nn.Identity()

        self.lstm = nn.LSTM(input_size=body_parts, hidden_size=256, num_layers=1, batch_first=True)
        # self.proportion_fc = nn.Linear(256, 2048)  # 추가된 부분

        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=2048+256, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=2)

        self.fc = nn.Linear(2048+256, 2)
        self.init_weight()
    def init_weight(self):
        nn.init.normal_(self.fc.weight)
    def forward(self, images, proportions):
        with torch.no_grad():
            image_embeddings = self.resnet(images)

        proportions = proportions.unsqueeze(1)
        proportion_embeddings, _ = self.lstm(proportions)
        proportion_embeddings = proportion_embeddings.squeeze(1)
        # proportion_embeddings = self.proportion_fc(proportion_embeddings)  # 추가된 부분

        combined_embeddings = torch.cat((image_embeddings, proportion_embeddings), dim=1) #2*2048 4096
        # combined_embeddings = image_embeddings + proportion_embeddings
        transformer_output = self.transformer_encoder(combined_embeddings.unsqueeze(1))
        transformer_output = transformer_output.squeeze(1)

        output = self.fc(transformer_output)

        return output
