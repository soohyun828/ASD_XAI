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
    def __init__(self, input_dim):
        super(PartModel, self).__init__()
        self.for_shap = nn.Identity()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        self.output_dim = 64

    def forward(self, x):
        x1 = self.for_shap(x)
        x2 = self.model(x1)
        return x2

class TransformerClassifier(nn.Module):
    def __init__(self, img_model_name, image_output_dim, part_output_dim, transformer_dim=256):
        super(TransformerClassifier, self).__init__()
        self.linear_layer = nn.Linear(image_output_dim + part_output_dim, transformer_dim)
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=transformer_dim, nhead=8, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=1)
        self.fc = nn.Linear(transformer_dim, 2)
        self.image_encoder = ImageModel(img_model_name)
        self.ratio_encoder = PartModel(16)
        self.gradients = []
        self.activations = []
        self.handles_list = []
        self.pruned_activations_mask = []
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def forward(self, x, csv):
        image_embeddings = self.image_encoder(x)
        csv_embeddings = self.ratio_encoder(csv)
        combined_embedding = torch.cat((image_embeddings, csv_embeddings), dim=1)
        reduced_embedding = self.linear_layer(combined_embedding)
        transformer_output = self.transformer_encoder(reduced_embedding.unsqueeze(1))
        transformer_output = transformer_output.squeeze(1)
        output = self.fc(transformer_output)
        return output


    def _forward(self, x,csv):
        self.activations = []
        self.gradients = []
        self.zero_grad()
        image_embeddings = self.image_encoder(x)
        csv_embeddings = self.ratio_encoder(csv)
        combined_embedding = torch.cat((image_embeddings, csv_embeddings), dim=1)
        reduced_embedding = self.linear_layer(combined_embedding)
        transformer_output = self.transformer_encoder(reduced_embedding.unsqueeze(1))
        transformer_output = transformer_output.squeeze(1)
        output = self.fc(transformer_output)

        return output
    def remove_handles(self):
        for handle in self.handles_list:
            handle.remove()
        self.handles_list.clear()
        self.activations = []
        self.gradients = []

    def _compute_taylor_scores(self, inputs,csv, labels,target_layer):
        self._hook_layers(target_layer)
        csv.requires_grad_(True)
        outputs = self._forward(inputs,csv)
        outputs[0, labels.item()].backward(retain_graph=True)
        
        first_order_taylor_scores = []
        self.gradients.reverse()
        self.gradients.append(csv.grad)
        for i, layer in enumerate(self.activations):
            first_order_taylor_scores.append(torch.mul(layer, self.gradients[i]))
                
        self.remove_handles()
        return first_order_taylor_scores, outputs

    def _hook_layers(self,target_layer):
        def backward_hook_relu(module, grad_input, grad_output):
            self.gradients.append(grad_output[0].to(self.device))

        def forward_hook_relu(module, input, output):
            # mask output by pruned_activations_mask
            # In the first model(input) call, the pruned_activations_mask
            # is not yet defined, thus we check for emptiness
            if self.pruned_activations_mask:
              output = torch.mul(output, self.pruned_activations_mask[len(self.activations)].to(self.device)) #+ self.pruning_biases[len(self.activations)].to(self.device)
            self.activations.append(output.to(self.device))
            return output

        for name,module in self.named_modules():
            if target_layer=='layer4':
                if isinstance(module, nn.AdaptiveAvgPool2d):
                    self.handles_list.append(module.register_forward_hook(forward_hook_relu))
                    self.handles_list.append(module.register_backward_hook(backward_hook_relu))
            elif target_layer=='head':
                if name=='fc':
                    self.handles_list.append(module.register_forward_hook(forward_hook_relu))
                    self.handles_list.append(module.register_backward_hook(backward_hook_relu))
            elif target_layer =='ratio':
                if 'for_shap' in name:
                    self.handles_list.append(module.register_forward_hook(forward_hook_relu))
                    self.handles_list.append(module.register_backward_hook(backward_hook_relu))
            else:
                print('error');exit(0)