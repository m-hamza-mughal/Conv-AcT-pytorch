#!/usr/bin/env python
import torch
from torch import nn, Tensor
from torchvision import models
import math
from torchvision.models.vision_transformer import Encoder

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class PatchEmbedding(nn.Module):
    def __init__(self, d_model: int, learnable=False, dropout: float = 0.1, max_len: int = 15):
        super(PatchEmbedding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        frames = max_len+1
        if learnable:
            self.pe = nn.Parameter(torch.randn((1, frames, d_model)))
        else:
            position = torch.arange(frames).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
            pe = torch.zeros(1, frames, d_model)
            pe[0, :, 0::2] = torch.sin(position * div_term)
            pe[0, :, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, n_frames, embedding_dim]
        """
        b, _, _ = x.shape
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pe #[:, :x.size(1), :]
        return self.dropout(x)

class BaselineMLP(nn.Module):
    def __init__(self, d_model: int):
        super(BaselineMLP, self).__init__()
        self.linear = nn.Linear(128*128*3, 1024)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(1024, d_model)
        self.gelu2 = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, n_frames, channels, height, width]
        """
        b, f, _, _, _ = x.shape
        x = x.view(b, f, -1)
        x = self.gelu(self.linear(x))
        x = self.gelu2(self.linear2(x))

        return x

class FeatureExtractor(nn.Module):
    def __init__(self, d_model: int, model_name: str, model_weights: str = 'DEFAULT'): #  ):
        super(FeatureExtractor, self).__init__()
        # assert model_name in ['efficientnet_v2_s', 'efficientnet_v2_m', 'efficientnet_v2_l', 'inception_v3', 'wide_resnet50_2', 'wide_resnet101_2']
        self.model = getattr(models, model_name)(weights=model_weights)
        self.d_model = None
        
        if model_name in ['efficientnet_v2_s', 'efficientnet_v2_m', 'efficientnet_v2_l', 'convnext_base', 'convnext_small', 'convnext_large']:
            # self.d_model = self.model.classifier[-1].in_features
            self.model.classifier[-1] = nn.Linear(in_features=self.model.classifier[-1].in_features, out_features = d_model)
        else:
            # self.d_model = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features=self.model.fc.in_features ,out_features=d_model)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, n_frames, channels, height, width]
        """
        b, f, _, _, _ = x.shape
        x = x.view(b*f, *x.size()[2:])
        x = self.model(x)
        x = x.view(b, f, *x.size()[1:])

        return x


class ConvAcTransformer(nn.Module):
    def __init__(self, 
                 d_model: int,
                 attention_heads: int,
                 num_layers: int,
                 num_classes: int,
                 num_frames: int,
                 drop_p: int,
                 feature_extractor_name: str,
                 learnable_pe: bool = False): 
        super(ConvAcTransformer, self).__init__()
        self.d_model = d_model
        self.feature_extractor_name = feature_extractor_name
        self.attention_heads = attention_heads
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.num_frames = num_frames
        self.learnable_pe = learnable_pe
        self.drop_p = drop_p

        
        self.feature_extract = FeatureExtractor(self.d_model, self.feature_extractor_name, model_weights='DEFAULT')
        self.patch_embed = PatchEmbedding(self.d_model, learnable=self.learnable_pe, max_len=self.num_frames, dropout=self.drop_p)

        # transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model,
        #                                                         nhead=self.attention_heads,
        #                                                         norm_first=True,
        #                                                         activation='gelu')
        # self.transformer_encoder = nn.TransformerEncoder(transformer_encoder_layer,
        #                                                  self.num_layers,
        #                                                  norm=nn.LayerNorm(self.d_model))

        self.transformer_encoder = Encoder(seq_length=self.num_frames+1,
                                           num_layers=self.num_layers,
                                           num_heads=self.attention_heads,
                                           hidden_dim=self.d_model, 
                                           mlp_dim=self.d_model,
                                           dropout=self.drop_p, attention_dropout=self.drop_p)
        
        self.dropout = nn.Dropout(self.drop_p)
        self.classification_head = nn.Linear(self.d_model, self.num_classes)
        

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, n_frames, channels, height, width]
        """
        # extract features from all frames
        x = self.feature_extract(x)

        # apply patch embedding from ViT
        x = self.patch_embed(x)

        # ViT encoder
        x = self.transformer_encoder(x)

        # select first token/classifier token
        x = x[:, 0, :]

        # classification head
        x = self.classification_head(self.dropout(x))

        return x



if __name__ == "__main__":
    # test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_tensor = torch.randn(4, 3, 3, 128, 128, dtype=torch.float32).to(device)
    model = ConvAcTransformer(
        d_model=128,
        attention_heads=2, 
        num_layers=2, 
        num_classes=50, 
        num_frames=3, 
        drop_p=0.1,
        feature_extractor_name='resnet18', 
        learnable_pe=False
    )
    print(model)
    model = model.to(device)
    out = model(test_tensor)
    print(out.size(), next(model.parameters()).device)
    # diff = out.mean().backward()
    print("done")