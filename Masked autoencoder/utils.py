import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import timm
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Block

from einops import repeat, rearrange
from einops.layers.torch import Rearrange

# Code is adapted from
# https://github.com/IcarusWizard/MAE/
#


def random_indexes(size : int):
    """
    Returns forward (shuffling) indices and backward (reverting) indices.
    args:
        size: Number of indices
    """
    forward_indexes = np.arange(size)
    np.random.shuffle(forward_indexes)
    backward_indexes = np.argsort(forward_indexes)
    return forward_indexes, backward_indexes


def take_indexes(sequences, indexes):
    return torch.gather(sequences, 0, repeat(indexes, 't b -> t b c', c=sequences.shape[-1]))

class PatchShuffle(torch.nn.Module):
    def __init__(self, ratio) -> None:
        super().__init__()
        self.ratio = ratio

    def forward(self, patches : torch.Tensor):
        T, B, C = patches.shape
        remain_T = int(T * (1 - self.ratio))

        indexes = [random_indexes(T) for _ in range(B)]
        forward_indexes = torch.as_tensor(np.stack([i[0] for i in indexes], axis=-1), dtype=torch.long).to(patches.device)
        backward_indexes = torch.as_tensor(np.stack([i[1] for i in indexes], axis=-1), dtype=torch.long).to(patches.device)

        patches = take_indexes(patches, forward_indexes)
        patches = patches[:remain_T]

        return patches, forward_indexes, backward_indexes


class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, emb_dim, in_channels):
        super().__init__()

        # Returns a patched output of size [batch, embed_dim, n_patch_col, n_patch_row]
        self.patchify = nn.Conv2d(
                in_channels=in_channels,
                out_channels=emb_dim,
                kernel_size=patch_size,
                stride=patch_size,
            )
        
        self.pos_embedding = nn.Parameter(torch.zeros((image_size // patch_size) ** 2, 1, emb_dim))
        self.init_weight()
        
    def init_weight(self):
        nn.init.trunc_normal_(self.pos_embedding, std=.02)
        
    def forward(self, x):
        # Get embedded patches, 
        patches = self.patchify(x)
        
        # t = (h w) = Number of patches, 
        # b = batch size, 
        # c = embedding dimension
        patches = rearrange(patches, 'b c h w -> (h w) b c')
        patches = patches + self.pos_embedding
        
        return patches


class MAE_Encoder(nn.Module):
    def __init__(self, img_size=32, patch_size=4, emb_dim=192, in_channels=1, num_layer=12, num_head=3, mask_ratio=0.75):
        super().__init__()

        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.shuffle = PatchShuffle(mask_ratio)
        
        self.patch_embed = PatchEmbedding(img_size, patch_size, emb_dim, in_channels)
        
        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])
        self.layer_norm = torch.nn.LayerNorm(emb_dim)

        self.init_weight()
        
    def init_weight(self):
        nn.init.trunc_normal_(self.cls_token, std=.02)
        
    def forward(self, x):
        # Get embedded patches, and use shuffling to mask patches.
        patches = self.patch_embed(x)
        patches, forward_indexes, backward_indexes = self.shuffle(patches)
        print(patches.shape)
        print(forward_indexes.shape)
        print(backward_indexes.shape)

        # Expand cls_token for each img in batch
        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        
        # Batch first for encoder
        patches = rearrange(patches, 't b c -> b t c')
        features = self.layer_norm(self.transformer(patches))
        features = rearrange(features, 'b t c -> t b c')
        
        return features, backward_indexes
    
    
class MAE_Decoder(nn.Module):
    def __init__(self, img_size=32, patch_size=4, emb_dim=192, in_channels=1, num_layer=12, num_head=3):
        super().__init__()
        
        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = nn.Parameter(torch.zeros((img_size // patch_size) ** 2 + 1, 1, emb_dim)) # + 1 for cls_token
        
        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])
        
        self.head = torch.nn.Linear(emb_dim, in_channels * patch_size ** 2)
        
        self.patch2img = Rearrange('(h w) b (c p1 p2) -> b c (h p1) (w p2)', p1=patch_size, p2=patch_size, h=img_size//patch_size)
        self.init_weight()

    def init_weight(self):
        nn.init.trunc_normal_(self.mask_token, std=.02)
        nn.init.trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, features, backward_indexes):
        # Number of patches
        T = features.shape[0]
        
        # Add zeros for cls_token
        backward_indexes = torch.cat([torch.zeros(1, backward_indexes.shape[1]).to(backward_indexes), backward_indexes + 1], dim=0)
        
        # Concatenate masked tokens to feature vector
        features = torch.cat([features, self.mask_token.expand(backward_indexes.shape[0] - features.shape[0], features.shape[1], -1)], dim=0)
        
        # Set features to correct places
        features = take_indexes(features, backward_indexes) 
        features = features + self.pos_embedding
        
        # Batch first for decoding
        features = rearrange(features, 't b c -> b t c')
        features = self.transformer(features)
        features = rearrange(features, 'b t c -> t b c')
        features = features[1:] # remove global feature
        
        # Features to image size
        patches = self.head(features)
        
        # Set the mask to 1 for masked patches
        mask = torch.zeros_like(patches)
        mask[T-1:] = 1
        mask = take_indexes(mask, backward_indexes[1:] - 1)
        
        # Transform patches to image shape
        img = self.patch2img(patches)
        mask = self.patch2img(mask)
        
        return img, mask
    
    
class MAE_ViT(torch.nn.Module):
    def __init__(self,
                 image_size=32,
                 patch_size=2,
                 emb_dim=192,
                 encoder_layer=12,
                 encoder_head=3,
                 decoder_layer=4,
                 decoder_head=3,
                 mask_ratio=0.75,
                 ) -> None:
        super().__init__()

        self.encoder = MAE_Encoder(image_size, patch_size, emb_dim, encoder_layer, encoder_head, mask_ratio)
        self.decoder = MAE_Decoder(image_size, patch_size, emb_dim, decoder_layer, decoder_head)

    def forward(self, img):
        features, backward_indexes = self.encoder(img)
        predicted_img, mask = self.decoder(features, backward_indexes)
        return predicted_img, mask