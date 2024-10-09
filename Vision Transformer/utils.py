import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    def __init__(self, embed_dim, patch_size, num_patches, dropout, in_channels):
        super().__init__()

        # Returns a patched output of size [batch, embed_dim, n_patch_col, n_patch_row]
        self.patcher = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=embed_dim,
                kernel_size=patch_size,
                stride=patch_size,
            ),
            nn.Flatten(2)
        )
        
        self.cls_token = nn.Parameter(torch.randn(size=(1, 1, embed_dim)), requires_grad=True)
        self.pos_embedding = nn.Parameter(torch.randn(size=(1, num_patches + 1, embed_dim)), requires_grad=True) # +1 for cls token
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # Transform to [batch, n_patches, emb_dim]
        x = self.patcher(x).permute(0, 2, 1)
        
        # Expand the cls_token for each element in batch and add that as the first element in dim=1
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        
        # Apply the position embedding and dropout
        x += self.pos_embedding
        x = self.dropout(x)
        return x

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, key_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.key_dim = key_dim
        
        # Use one matrix for Key, Query, Value weight matrices
        self.W = nn.Parameter(torch.empty(embed_dim, 3*key_dim))
        self.softmax = nn.Softmax(dim = -1)
    
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.W)
        
    def forward(self, x):
        QKV = torch.matmul(x, self.W)
        Q = QKV[:, :, :self.key_dim]
        K = QKV[:, :, self.key_dim:self.key_dim*2 ]
        V = QKV[:, :, self.key_dim*2:]
        
        # Scaled dot product
        K_T = K.transpose(-2, -1)
        dot_product = torch.matmul(Q, K_T) 
        
        # Scaled dot product
        Z = dot_product / (self.key_dim ** 0.5)
        
        # Apply softmax to obtain attention scores
        score = self.softmax(Z)
        
        # Get weighted values
        output = torch.matmul(score, V)
        
        return output
        
# Multi head attention
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        
        assert embed_dim % num_heads == 0, "Embedding dimension should be divisible with the number of heads"
        self.key_dim = embed_dim // num_heads
        
        # Init multihead-attention
        self.multi_head_attention = nn.ModuleList([SelfAttention(embed_dim, self.key_dim) for _ in range(num_heads)])
        
        # Multihead-attention weight
        self.W = nn.Parameter(torch.empty((embed_dim, embed_dim)))
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.W)
        
    def forward(self, x):
        # Compute self-attention scores of each head 
        scores = [attention(x) for attention in self.multi_head_attention]
        
        # Concat attentions
        Z = torch.cat(scores, -1)
        
        # Compute multi-head attention score
        attention_score = torch.matmul(Z, self.W)
        
        return attention_score

class MLP(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout=0.):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        
        self.gelu = nn.GELU()
        self.dropout1 = nn.Dropout(p = dropout)
        self.dropout2 = nn.Dropout(p = dropout)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.dropout2(x)
        return x
        
        

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)
        
        self.mlp = MLP(embed_dim, hidden_dim, dropout)
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        
    def forward(self,x):
        norm1 = self.norm1(x)
        
        # Skip connection 1
        attention = self.attention(norm1)
        x = x + self.dropout1(attention)
    
        # Skip connection 2
        norm2 = self.norm2(x)
        x = x + self.mlp(norm2)
        x = self.dropout2(x)
        return x
    
class MLPHead(nn.Module):
    def __init__(self, embedding_dim, num_classes):
        super(MLPHead, self).__init__()
        self.num_classes = num_classes
        
        # single linear layer
        self.mlp_head = nn.Linear(embedding_dim, num_classes)
        
    def forward(self, x):
        x = self.mlp_head(x)
        return x

class ViT_torch(nn.Module):
    def __init__(self, embed_dim, patch_size, num_patches, dropout, in_channels, num_heads, num_encoders, hidden_dim, num_classes):
        super().__init__()
        self.embedding = PatchEmbedding(embed_dim, patch_size, num_patches, dropout, in_channels)
                
        transformer_encoder_list = [TransformerEncoderLayer(embed_dim, num_heads, hidden_dim, dropout) for _ in range(num_encoders)] 
        self.transformer_encoder_layers = nn.Sequential(*transformer_encoder_list)
        
        # mlp head
        self.mlp_head = MLPHead(embed_dim, num_classes)
        
    def forward(self, x):
        # Embed the input
        patched_embeds = self.embedding(x)
        
        # feed patch embeddings into a stack of Transformer encoders
        encoder_output = self.transformer_encoder_layers(patched_embeds)
        
        # extract [class] token from encoder output
        output_class_token = encoder_output[:, 0]
        
        # pass token through mlp head for classification
        y = self.mlp_head(output_class_token)
        
        return y
    
        
        
        