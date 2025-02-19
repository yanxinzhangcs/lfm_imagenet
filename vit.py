#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
vit_transformer.py

Transformer version of the model:
    1. Uses PatchEmbedding to split the input image into patch tokens.
    2. Adds a classification token (cls token) and positional encoding.
    3. Uses multiple TransformerBlocks for token interactions (including multi-head self-attention and MLP).
    4. Finally, maps the cls token to the number of classes via a classification head to get the final prediction.

This model structure is similar to ViT and can be used for comparison with the previous VIL model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

##############################################
# 1. PatchEmbedding
##############################################

class PatchEmbedding(nn.Module):
    """
    Splits the input image into patches and maps them to the embedding dimension.
    This is implemented using Conv2d to both split into patches and perform the linear mapping.
    """
    def __init__(self, img_size: int, patch_size: int, in_channels: int, embed_dim: int):
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, in_channels, H, W]
        x = self.proj(x)                # [B, embed_dim, H/patch_size, W/patch_size]
        x = x.flatten(2)                # [B, embed_dim, n_patches]
        x = x.transpose(1, 2)           # [B, n_patches, embed_dim]
        return x

##############################################
# 2. TransformerBlock
##############################################

class TransformerBlock(nn.Module):
    """
    TransformerBlock consists of two parts:
        1. Multi-head self-attention layer (MultiheadAttention) with residual connection and LayerNorm.
        2. MLP (Feed Forward Network) with residual connection and LayerNorm.
    """
    def __init__(self, embed_dim: int, num_heads: int, mlp_hidden_dim: int, dropout: float = 0.1):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x shape: [seq_len, B, embed_dim]
        # Self-attention part
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        # MLP part
        x = x + self.mlp(self.norm2(x))
        return x

##############################################
# 3. ViTTransformer Model
##############################################

class ViTTransformer(nn.Module):
    """
    Transformer version of the ViT model:
        1. Uses PatchEmbedding to convert the image into patch tokens.
        2. Adds a cls token and positional encoding.
        3. Passes the tokens through multiple TransformerBlocks for global information exchange.
        4. Uses a classification head to map the cls token to the number of classes.
    """
    def __init__(self,
                 img_size: int = 224,
                 patch_size: int = 16,
                 in_channels: int = 3,
                 num_classes: int = 200,
                 embed_dim: int = 64,
                 depth: int = 12,
                 num_heads: int = 4,
                 mlp_hidden_dim: int = 128,
                 dropout: float = 0.1):
        super(ViTTransformer, self).__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        n_patches = self.patch_embed.n_patches
        
        # cls token used for classification
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # Positional encoding, shape: [1, 1 + n_patches, embed_dim]
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + n_patches, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        
        # Transformer encoder blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_hidden_dim, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        # Classification head
        self.head = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        # Convert the image into patch tokens, shape [B, n_patches, embed_dim]
        x = self.patch_embed(x)
        # Add the cls token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, embed_dim]
        x = torch.cat((cls_tokens, x), dim=1)           # [B, 1 + n_patches, embed_dim]
        # Add positional encoding
        x = x + self.pos_embed
        x = self.pos_drop(x)
        # Transformer expects input shape: [seq_len, B, embed_dim]
        x = x.transpose(0, 1)                           # [seq_len, B, embed_dim]
        for block in self.blocks:
            x = block(x)
        # Transpose back to [B, seq_len, embed_dim]
        x = x.transpose(0, 1)
        x = self.norm(x)
        # Use the cls token as the image representation
        cls_final = x[:, 0]                             # [B, embed_dim]
        logits = self.head(cls_final)                   # [B, num_classes]
        return logits

##############################################
# 4. Testing Example
##############################################

if __name__ == "__main__":
    img_size = 224
    patch_size = 16
    in_channels = 3
    num_classes = 200
    embed_dim = 64   # Using a smaller embed_dim for simple testing
    batch_size = 2
    depth = 6          # Use a shallower depth for testing
    num_heads = 4
    mlp_hidden_dim = 128
    
    # Randomly generate an image input of shape [B, in_channels, H, W]
    x = torch.randn(batch_size, in_channels, img_size, img_size)
    model = ViTTransformer(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=in_channels,
        num_classes=num_classes,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_hidden_dim=mlp_hidden_dim,
        dropout=0.1
    )
    logits = model(x)
    print("ViTTransformer output shape:", logits.shape)  # Expected output: [B, num_classes]