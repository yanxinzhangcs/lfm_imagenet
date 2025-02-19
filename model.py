#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
vil.py

New VIL model:
    1. Uses PatchEmbedding to split the image into patches and map them to the embedding dimension.
    2. Uses an Adaptive Recurrent Unit (AdaptiveRecurrentCell) to update the hidden state for each token sequentially,
       replacing the original Transformer attention module, to integrate information from all patches.
    3. Finally, the last hidden state is used for the classification output.

Note:
    The adaptive recurrent part of this model is similar to an RNN, and due to the sequential dependency,
    it may suffer from gradient vanishing on long sequences. In practice, incorporating gating mechanisms might be beneficial.
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
    This is implemented using a Conv2d layer to both split the image into patches and perform the linear mapping.
    """
    def __init__(self, img_size: int, patch_size: int, in_channels: int, embed_dim: int):
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input image tensor of shape [B, in_channels, H, W]
        Returns:
            patches: Tensor of shape [B, n_patches, embed_dim]
        """
        x = self.proj(x)          # [B, embed_dim, H/patch_size, W/patch_size]
        x = x.flatten(2)          # [B, embed_dim, n_patches]
        x = x.transpose(1, 2)     # [B, n_patches, embed_dim]
        return x

##############################################
# 2. AdaptiveRecurrentCell
##############################################

class AdaptiveRecurrentCell(nn.Module):
    """
    Adaptive Recurrent Cell:
    For each input token, first map the token to the hidden space,
    then generate an update matrix delta (using self.adapt) based on the current token,
    which is added to a shared base weight to obtain the current state transition matrix.
    The matrix is then applied to the previous hidden state, and the result is added to the token's mapped output and a bias.
    Finally, an activation function is applied to update the current hidden state.
    """
    def __init__(self, input_dim: int, hidden_dim: int, adapt_dim: int):
        super(AdaptiveRecurrentCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Map the token to the hidden state space
        self.input_linear = nn.Linear(input_dim, hidden_dim)
        
        # Base state transition weight and bias (shared)
        self.W_base = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.bias = nn.Parameter(torch.zeros(hidden_dim))
        
        # Generate an update matrix delta based on the current token
        self.adapt = nn.Linear(adapt_dim, hidden_dim * hidden_dim)
        
        # Activation function
        self.activation = nn.Tanh()
    
    def forward(self, x: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Current token, of shape [B, input_dim]
            h_prev: Previous hidden state, of shape [B, hidden_dim]
        Returns:
            h_new: New hidden state, of shape [B, hidden_dim]
        """
        # Map token through a linear layer
        x_mapped = self.input_linear(x)  # [B, hidden_dim]
        
        # Generate delta, output shape [B, hidden_dim * hidden_dim]
        # Reshape to [B, hidden_dim, hidden_dim] for matrix multiplication
        delta = self.adapt(x)            
        delta = delta.view(-1, self.hidden_dim, self.hidden_dim)
        
        # Current state transition matrix = base weight + delta
        # Expand the shared base weight to [B, hidden_dim, hidden_dim]
        W_base_expanded = self.W_base.unsqueeze(0).expand_as(delta)
        W_current = W_base_expanded + delta  # [B, hidden_dim, hidden_dim]
        
        # Expand h_prev to [B, 1, hidden_dim] for batch matrix multiplication
        h_prev_unsq = h_prev.unsqueeze(1)  # [B, 1, hidden_dim]
        # Compute h_transformed = h_prev * W_current^T, resulting in [B, hidden_dim]
        h_transformed = torch.bmm(h_prev_unsq, W_current.transpose(1, 2)).squeeze(1)
        
        # Update hidden state: sum the token mapping, transformed previous state, and bias, then apply activation
        h_new = self.activation(x_mapped + h_transformed + self.bias)
        return h_new

##############################################
# 3. AdaptiveRecurrentModel
##############################################

class AdaptiveRecurrentModel(nn.Module):
    """
    Adaptive Recurrent Model:
    Accepts a sequence of tokens and sequentially updates the hidden state using the AdaptiveRecurrentCell.
    Finally, returns the last hidden state as the overall representation.
    """
    def __init__(self, input_dim: int, hidden_dim: int, adapt_dim: int):
        super(AdaptiveRecurrentModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.cell = AdaptiveRecurrentCell(input_dim, hidden_dim, adapt_dim)
    
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: Sequence of tokens with shape [T, B, input_dim]
        Returns:
            h: Final hidden state of shape [B, hidden_dim]
        """
        T, B, _ = tokens.size()
        h = torch.zeros(B, self.hidden_dim, device=tokens.device, dtype=tokens.dtype)
        for t in range(T):
            x = tokens[t]  # Current token, [B, input_dim]
            h = self.cell(x, h)
        return h

##############################################
# 4. VIL Model
##############################################

class VIL(nn.Module):
    """
    VIL Model:
        1. Uses PatchEmbedding to convert the input image into patch tokens.
        2. Reshapes the token sequence to [T, B, embed_dim] and sequentially updates the hidden state using the AdaptiveRecurrentModel.
           The final hidden state represents the overall image information.
        3. A classification head then maps the hidden state to the number of classes to produce the output.
    """
    def __init__(self,
                 img_size: int = 224,
                 patch_size: int = 16,
                 in_channels: int = 3,
                 num_classes: int = 1000,
                 embed_dim: int = 768):
        super(VIL, self).__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        # Process patch tokens with the AdaptiveRecurrentModel (both input and hidden state have embed_dim dimension)
        self.adaptive_recurrent = AdaptiveRecurrentModel(input_dim=embed_dim, hidden_dim=embed_dim, adapt_dim=embed_dim)
        # Classification head: maps the final hidden state to the number of classes
        self.head = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input image tensor of shape [B, in_channels, H, W]
        Returns:
            logits: Classification output of shape [B, num_classes]
        """
        B = x.size(0)
        # Convert the image into patch tokens, shape [B, n_patches, embed_dim]
        tokens = self.patch_embed(x)
        # Reshape to [T, B, embed_dim], where T is the number of patches
        tokens = tokens.transpose(0, 1)
        # Sequentially update the hidden state using the AdaptiveRecurrentModel, yielding a final hidden state of shape [B, embed_dim]
        h_final = self.adaptive_recurrent(tokens)
        # Apply the classification head to produce logits
        logits = self.head(h_final)
        return logits

##############################################
# 5. Testing Example
##############################################

if __name__ == "__main__":
    img_size = 224
    patch_size = 16
    in_channels = 3
    num_classes = 1000
    embed_dim = 64   # Using a smaller embed_dim for simple testing
    batch_size = 2

    # Randomly generate an input image tensor of shape [B, in_channels, H, W]
    x = torch.randn(batch_size, in_channels, img_size, img_size)
    model = VIL(img_size=img_size, patch_size=patch_size, in_channels=in_channels,
                num_classes=num_classes, embed_dim=embed_dim)
    logits = model(x)
    print("VIL output shape:", logits.shape)  # Expected output shape: [B, num_classes]