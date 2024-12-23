import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from typing import Optional, Tuple
import math

class AdaptiveLinear(nn.Module):
    """
    Adaptive Linear layer whose weight and bias adapt based on input.
    """

    def __init__(
        self, in_features: int, out_features: int, adapt_dim: int
    ):
        super(AdaptiveLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(
            torch.randn(out_features, in_features)
        )
        self.bias = nn.Parameter(torch.randn(out_features))

        # Linear transformation for adapting the weight based on input
        self.adapt = nn.Linear(adapt_dim, out_features * in_features)

    def forward(
        self, x: torch.Tensor, adapt_input: torch.Tensor
    ) -> torch.Tensor:
        adapt_input = adapt_input.mean(dim=0)
        adapt_weight = self.adapt(adapt_input).reshape(
            self.out_features, self.in_features
        )
        weight = self.weight + adapt_weight
        return F.linear(x, weight, self.bias)


class TokenMixing(nn.Module):
    """
    Token mixing layer that performs token-wise interactions using adaptive linear layers.
    Operates across the sequence dimension (sequence_length).
    """

    def __init__(self, token_dim: int, adapt_dim: int):
        super(TokenMixing, self).__init__()
        self.token_mixing = AdaptiveLinear(
            token_dim, token_dim, adapt_dim
        )
       

    def forward(
        self, x: torch.Tensor, adapt_input: torch.Tensor
    ) -> torch.Tensor:
        # x: [batch_size, sequence_length, embedding_dim]
        batch_size, seq_length, embed_dim = x.shape
        x = x.reshape(
            batch_size * seq_length, embed_dim
        )  # Flatten sequence for linear transformation
        x_mixed = self.token_mixing(x, adapt_input)
        return x_mixed.reshape(batch_size, seq_length, embed_dim)


class ChannelMixing(nn.Module):
    """
    Channel mixing layer that performs cross-channel interactions using adaptive linear layers.
    Operates across the embedding dimension (embedding_dim).
    """

    def __init__(self, channel_dim: int, adapt_dim: int):
        super(ChannelMixing, self).__init__()
        self.channel_mixing = AdaptiveLinear(
            channel_dim, channel_dim, adapt_dim
        )

    def forward(
        self, x: torch.Tensor, adapt_input: torch.Tensor
    ) -> torch.Tensor:
        # x: [batch_size, sequence_length, embedding_dim]
        return self.channel_mixing(x, adapt_input)


class MixtureOfExperts(nn.Module):
    """
    Mixture of Experts (MoE) module that dynamically selects experts based on input.
    Operates after channel and token mixing.
    """

    def __init__(
        self, expert_dim: int, num_experts: int, adapt_dim: int
    ):
        super(MixtureOfExperts, self).__init__()
        self.experts = nn.ModuleList(
            [
                AdaptiveLinear(expert_dim, expert_dim, adapt_dim)
                for _ in range(num_experts)
            ]
        )
        self.gating = nn.Linear(adapt_dim, num_experts)

    def forward(
        self, x: torch.Tensor, adapt_input: torch.Tensor
    ) -> torch.Tensor:
        gate_scores = F.softmax(self.gating(adapt_input), dim=-1)
        output = sum(
            gate_scores[:, i].unsqueeze(1).unsqueeze(2) * expert(x, adapt_input)
            for i, expert in enumerate(self.experts)
        )
        return output




class ImageToEmbedding(nn.Module):
    """
    Image to Patch Embedding with Positional Encoding.
    """
    def __init__(self, img_size=64, patch_size=16, in_channels=3, embed_dim=512):
        super(ImageToEmbedding, self).__init__()
        assert img_size % patch_size == 0, "Image size must be divisible by patch size."
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim

        # 将每个图像块映射到固定嵌入维度
        self.patch_embedding = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        # 学习的位置信息
        self.positional_encoding = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim)
        )

        # 初始化位置编码
        nn.init.trunc_normal_(self.positional_encoding, std=0.02)

    def forward(self, x):
        """
        Forward pass:
        - x: [batch_size, in_channels, img_size, img_size]
        Returns:
        - embeddings: [batch_size, num_patches, embed_dim]
        """
        batch_size = x.size(0)
        # 将图像转为嵌入
        embeddings = self.patch_embedding(x)  # [batch_size, embed_dim, h_patches, w_patches]
        embeddings = embeddings.flatten(2).transpose(1, 2)  # [batch_size, num_patches, embed_dim]

        # 加入位置信息
        embeddings = embeddings + self.positional_encoding
        return embeddings

class LFModel(nn.Module):
    """
    Custom LF Model architecture combining token mixing, channel mixing, and MoE.
    Accepts 3D input tensor: [batch_size, sequence_length, embedding_dim].
    """

    def __init__(
        self,
        token_dim: int,
        channel_dim: int,
        expert_dim: int,
        adapt_dim: int,
        num_experts: int,
        img_size=64,
        patch_size=16,
        in_channels=3,
    ):
        super(LFModel, self).__init__()
        self.img_to_embedding = ImageToEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=token_dim
        )
        self.featurizer = nn.Linear(token_dim, adapt_dim)
        self.token_mixer = TokenMixing(token_dim, adapt_dim)
        self.channel_mixer = ChannelMixing(channel_dim, adapt_dim)
        self.moe = MixtureOfExperts(
            expert_dim, num_experts, adapt_dim
        )
        self.global_pooling = nn.AdaptiveAvgPool1d(1)  # 平均池化，整合块信息
        self.output_layer = nn.Linear(token_dim, 200)  # 最终分类头，假设 200 类

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #print("Input shape: {}", x.shape)

        x = self.img_to_embedding(x)  # [batch_size, num_patches, token_dim]
        #logger.info("Embedding shape: {}", x.shape)

        # Featurization stage
        batch_size, seq_length, embed_dim = x.shape
        adapt_input = self.featurizer(
            x.mean(dim=1)
        )  # Aggregate across sequence for adaptation
        #logger.info("Featurization complete. Shape: {}", adapt_input.shape)

        # Token Mixing
        token_mixed = self.token_mixer(x, adapt_input)
        #logger.info("Token mixing complete. Shape: {}", token_mixed.shape)

        # Channel Mixing
        channel_mixed = self.channel_mixer(token_mixed, adapt_input)
        #logger.info("Channel mixing complete. Shape: {}", channel_mixed.shape)

        # Mixture of Experts
        expert_output = self.moe(channel_mixed, adapt_input)
        #logger.info("Mixture of Experts complete. Shape: {}",expert_output.shape,)

        # 全局池化
        global_features = self.global_pooling(expert_output.transpose(1, 2)).squeeze(-1)  # [batch_size, token_dim]

        # 分类
        output = self.output_layer(global_features)  # [batch_size, num_classes]
        #logger.info("Output shape: {}", output.shape)
        return output


# Instantiate and test the model
if __name__ == "__main__":
    batch_size, seq_length, embedding_dim = 32, 128, 512
    token_dim, channel_dim, expert_dim, adapt_dim, num_experts = (
        embedding_dim,
        embedding_dim,
        embedding_dim,
        128,
        4,
    )
    model = LFModel(
        token_dim, channel_dim, expert_dim, adapt_dim, num_experts
    )

    input_tensor = torch.randn(
        batch_size, seq_length, embedding_dim
    )  # 3D text tensor
    output = model(input_tensor)
    logger.info("Model forward pass complete.")

def test_inference(model, dataloader, device):
    model.eval()  # 切换到推理模式
    with torch.no_grad():  # 禁用梯度计算
        for inputs, _ in dataloader:
            inputs = inputs.to(device)

            print("Testing inference...")
            # 检查输入是否有 NaN
            if torch.isnan(inputs).any():
                print("NaN detected in inputs!")
                return

            # 前向传播并检查每层输出
            x = inputs
            for name, module in model.named_children():
                try:
                    x = module(x)
                except Exception as e:
                    print(f"Error in module {name}: {e}")
                    return

                # 检查中间结果是否为 NaN
                if torch.isnan(x).any():
                    print(f"NaN detected in module {name}'s output!")
                    return
                print(f"{name} output - min: {x.min()}, max: {x.max()}")
            
            print("Inference completed successfully!")
            break  # 只测试一个批次
