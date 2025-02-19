import os
import torch

class Config:
    # 数据路径
    DATA_ROOT = "/workspace/lfm_imagenet/data/tiny-imagenet-200"
    TRAIN_DIR = os.path.join(DATA_ROOT, "train")
    VAL_DIR = os.path.join(DATA_ROOT, "val")

    # 数据加载参数
    BATCH_SIZE = 64
    NUM_WORKERS = 4
    PIN_MEMORY = True

    # 模型参数（Transformer版 ViT）
    IMG_SIZE = 64            # 图像尺寸
    PATCH_SIZE = 16           # patch 尺寸
    IN_CHANNELS = 3           # 输入通道数
    EMBED_DIM = 256            # patch embedding 维度
    NUM_CLASSES = 200         # TinyImageNet 类别数

    # Transformer 特有参数
    DEPTH = 6                 # Transformer Block 数量（可根据需要调整）
    NUM_HEADS = 4             # 注意力头数量
    MLP_HIDDEN_DIM = 128      # MLP 隐藏层维度
    DROPOUT = 0.1             # dropout 概率

    # 训练超参数
    NUM_EPOCHS = 90
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    STEP_SIZE = 10
    GAMMA = 0.1

    # 设备配置
    DEVICE = "cuda:5" if torch.cuda.is_available() else "cpu"

    # 日志与模型保存
    LOG_DIR = "./logs"
    CHECKPOINT_DIR = "./checkpoints"
    SAVE_FREQUENCY = 10

# 自动创建日志和检查点目录
os.makedirs(Config.LOG_DIR, exist_ok=True)
os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)