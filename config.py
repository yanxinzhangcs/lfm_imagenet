import os
import torch
class Config:
    # 数据路径
    DATA_ROOT = "/workspace/lfm_imagenet/data/tiny-imagenet-200"  # 根目录
    TRAIN_DIR = os.path.join(DATA_ROOT, "train")  # 训练集目录
    VAL_DIR = os.path.join(DATA_ROOT, "val")      # 验证集目录
    BATCH_SIZE = 64                              # 批量大小
    NUM_WORKERS = 4                              # 数据加载器的线程数
    PIN_MEMORY = True                            # 是否使用 pinned memory 加速数据加载

    # 模型参数
    TOKEN_DIM = 512                              # Token 维度
    CHANNEL_DIM = 512                            # 通道维度
    EXPERT_DIM = 512                             # 专家模块维度
    ADAPT_DIM = 128                              # 自适应层维度
    NUM_EXPERTS = 4                              # 专家数量

    # 训练超参数
    NUM_EPOCHS = 90                              # 总训练轮数
    LEARNING_RATE = 1e-5                         # 初始学习率
    MOMENTUM = 0.8                               # SGD 动量参数
    WEIGHT_DECAY = 1e-4                          # 权重衰减
    STEP_SIZE = 10                               # 学习率衰减步长
    GAMMA = 0.1                                  # 学习率衰减因子

    # 设备
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # 自动选择设备

    # 日志与模型保存
    LOG_DIR = "./logs"                           # 日志文件目录
    CHECKPOINT_DIR = "./checkpoints"             # 模型检查点保存目录
    SAVE_FREQUENCY = 10                          # 模型保存间隔（以轮为单位）

# 创建日志和检查点目录
os.makedirs(Config.LOG_DIR, exist_ok=True)
os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)