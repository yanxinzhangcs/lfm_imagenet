import os
import torch

class Config:
    # 数据路径
    DATA_ROOT = "/workspace/lfm_imagenet/data/tiny-imagenet-200"  # 数据根目录
    TRAIN_DIR = os.path.join(DATA_ROOT, "train")  # 训练集目录
    VAL_DIR = os.path.join(DATA_ROOT, "val")      # 验证集目录

    # 数据加载参数
    BATCH_SIZE = 64           # 批量大小
    NUM_WORKERS = 4           # 数据加载器线程数
    PIN_MEMORY = True         # 是否使用 pinned memory 加速数据加载

    # 模型参数（适用于 VIL 模型）
    IMG_SIZE = 64            # 图像尺寸
    PATCH_SIZE = 16           # patch 尺寸
    IN_CHANNELS = 3           # 图像通道数
    EMBED_DIM = 256            # patch embedding 维度
    NUM_CLASSES = 200         # TinyImageNet 类别数

    # 训练超参数
    NUM_EPOCHS = 90           # 总训练轮数
    LEARNING_RATE = 1e-4      # 初始学习率
    WEIGHT_DECAY = 1e-4       # 权重衰减
    STEP_SIZE = 10            # 学习率衰减步长
    GAMMA = 0.1               # 学习率衰减因子

    # 设备配置
    DEVICE = "cuda:4" if torch.cuda.is_available() else "cpu"

    # 日志与模型保存
    LOG_DIR = "./logs"              # 日志文件目录
    CHECKPOINT_DIR = "./checkpoints"  # 模型检查点保存目录
    SAVE_FREQUENCY = 10             # 模型保存间隔（以轮为单位）

# 自动创建日志和检查点目录
os.makedirs(Config.LOG_DIR, exist_ok=True)
os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)