from config import Config
from data.dataloader import get_dataloaders
from model import VIL  # 使用新的 VIL 模型
from train import train_one_epoch
from validate import validate
import torch
from torch.optim.lr_scheduler import StepLR
from torch.nn import CrossEntropyLoss
from loguru import logger

# 配置日志
log_file = f"{Config.LOG_DIR}/train.log"  # 日志文件路径
logger.add(log_file, level="INFO", format="{time} - {level} - {message}", rotation="10 MB")  # 保存到文件

# 删除默认日志处理器
logger.remove()

# 添加终端日志，只输出训练进度和准确率
logger.add(lambda msg: print(msg, end=""), level="INFO", format="{message}")

def main():
    # 加载数据
    train_loader, val_loader = get_dataloaders(
        Config.DATA_ROOT,   # 数据根目录
        Config.BATCH_SIZE,
        Config.NUM_WORKERS
    )

    # 初始化新的 VIL 模型
    model = VIL(
        img_size=Config.IMG_SIZE,           # 例如 224
        patch_size=Config.PATCH_SIZE,         # 例如 16
        in_channels=Config.IN_CHANNELS,       # 例如 3
        num_classes=200,                      # TinyImageNet 有 200 个类别
        embed_dim=Config.EMBED_DIM            # 例如 64 或 768，根据实际配置
    )
    model = model.to(Config.DEVICE)

    # 损失函数和优化器
    criterion = CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0001,      # 初始学习率
        weight_decay=1e-4
    )
    scheduler = StepLR(optimizer, step_size=Config.STEP_SIZE, gamma=Config.GAMMA)

    # 训练与验证循环
    for epoch in range(Config.NUM_EPOCHS):
        logger.info(f"\nEpoch {epoch + 1}/{Config.NUM_EPOCHS}")

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, Config.DEVICE)
        val_loss, val_acc = validate(model, val_loader, criterion, Config.DEVICE)
        scheduler.step()

        # 输出进度和准确率到终端
        print(f"Epoch {epoch + 1}/{Config.NUM_EPOCHS} - Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%\n")

        # 输出训练日志
        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        if (epoch + 1) % Config.SAVE_FREQUENCY == 0:
            save_path = f"{Config.CHECKPOINT_DIR}/vil_epoch{epoch + 1}.pth"
            torch.save(model.state_dict(), save_path)
            logger.info(f"Model checkpoint saved at {save_path}")

if __name__ == "__main__":
    main()