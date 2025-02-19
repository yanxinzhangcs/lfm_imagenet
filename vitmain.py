from vitconfig import Config
from data.dataloader import get_dataloaders
from vit import ViTTransformer  # Transformer版本的模型
from train import train_one_epoch
from validate import validate
import torch
from torch.optim.lr_scheduler import StepLR
from torch.nn import CrossEntropyLoss
from loguru import logger

# 配置日志：保存日志到文件，同时在终端输出训练进度和准确率
log_file = f"{Config.LOG_DIR}/train.log"
logger.add(log_file, level="INFO", format="{time} - {level} - {message}", rotation="10 MB")
logger.remove()  # 删除默认处理器
logger.add(lambda msg: print(msg, end=""), level="INFO", format="{message}")

def main():
    # 加载数据
    train_loader, val_loader = get_dataloaders(
        Config.DATA_ROOT,
        Config.BATCH_SIZE,
        Config.NUM_WORKERS
    )

    # 初始化 Transformer 版本的 ViT 模型
    model = ViTTransformer(
        img_size=Config.IMG_SIZE,
        patch_size=Config.PATCH_SIZE,
        in_channels=Config.IN_CHANNELS,
        num_classes=Config.NUM_CLASSES,
        embed_dim=Config.EMBED_DIM,
        depth=Config.DEPTH,
        num_heads=Config.NUM_HEADS,
        mlp_hidden_dim=Config.MLP_HIDDEN_DIM,
        dropout=Config.DROPOUT
    )
    model = model.to(Config.DEVICE)

    # 损失函数和优化器
    criterion = CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
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
        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # 每隔 SAVE_FREQUENCY 轮保存一次模型
        if (epoch + 1) % Config.SAVE_FREQUENCY == 0:
            save_path = f"{Config.CHECKPOINT_DIR}/vit_epoch{epoch + 1}.pth"
            torch.save(model.state_dict(), save_path)
            logger.info(f"Model checkpoint saved at {save_path}")

if __name__ == "__main__":
    main()