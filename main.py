from config import Config
from data.dataloader import get_dataloaders
from model import LFModel,test_inference
from train import train_one_epoch
from validate import validate
import torch
from torch.optim import SGD
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
        Config.DATA_ROOT,  # TinyImageNet 数据根目录
        Config.BATCH_SIZE,
        Config.NUM_WORKERS
    )

    # 初始化模型
    model = LFModel(
        token_dim=Config.TOKEN_DIM,
        channel_dim=Config.CHANNEL_DIM,
        expert_dim=Config.EXPERT_DIM,
        adapt_dim=Config.ADAPT_DIM,
        num_experts=Config.NUM_EXPERTS
    )
    model.output_layer = torch.nn.Linear(Config.TOKEN_DIM, 200)  # TinyImageNet 有 200 个类别
    model = model.to(Config.DEVICE)

    ###############
    test_inference(model, train_loader, Config.DEVICE)
    ###############


    # 损失函数和优化器
    criterion = CrossEntropyLoss()
    optimizer = SGD(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        momentum=Config.MOMENTUM,
        weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = StepLR(optimizer, step_size=Config.STEP_SIZE, gamma=Config.GAMMA)

    # 训练与验证循环
    for epoch in range(Config.NUM_EPOCHS):
        # 记录当前 epoch
        logger.info(f"\nEpoch {epoch + 1}/{Config.NUM_EPOCHS}")

        # 训练和验证
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, Config.DEVICE)
        val_loss, val_acc = validate(model, val_loader, criterion, Config.DEVICE)
        scheduler.step()

        # 输出进度和准确率到终端
        print(f"Epoch {epoch + 1}/{Config.NUM_EPOCHS} - Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%\n")

        # 保存模型到日志文件
        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        if (epoch + 1) % Config.SAVE_FREQUENCY == 0:
            save_path = f"{Config.CHECKPOINT_DIR}/lfmodel_epoch{epoch + 1}.pth"
            torch.save(model.state_dict(), save_path)
            logger.info(f"Model checkpoint saved at {save_path}")

if __name__ == "__main__":
    main()