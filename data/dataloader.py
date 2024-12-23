from torch.utils.data import DataLoader
from torchvision import transforms
from .tinyimagenet import TinyImageNet  # 引入自定义的数据集类

def get_dataloaders(data_dir, batch_size, num_workers):
    # 定义数据增强和预处理
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(64),  # TinyImageNet 图片大小为 64x64
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载训练集和验证集
    dataset_train = TinyImageNet(data_dir, train=True, transform=train_transform)
    dataset_val = TinyImageNet(data_dir, train=False, transform=val_transform)

    # 创建数据加载器
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader