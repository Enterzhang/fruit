import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import os
import json
from utils import plot_grouped_confusion_matrix

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号

# 定义MobileNet模型
class MobileNetTransferLearning(nn.Module):
    def __init__(self, num_classes):
        super(MobileNetTransferLearning, self).__init__()
        self.model = models.mobilenet_v2(pretrained=True)
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)

    def forward(self, x):
        return self.model(x)

def train_mobilenet_model(train_data_path, val_data_path, model_save_dir, plots_save_dir, num_epochs):
    print("开始训练 MobileNet 模型...")

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),         # 随机垂直翻转
        transforms.RandomRotation(15),           # 随机旋转
        transforms.RandomResizedCrop(64, scale=(0.8, 1.0)),  # 随机裁剪
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 颜色抖动
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载数据集
    train_dataset = datasets.ImageFolder(root=train_data_path, transform=transform)
    val_dataset = datasets.ImageFolder(root=val_data_path, transform=transform)

    # 获取类别名称
    class_names = train_dataset.classes
    num_classes = len(class_names)

    # 保存类别名称到JSON文件
    class_names_path = os.path.join(plots_save_dir, 'mobilenet_class_names.json')
    with open(class_names_path, 'w', encoding='utf-8') as f:
        json.dump(class_names, f, ensure_ascii=False, indent=4)
    print(f'MobileNet 类别名称保存到: {class_names_path}')

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    model = MobileNetTransferLearning(num_classes=num_classes)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()

    # 使用SGD优化器并增加动量
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)

    # 早停参数设置
    best_val_acc = 0.0
    patience = 10
    no_improvement_count = 0

    # 学习率调整
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 打印设备信息
    print(f"Using device for MobileNet training: {device}")

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    y_true = []
    y_pred = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)
        train_accuracy = 100 * correct / total
        train_accuracies.append(train_accuracy)

        print(f'Epoch [{epoch + 1}/{num_epochs}], MobileNet Train Loss: {avg_loss:.4f}, MobileNet Train Accuracy: {train_accuracy:.2f}%')

        # 验证
        model.eval()
        running_val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                running_val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())

        avg_val_loss = running_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        val_accuracy = 100 * correct / total
        val_accuracies.append(val_accuracy)

        print(f'MobileNet Validation Loss: {avg_val_loss:.4f}, MobileNet Validation Accuracy: {val_accuracy:.2f}%')

        # 早停
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            no_improvement_count = 0
            # 保存最佳模型
            torch.save(model.state_dict(), os.path.join(model_save_dir, 'model_mobilenet100_best.pth'))
        else:
            no_improvement_count += 1
            if no_improvement_count >= patience:
                print(f'MobileNet Early stopping at epoch {epoch + 1}')
                break

        # 学习率调整
        scheduler.step(val_accuracy)

    # 保存最终模型
    final_model_path = os.path.join(model_save_dir, 'model_mobilenet100_final.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f'MobileNet 模型保存到: {final_model_path}')

    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='训练损失')
    plt.plot(val_losses, label='验证损失', color='orange')
    plt.title('MobileNet 损失曲线')
    plt.xlabel('轮次')
    plt.ylabel('损失')
    plt.legend()
    plt.savefig(os.path.join(plots_save_dir, 'mobilenet_loss_curve.png'))
    plt.close() # 关闭图形，防止显示

    # 绘制精度曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_accuracies, label='训练精度')
    plt.plot(val_accuracies, label='验证精度', color='orange')
    plt.title('MobileNet 精度曲线')
    plt.xlabel('轮次')
    plt.ylabel('精度 (%)')
    plt.legend()
    plt.savefig(os.path.join(plots_save_dir, 'mobilenet_accuracy_curve.png'))
    plt.close() # 关闭图形，防止显示

    # 生成混淆矩阵
    conf_matrix = confusion_matrix(y_true, y_pred)

    # 绘制混淆矩阵
    plot_grouped_confusion_matrix(y_true, y_pred, class_names, plots_save_dir, "MobileNet")

    print("MobileNet 模型训练完成。")

    return train_accuracies, val_accuracies 