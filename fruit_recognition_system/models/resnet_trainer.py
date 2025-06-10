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

# 设置字体以支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体，SimHei 是常见的黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 定义ResNet模型
class ResNetTransferLearning(nn.Module):
    def __init__(self, num_classes):
        super(ResNetTransferLearning, self).__init__()
        self.model = models.resnet18(pretrained=True)  # 加载预训练的ResNet18模型
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)  # 替换最后的全连接层

    def forward(self, x):
        return self.model(x)

def train_resnet_model(train_data_path, val_data_path, model_save_dir, plots_save_dir, num_epochs):
    print("开始训练 ResNet18 模型...")

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.RandomRotation(10),      # 随机旋转
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
    class_names_path = os.path.join(plots_save_dir, 'resnet_class_names.json')
    with open(class_names_path, 'w', encoding='utf-8') as f:
        json.dump(class_names, f, ensure_ascii=False, indent=4)
    print(f'ResNet 类别名称保存到: {class_names_path}')

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    model = ResNetTransferLearning(num_classes=num_classes)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.001)

    # 早停参数
    best_val_acc = 0.0
    patience = 10  # 增加耐心到10个epoch
    no_improvement_count = 0

    # 学习率调整
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 打印设备信息
    print(f"Using device for ResNet18 training: {device}")

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

        print(f'Epoch [{epoch + 1}/{num_epochs}], ResNet18 Train Loss: {avg_loss:.4f}, ResNet18 Train Accuracy: {train_accuracy:.2f}%')

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

        print(f'ResNet18 Validation Loss: {avg_val_loss:.4f}, ResNet18 Validation Accuracy: {val_accuracy:.2f}%')

        # 早停
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            no_improvement_count = 0
            # 保存最佳模型
            torch.save(model.state_dict(), os.path.join(model_save_dir, 'model_resnet100_best.pth'))
        else:
            no_improvement_count += 1
            if no_improvement_count >= patience:
                print(f'ResNet18 Early stopping at epoch {epoch + 1}')
                break

        # 学习率调整
        scheduler.step(val_accuracy)

    # 保存最终模型
    final_model_path = os.path.join(model_save_dir, 'model_resnet100_final.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f'ResNet18 模型保存到: {final_model_path}')

    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='训练损失')
    plt.plot(val_losses, label='验证损失', color='orange')
    plt.title('ResNet18 损失曲线')
    plt.xlabel('轮次')
    plt.ylabel('损失')
    plt.legend()
    plt.savefig(os.path.join(plots_save_dir, 'resnet_loss_curve.png'))
    plt.close() # 关闭图形，防止显示

    # 绘制精度曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_accuracies, label='训练精度')
    plt.plot(val_accuracies, label='验证精度', color='orange')
    plt.title('ResNet18 精度曲线')
    plt.xlabel('轮次')
    plt.ylabel('精度 (%)')
    plt.legend()
    plt.savefig(os.path.join(plots_save_dir, 'resnet_accuracy_curve.png'))
    plt.close() # 关闭图形，防止显示

    # 生成混淆矩阵
    conf_matrix = confusion_matrix(y_true, y_pred)

    # 绘制混淆矩阵
    plot_grouped_confusion_matrix(y_true, y_pred, class_names, plots_save_dir, "ResNet")

    print("ResNet18 模型训练完成。")

    return train_accuracies, val_accuracies 