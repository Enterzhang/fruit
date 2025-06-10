import torch
import torch.nn as nn
from torchvision import models
from torchviz import make_dot
import os

# 导入模型定义 (假设 num_classes 为 82，与您的数据集匹配)
from fruit_recognition_system.models.cnn_trainer import SimpleCNN
from fruit_recognition_system.models.mobilenet_trainer import MobileNetTransferLearning
from fruit_recognition_system.models.resnet_trainer import ResNetTransferLearning

NUM_CLASSES = 82

def generate_cnn_arch_diagram():
    print("生成 CNN 模型架构图...")
    model = SimpleCNN(num_classes=NUM_CLASSES)
    dummy_input = torch.randn(1, 3, 64, 64) # CNN 输入图像大小
    dot = make_dot(model(dummy_input), params=dict(model.named_parameters()))
    dot.render(os.path.join('fruit_recognition_system', 'results', 'plots', "cnn_model_architecture"), format="dot", cleanup=True)
    print("cnn_model_architecture.dot 已生成。")

def generate_mobilenet_arch_diagram():
    print("生成 MobileNet 模型架构图...")
    model = MobileNetTransferLearning(num_classes=NUM_CLASSES)
    dummy_input = torch.randn(1, 3, 64, 64) # MobileNet 输入图像大小 (根据您的数据预处理)
    dot = make_dot(model(dummy_input), params=dict(model.named_parameters()))
    dot.render(os.path.join('fruit_recognition_system', 'results', 'plots', "mobilenet_model_architecture"), format="dot", cleanup=True)
    print("mobilenet_model_architecture.dot 已生成。")

def generate_resnet_arch_diagram():
    print("生成 ResNet18 模型架构图...")
    model = ResNetTransferLearning(num_classes=NUM_CLASSES)
    dummy_input = torch.randn(1, 3, 64, 64) # ResNet18 输入图像大小 (根据您的数据预处理)
    dot = make_dot(model(dummy_input), params=dict(model.named_parameters()))
    dot.render(os.path.join('fruit_recognition_system', 'results', 'plots', "resnet_model_architecture"), format="dot", cleanup=True)
    print("resnet_model_architecture.dot 已生成。")

if __name__ == "__main__":
    # 确保 plots 目录存在
    plots_dir = os.path.join('fruit_recognition_system', 'results', 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    generate_cnn_arch_diagram()
    generate_mobilenet_arch_diagram()
    generate_resnet_arch_diagram()
    print("所有模型架构图的 .dot 文件已生成在 fruit_recognition_system/results/plots/ 目录下。")
    print("请使用 Graphviz 工具将 .dot 文件转换为 SVG 图像，例如：")
    print("dot -Tsvg fruit_recognition_system/results/plots/cnn_model_architecture.dot -o fruit_recognition_system/results/plots/cnn_model_architecture.svg") 