# 水果识别系统

## 1. 项目概述

本项目旨在构建一个基于深度学习的水果识别系统，通过训练不同类型的卷积神经网络（CNN）模型，实现对多种水果的准确识别。系统设计包含模型训练、训练结果管理和用户友好的识别应用界面，方便用户选择预训练模型进行图像识别。

**主要功能:**
*   支持三种主流深度学习模型（自定义CNN、MobileNetV2、ResNet18）的训练。
*   自动化训练过程，并保存训练好的模型权重、损失曲线、精度曲线和混淆矩阵。
*   提供一个交互式应用界面，用户可选择不同模型进行实时水果图像识别。
*   代码结构清晰，易于理解和扩展，并包含详细注释。

## 2. 目录结构

本项目的目录结构设计如下，旨在清晰地组织各个功能模块：

```
fruit_recognition_system/
├── models/                     # 模型训练相关文件
│   ├── cnn_trainer.py          # 自定义CNN模型的训练函数
│   ├── mobilenet_trainer.py    # MobileNetV2模型的训练函数
│   ├── resnet_trainer.py       # ResNet18模型的训练函数
│   └── train_main.py           # 主训练脚本，用于调用和协调模型训练
├── results/                    # 训练结果保存文件夹
│   ├── trained_models/         # 保存训练好的模型权重文件（.pth）
│   └── plots/                  # 保存训练过程中的图像（损失图、准确率图、混淆矩阵）和类别名称文件（.json）
└── app/                        # 应用可视化界面相关文件
    ├── cnn_app.py              # CNN模型的识别应用界面
    ├── mobilenet_app.py        # MobileNetV2模型的识别应用界面
    ├── resnet_app.py           # ResNet18模型的识别应用界面
    └── app_main.py             # 应用主入口，提供模型选择和启动功能
```

## 3. 模型设计与配置

本项目对比了三种不同架构的深度学习模型：自定义CNN、MobileNetV2 和 ResNet18。所有模型均使用PyTorch框架实现。

### 3.1 通用配置

所有模型在训练前都进行了以下通用设置：

*   **设备选择**: 自动检测并优先使用CUDA（GPU）进行训练，如果不可用则回退到CPU。
    ```python
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ```
*   **中文字体支持**: 确保Matplotlib图表中能正确显示中文标签。
    ```python
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    ```

### 3.2 数据预处理和加载

数据集路径：
*   训练集: `D:/fruit/fruitDate/train`
*   测试集/验证集: `D:/fruit/fruitDate/test`

### 3.2.1 数据集详细信息

**数据集类别 (82类)**:
*   耙耙柑
*   白兰瓜
*   白萝卜
*   白心火龙果
*   百香果
*   菠萝
*   菠萝莓
*   菠萝蜜

**数据集组织结构**:
数据集应按照以下结构组织，其中每个类别的图像存放在对应的子文件夹中：
```
fruitDate/
├── train/
│   ├── 耙耙柑/
│   │   ├── image1.jpg
│   │   ├── ...
│   ├── 白兰瓜/
│   │   ├── imageX.jpg
│   │   ├── ...
│   └── ...
└── test/
    ├── 耙耙柑/
    │   ├── imageA.jpg
    │   ├── ...
    ├── 白兰瓜/
    │   ├── imageY.jpg
    │   ├── ...
    └── ...
```

**数据集大小**: 训练集包含10198张图片，测试集包含2389张图片，总计12587张图片。

数据预处理和增强策略：

*   **图像大小调整**: 所有图像统一调整为 `(64, 64)` 或 `(224, 224)`（MobileNetV2 专用）。
*   **数据增强**:
    *   `transforms.RandomHorizontalFlip()`: 随机水平翻转。
    *   `transforms.RandomRotation(10/15)`: 随机旋转10或15度。
    *   `transforms.RandomVerticalFlip()` (MobileNetV2): 随机垂直翻转。
    *   `transforms.RandomResizedCrop(64, scale=(0.8, 1.0))` (MobileNetV2): 随机裁剪。
    *   `transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)` (MobileNetV2): 颜色抖动。
*   **归一化**: 使用ImageNet数据集的均值和标准差进行归一化。
*   **数据集加载**: 使用`torchvision.datasets.ImageFolder`加载数据集，并使用`torch.utils.data.DataLoader`创建数据加载器，支持批量训练和随机打乱。

### 3.3 自定义CNN模型 (cnn_trainer.py)

*   **模型结构**: `SimpleCNN` 类，包含三层卷积层、池化层和两层全连接层。
    *   卷积层1: 32个3x3滤波器。
    *   卷积层2: 64个3x3滤波器。
    *   卷积层3: 128个3x3滤波器。
    *   池化层: MaxPool2d。
    *   全连接层: 将特征展平后连接到512个神经元，再连接到分类输出。
*   **损失函数**: `nn.CrossEntropyLoss()`，用于多类别分类任务。
*   **优化器**: `optim.Adam()`。
    *   学习率 (`lr`): `0.0001`
    *   权重衰减 (`weight_decay`): `0.001`
*   **学习率调度器**: `optim.lr_scheduler.ReduceLROnPlateau()`，当验证准确率在`patience`个epoch内没有提升时，学习率会降低`factor`倍。
    *   模式 (`mode`): `'max'` (监控最大化的指标，如准确率)
    *   因子 (`factor`): `0.1` (学习率降低10倍)
    *   耐心值 (`patience`): `3`
*   **早停策略**: 如果验证准确率在`patience`（`5`）个epoch内没有提升，训练将停止。

### 3.4 MobileNetV2 模型 (mobilenet_trainer.py)

*   **模型结构**: `MobileNetTransferLearning` 类，基于预训练的`mobilenet_v2`进行迁移学习。
    *   `models.mobilenet_v2(pretrained=True)`: 加载在ImageNet上预训练的MobileNetV2模型。
    *   **替换分类器**: 将原始分类器的最后一层替换为适应本项目类别数的全连接层。
*   **损失函数**: `nn.CrossEntropyLoss()`。
*   **优化器**: `optim.SGD()` (随机梯度下降)。
    *   学习率 (`lr`): `0.001`
    *   动量 (`momentum`): `0.9`
    *   权重衰减 (`weight_decay`): `0.0001`
*   **学习率调度器**: `optim.lr_scheduler.ReduceLROnPlateau()`。
    *   模式 (`mode`): `'max'`
    *   因子 (`factor`): `0.1`
    *   耐心值 (`patience`): `5`
*   **早停策略**: 如果验证准确率在`patience`（`10`）个epoch内没有提升，训练将停止。

### 3.5 ResNet18 模型 (resnet_trainer.py)

*   **模型结构**: `ResNetTransferLearning` 类，基于预训练的`resnet18`进行迁移学习。
    *   `models.resnet18(pretrained=True)`: 加载在ImageNet上预训练的ResNet18模型。
    *   **替换全连接层**: 将原始的全连接层替换为适应本项目类别数的全连接层。
*   **损失函数**: `nn.CrossEntropyLoss()`。
*   **优化器**: `optim.Adam()`。
    *   学习率 (`lr`): `0.0001`
    *   权重衰减 (`weight_decay`): `0.001`
*   **学习率调度器**: `optim.lr_scheduler.ReduceLROnPlateau()`。
    *   模式 (`mode`): `'max'`
    *   因子 (`factor`): `0.1`
    *   耐心值 (`patience`): `5`
*   **早停策略**: 如果验证准确率在`patience`（`10`）个epoch内没有提升，训练将停止。

## 4. 训练与评估

### 4.1 训练流程

`train_main.py` 是项目的主训练脚本。它负责：
1.  定义数据集的训练集和测试集路径（验证集）。
2.  定义模型和图表保存的目录。
3.  确保所有必要的保存目录存在。
4.  依次调用 `cnn_trainer.py` 中的 `train_cnn_model`、`mobilenet_trainer.py` 中的 `train_mobilenet_model` 和 `resnet_trainer.py` 中的 `train_resnet_model` 函数，对各个模型进行训练。

**运行训练**:
在项目根目录（`D:fruit`）下执行以下命令：
```bash
python fruit_recognition_system/models/train_main.py
```

### 4.2 损失函数与优化

*   **损失函数**: 所有模型均采用 `nn.CrossEntropyLoss` 作为损失函数。该损失函数适用于多类别分类任务，它结合了 `LogSoftmax` 和 `NLLLoss`。其计算方式是：首先对模型的输出（logits）进行Softmax操作，得到每个类别的概率分布，然后计算预测概率分布与真实标签之间的负对数似然损失。

*   **优化器**:
    *   自定义CNN和ResNet18模型使用 **Adam优化器**。Adam是一种自适应学习率优化算法，结合了RMSprop和Adagrad的优点，适用于处理稀疏梯度和非平稳目标。
    *   MobileNetV2模型使用 **SGD优化器** (随机梯度下降) **加动量**。SGD是基本的优化器，动量项有助于加速梯度下降过程，减少震荡，更快地收敛。

*   **损失计算**: 在每个训练批次中，模型通过前向传播计算预测结果，然后将预测结果和真实标签输入损失函数计算损失值。接着，通过调用 `.backward()` 方法执行反向传播，计算损失相对于模型参数的梯度。最后，优化器通过 `optimizer.step()` 方法更新模型参数。

### 4.3 关键参数设置

以下是可修改的基本参数及其在代码中的位置：

*   **训练轮次 (Epochs)**:
    *   在 `cnn_trainer.py` 中的 `train_cnn_model` 函数中，`num_epochs` 参数。
    *   在 `mobilenet_trainer.py` 中的 `train_mobilenet_model` 函数中，`num_epochs` 参数。
    *   在 `resnet_trainer.py` 中的 `train_resnet_model` 函数中，`num_epochs` 参数。
    *   **修改方式**: 直接修改函数定义中的默认值或在 `train_main.py` 调用时传入新值。

*   **学习率 (Learning Rate)**:
    *   在各自 `_trainer.py` 文件中，优化器定义时的 `lr` 参数。
    *   **修改方式**: 直接修改优化器定义中的 `lr` 值。

*   **批大小 (Batch Size)**:
    *   在各自 `_trainer.py` 文件中，`DataLoader` 定义时的 `batch_size` 参数。
    *   **修改方式**: 直接修改 `DataLoader` 定义中的 `batch_size` 值。

*   **分类数 (Number of Classes)**:
    *   在各自 `_trainer.py` 文件中，`SimpleCNN`、`MobileNetTransferLearning`、`ResNetTransferLearning` 类的 `__init__` 函数的 `num_classes` 参数。这个值是根据数据集自动推断的 `len(train_dataset.classes)`，通常不需要手动修改，除非数据集类别发生变化。
    *   **修改方式**: 确保您的数据集结构正确，类别数将自动匹配。

*   **其他参数**:
    *   **数据增强**: 在各自 `_trainer.py` 文件中的 `transforms.Compose` 部分，可以添加、修改或删除各种数据增强操作。
    *   **优化器参数**: 例如Adam的`weight_decay`，SGD的`momentum`和`weight_decay`。
    *   **早停耐心值**: `patience` 变量。
    *   **学习率调度器参数**: `scheduler` 定义中的 `factor` 和 `patience`。

### 4.4 训练结果保存

*   **模型文件**: 训练好的模型（包括最佳模型和最终模型）将以 `.pth` 格式保存在 `fruit_recognition_system/results/trained_models/` 目录下。文件名格式为 `model_{model_type}_final.pth` 和 `model_{model_type}_best.pth`。
*   **可视化图表**: 训练过程中的损失曲线 (`_loss_curve.png`)、精度曲线 (`_accuracy_curve.png`) 和混淆矩阵 (`_confusion_matrix.png`) 将以 `.png` 格式保存在 `fruit_recognition_system/results/plots/` 目录下。
*   **类别名称文件**: 每个模型训练后，其对应的类别名称将以 `_class_names.json` 格式保存在 `fruit_recognition_system/results/plots/` 目录下。

### 4.5 模型性能比较与分析 (待补充)

**请在此处填写您运行 `train_main.py` 后获得的训练和验证结果，并进行详细的对比分析。**

例如：
*   **模型准确率对比**: 哪个模型在训练集和验证集上表现最好？哪个最差？
*   **收敛速度对比**: 哪个模型收敛最快？观察损失曲线和精度曲线。
*   **过拟合/欠拟合分析**: 哪个模型可能存在过拟合或欠拟合问题？
*   **混淆矩阵分析**: 分析每个模型在不同类别上的识别效果，找出容易混淆的类别。
*   **性能探讨**: 结合模型的架构特点、优化器选择、数据增强策略等，讨论导致这些性能差异的可能原因。

## 5. 应用可视化

`app_main.py` 是整个水果识别应用的主入口。它提供了一个简洁的用户界面，允许用户选择不同的预训练模型进行图像识别。

**运行应用**:
在项目根目录（`D:fruit`）下执行以下命令：
```bash
python fruit_recognition_system/app/app_main.py
```

**应用功能**:
1.  **模型选择**: 通过下拉菜单选择要使用的模型（CNN、MobileNet、ResNet18）。
2.  **启动识别**: 点击"启动识别应用"按钮，会打开一个新窗口，该窗口是所选模型的独立识别界面。
3.  **图像上传与识别**: 在识别窗口中，用户可以上传本地图片，模型将对图片进行推理并显示预测结果。
4.  **动态类别加载**: 应用会根据训练时保存的类别信息动态加载水果类别名称，确保识别结果的准确性。

