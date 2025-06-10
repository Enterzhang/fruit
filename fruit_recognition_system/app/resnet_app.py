import sys
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog
from PyQt5.QtGui import QPixmap, QPalette, QColor, QFont
from PyQt5.QtCore import Qt
import os

# 定义一个用于迁移学习的ResNet模型类
class ResNetTransferLearning(nn.Module):
    def __init__(self, num_classes, weights=None):
        super(ResNetTransferLearning, self).__init__()
        # 加载预训练的ResNet18模型
        self.model = models.resnet18(weights=weights)
        # 替换最后一层全连接层，以适应特定的输出类别数
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        # 前向传播定义
        return self.model(x)

# 图像预处理函数 (现在将移入类中作为实例属性)

# 定义一个PyQt5窗口类用于水果识别
class FruitPredictor(QWidget):
    def __init__(self, model_path, num_classes, class_names):
        super().__init__()
        self.model = ResNetTransferLearning(num_classes=num_classes)
        # 从文件中加载模型参数
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model.eval()  # 切换模型到评估模式
        self.class_names = class_names

        # 定义图像预处理变换作为实例属性
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),  # 调整图像大小
            transforms.ToTensor(),  # 将图像转换为张量
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 正则化处理
        ])

        self.init_ui()  # 初始化用户界面

    def init_ui(self):
        self.setWindowTitle('水果识别器')  # 设置窗口标题
        self.setGeometry(100, 100, 400, 400)  # 设置窗口大小和位置

        # 创建并设置窗口的背景色
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(224, 247, 250))  # 淡蓝色
        self.setPalette(palette)

        layout = QVBoxLayout()  # 创建垂直布局
        layout.setSpacing(10)  # 设置布局组件间距
        layout.setContentsMargins(20, 20, 20, 20)  # 设置布局边距

        # 设置通用字体
        font = QFont('Arial', 12)

        self.image_label = QLabel(self)  # 用于显示图像的标签
        self.image_label.setAlignment(Qt.AlignCenter)  # 居中对齐
        self.image_label.setFixedSize(200, 200)  # 固定标签大小
        layout.addWidget(self.image_label, alignment=Qt.AlignCenter)  # 添加标签到布局，居中显示

        self.result_label = QLabel('')  # 显示预测结果的标签
        self.result_label.setAlignment(Qt.AlignCenter)  # 居中对齐
        self.result_label.setFont(font)  # 应用字体
        layout.addWidget(self.result_label)

        self.button = QPushButton('上传图片', self)  # 上传图片按钮
        self.button.setFont(font)  # 应用字体
        # 设置按钮样式
        self.button.setStyleSheet('''
            QPushButton {
                background-color: #4CAF50;  /* 绿色背景 */
                color: white;  /* 白色字体 */
                border-radius: 5px;  /* 圆角 */
                padding: 10px;  /* 内边距 */
            }
            QPushButton:hover {
                background-color: #45a049;  /* 悬停时的颜色 */
            }
        ''')
        self.button.clicked.connect(self.upload_image)  # 绑定按钮点击事件处理方法
        layout.addWidget(self.button, alignment=Qt.AlignCenter)  # 添加按钮到布局，居中显示

        self.setLayout(layout)  # 设置窗口的主布局

    def upload_image(self):
        # 打开文件对话框选择图片
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Image Files (*.png *.jpg *.jpeg)", options=options)
        if file_name:
            self.display_image(file_name)  # 显示选中的图片
            self.predict_image(file_name)  # 预测图片中的水果类别

    def display_image(self, file_path):
        pixmap = QPixmap(file_path)  # 创建QPixmap对象
        # 缩放图像以适应标签大小，保持比例并使用平滑转换
        self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def predict_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0)  # 使用实例的transform
        output = self.model(image_tensor)  # 使用模型进行预测
        _, predicted = torch.max(output, 1)  # 获取预测类别
        self.result_label.setText(f'预测结果: {self.class_names[predicted.item()]}')  # 显示预测结果

    # 新增方法，用于获取模型实例
    def get_model(self):
        return self.model

    # 新增方法，用于获取图像预处理转换
    def get_transform(self):
        return self.transform

    # 新增方法，用于获取类别名称
    def get_class_names(self):
        return self.class_names

# 运行应用
if __name__ == '__main__':
    # 这个块现在主要用于测试或从app_main.py调用
    # 假设一个测试用的模型路径和类别数
    test_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'results', 'trained_models', 'model_resnet_best.pth')
    test_num_classes = 82 # 修改为82类
    # 注意：这里的class_names应该从实际训练结果中加载，这里为了测试方便暂时硬编码
    test_class_names = [f'class_{i}' for i in range(test_num_classes)] # 示例类别名称

    app = QApplication(sys.argv)  # 创建应用程序对象
    # 实例化 FruitPredictor 时传入模型路径和类别数
    ex = FruitPredictor(test_model_path, test_num_classes, test_class_names)  # 传入参数
    ex.show()  # 显示窗口
    sys.exit(app.exec_())  # 进入应用程序主循环
