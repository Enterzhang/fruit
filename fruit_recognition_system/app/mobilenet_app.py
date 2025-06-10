import sys
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QFileDialog
from PyQt5.QtGui import QPixmap, QPalette, QColor, QFont
from PyQt5.QtCore import Qt
import os

# 定义模型结构
class MobileNetTransferLearning(nn.Module):
    def __init__(self, num_classes):
        super(MobileNetTransferLearning, self).__init__()
        self.model = models.mobilenet_v2(weights='DEFAULT')  # 使用预训练的MobileNetV2
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)  # 修改最后一层

    def forward(self, x):
        return self.model(x)

# 创建应用程序
class FruitRecognizer(QWidget):
    def __init__(self, model_path, num_classes, class_names):
        super().__init__()
        self.initUI()
        self.model = MobileNetTransferLearning(num_classes=num_classes)

        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),  # MobileNet通常输入224x224的图像, 但我们统一为64x64
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.class_names = class_names

    def initUI(self):
        self.setWindowTitle('水果识别器')
        self.setGeometry(100, 100, 400, 400)

        # 设置背景色为淡蓝色
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(224, 247, 250))
        self.setPalette(palette)

        layout = QVBoxLayout()
        layout.setSpacing(10)  # 设置组件间距
        layout.setContentsMargins(20, 20, 20, 20)  # 设置边距

        # 设置字体
        font = QFont('Arial', 12)

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.image_label)

        self.result_label = QLabel(self)
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setFont(font)
        layout.addWidget(self.result_label)

        self.upload_button = QPushButton('上传图片', self)
        self.upload_button.setFont(font)
        self.upload_button.setStyleSheet('''
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
        self.upload_button.clicked.connect(self.uploadImage)
        layout.addWidget(self.upload_button)

        self.setLayout(layout)

    def uploadImage(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "打开图像文件", "", "Images (*.png *.xpm *.jpg *.bmp)", options=options)
        if file_path:
            self.displayImage(file_path)
            self.predictImage(file_path)

    def displayImage(self, file_path):
        pixmap = QPixmap(file_path)
        self.image_label.setPixmap(pixmap.scaled(300, 300, Qt.KeepAspectRatio))

    def predictImage(self, file_path):
        image = Image.open(file_path).convert('RGB')
        image = self.transform(image).unsqueeze(0)

        with torch.no_grad():
            output = self.model(image)
            _, predicted = torch.max(output, 1)
            predicted_class = self.class_names[predicted.item()]

        self.result_label.setText(f'预测结果: {predicted_class}')

    # 新增方法，用于获取模型实例
    def get_model(self):
        return self.model

    # 新增方法，用于获取图像预处理转换
    def get_transform(self):
        return self.transform

    # 新增方法，用于获取类别名称
    def get_class_names(self):
        return self.class_names

if __name__ == '__main__':
    test_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'results', 'trained_models', 'model_mobilenet100_best.pth')
    test_num_classes = 82 # 修改为82类
    test_class_names = [f'class_{i}' for i in range(test_num_classes)] # 示例类别名称

    app = QApplication(sys.argv)
    ex = FruitRecognizer(test_model_path, test_num_classes, test_class_names)
    ex.show()
    sys.exit(app.exec_())
