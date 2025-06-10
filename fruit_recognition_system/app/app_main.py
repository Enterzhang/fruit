import sys
import os
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QComboBox, QMessageBox, QSpinBox
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt
import json
import torch
from torchvision import datasets, transforms
import random

# 导入各个模型的应用模块
from resnet_app import FruitPredictor as ResNetApp
from mobilenet_app import FruitRecognizer as MobileNetApp
from cnn_app import FruitRecognizer as CNNApp

# 移除硬编码的类别名称和数量
# CLASS_NAMES = ['耙耙柑', '白兰瓜', '白萝卜', '白心火龙果', '百香果', '菠萝', '菠萝莓', '菠萝蜜']
# NUM_CLASSES = len(CLASS_NAMES)

class MainApp(QWidget):
    def __init__(self):
        super().__init__()
        self.current_app = None # 用于存储当前启动的应用实例
        self.test_data_path = "D:\\fruit\\fruitDate\\test" # 测试集路径
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('水果识别系统主页')
        self.setGeometry(200, 200, 600, 500) # 调整窗口大小以容纳新元素

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(15)

        title_label = QLabel('选择一个模型进行水果识别')
        title_label.setFont(QFont('Arial', 16, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        self.model_selector = QComboBox()
        self.model_selector.setFont(QFont('Arial', 12))
        self.model_selector.addItem("CNN 模型", "cnn")
        self.model_selector.addItem("MobileNet 模型", "mobilenet")
        self.model_selector.addItem("ResNet18 模型", "resnet")
        layout.addWidget(self.model_selector, alignment=Qt.AlignCenter)

        self.start_button = QPushButton('启动识别应用')
        self.start_button.setFont(QFont('Arial', 14, QFont.Bold))
        self.start_button.setStyleSheet('''
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border-radius: 8px;
                padding: 12px 25px;
                border: none;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        ''')
        self.start_button.setFixedSize(200, 50)
        self.start_button.clicked.connect(self.launch_selected_app)
        layout.addWidget(self.start_button, alignment=Qt.AlignCenter)

        # 添加随机测试功能相关的UI元素
        separator_label = QLabel('----------------------------------')
        separator_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(separator_label)

        test_title_label = QLabel('随机测试准确率')
        test_title_label.setFont(QFont('Arial', 14, QFont.Bold))
        test_title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(test_title_label)

        num_images_layout = QVBoxLayout() # 修改为QVBoxLayout
        num_images_layout.setAlignment(Qt.AlignCenter)
        num_images_label = QLabel('随机测试图片数量:')
        num_images_label.setFont(QFont('Arial', 10))
        num_images_layout.addWidget(num_images_label, alignment=Qt.AlignCenter)

        self.num_test_images_spinbox = QSpinBox()
        self.num_test_images_spinbox.setRange(1, 100000) # 设置一个合理的范围
        self.num_test_images_spinbox.setValue(100) # 默认测试100张图片
        self.num_test_images_spinbox.setFont(QFont('Arial', 10))
        num_images_layout.addWidget(self.num_test_images_spinbox, alignment=Qt.AlignCenter)
        layout.addLayout(num_images_layout)

        self.test_button = QPushButton('开始随机测试')
        self.test_button.setFont(QFont('Arial', 12, QFont.Bold))
        self.test_button.setStyleSheet('''
            QPushButton {
                background-color: #008CBA; /* 蓝色 */
                color: white;
                border-radius: 8px;
                padding: 10px 20px;
                border: none;
            }
            QPushButton:hover {
                background-color: #007B9E;
            }
        ''')
        self.test_button.setFixedSize(180, 40)
        self.test_button.clicked.connect(self.perform_random_test)
        layout.addWidget(self.test_button, alignment=Qt.AlignCenter)

        self.accuracy_label = QLabel('准确率: --')
        self.accuracy_label.setFont(QFont('Arial', 12))
        self.accuracy_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.accuracy_label)

        self.setLayout(layout)

    def launch_selected_app(self):
        selected_model_type = self.model_selector.currentData()
        model_file_name = f'model_{selected_model_type}_final.pth'
        class_names_file = f'{selected_model_type}_class_names.json'

        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_save_dir = os.path.join(current_dir, os.pardir, 'results', 'trained_models')
        plots_save_dir = os.path.join(current_dir, os.pardir, 'results', 'plots')

        model_path = os.path.join(model_save_dir, model_file_name)
        class_names_path = os.path.join(plots_save_dir, class_names_file)

        if not os.path.exists(model_path):
            QMessageBox.warning(self, "模型文件不存在", f"找不到模型文件: {model_path}\n请确保您已先运行训练脚本并生成了模型文件。")
            return
        
        if not os.path.exists(class_names_path):
            QMessageBox.warning(self, "类别文件不存在", f"找不到类别文件: {class_names_path}\n请确保您已先运行训练脚本并生成了类别信息文件。")
            return

        try:
            with open(class_names_path, 'r', encoding='utf-8') as f:
                loaded_class_names = json.load(f)
            loaded_num_classes = len(loaded_class_names)
        except Exception as e:
            QMessageBox.critical(self, "加载类别文件失败", f"无法加载类别文件 {class_names_path}: {e}")
            return

        print(f"启动 {selected_model_type} 模型应用...")

        # 关闭之前的应用实例（如果有的话）
        if self.current_app is not None:
            self.current_app.close() # 关闭窗口
            self.current_app.deleteLater() # 标记为待删除
            self.current_app = None # 清除引用

        if selected_model_type == "cnn":
            self.current_app = CNNApp(model_path, loaded_num_classes, loaded_class_names)
        elif selected_model_type == "mobilenet":
            self.current_app = MobileNetApp(model_path, loaded_num_classes, loaded_class_names)
        elif selected_model_type == "resnet":
            self.current_app = ResNetApp(model_path, loaded_num_classes, loaded_class_names)
        else:
            QMessageBox.critical(self, "错误", "未知的模型类型！")
            return

        self.current_app.show()
        # self.hide() # 可以选择隐藏主窗口

    def perform_random_test(self):
        if self.current_app is None:
            QMessageBox.warning(self, "未选择模型", "请先选择并启动一个模型应用！")
            return

        num_images_to_test = self.num_test_images_spinbox.value()
        if num_images_to_test <= 0:
            QMessageBox.warning(self, "无效数量", "请选择大于0的图片数量进行测试！")
            return

        model = self.current_app.get_model()
        transform = self.current_app.get_transform()
        class_names = self.current_app.get_class_names()

        if model is None or transform is None or class_names is None:
            QMessageBox.critical(self, "错误", "无法获取模型或转换信息，请确保模型应用已正确加载！")
            return

        try:
            # 创建测试数据集
            test_dataset = datasets.ImageFolder(root=self.test_data_path, transform=transform)
        except Exception as e:
            QMessageBox.critical(self, "加载数据集失败", f"无法加载测试数据集: {e}")
            return

        if len(test_dataset) == 0:
            QMessageBox.warning(self, "数据集为空", f"测试数据集路径 {self.test_data_path} 下没有找到任何图片。")
            self.accuracy_label.setText('准确率: 无图片')
            return

        # 确保要测试的图片数量不超过数据集的总数量
        num_to_sample = min(num_images_to_test, len(test_dataset))
        if num_to_sample < num_images_to_test:
            QMessageBox.information(self, "提示", f"测试集只有 {len(test_dataset)} 张图片，将测试全部可用图片。")

        # 随机抽取图片索引
        sampled_indices = random.sample(range(len(test_dataset)), num_to_sample)

        correct_predictions = 0
        total_predictions = 0

        model.eval() # 确保模型处于评估模式
        with torch.no_grad():
            for i in sampled_indices:
                image, true_label_idx = test_dataset[i] # image已经是tensor，true_label_idx是类别索引
                output = model(image.unsqueeze(0)) # 增加batch维度
                _, predicted_label_idx = torch.max(output, 1)

                if predicted_label_idx.item() == true_label_idx:
                    correct_predictions += 1
                total_predictions += 1

        accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
        self.accuracy_label.setText(f'准确率: {accuracy:.2f}%')
        QMessageBox.information(self, "测试完成", f"已对 {total_predictions} 张图片进行随机测试，准确率为: {accuracy:.2f}%")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_app = MainApp()
    main_app.show()
    sys.exit(app.exec_()) 