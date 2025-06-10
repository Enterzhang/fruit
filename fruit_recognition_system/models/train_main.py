import os
import matplotlib.pyplot as plt
from cnn_trainer import train_cnn_model
from mobilenet_trainer import train_mobilenet_model
from resnet_trainer import train_resnet_model

def plot_model_accuracies_comparison(cnn_train_acc, cnn_val_acc, mobilenet_train_acc, mobilenet_val_acc, resnet_train_acc, resnet_val_acc, plots_save_dir):
    plt.figure(figsize=(15, 7))

    plt.plot(cnn_train_acc, label='CNN 训练准确率', color='blue', linestyle='-')
    plt.plot(cnn_val_acc, label='CNN 验证准确率', color='blue', linestyle='--')

    plt.plot(mobilenet_train_acc, label='MobileNet 训练准确率', color='green', linestyle='-')
    plt.plot(mobilenet_val_acc, label='MobileNet 验证准确率', color='green', linestyle='--')

    plt.plot(resnet_train_acc, label='ResNet 训练准确率', color='red', linestyle='-')
    plt.plot(resnet_val_acc, label='ResNet 验证准确率', color='red', linestyle='--')

    plt.title('模型训练与验证准确率对比')
    plt.xlabel('轮次')
    plt.ylabel('准确率 (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_save_dir, 'all_models_accuracy_comparison.png'))
    plt.close()
    print(f"模型准确率对比图保存到: {os.path.join(plots_save_dir, 'all_models_accuracy_comparison.png')}")

def print_accuracy_table(cnn_train_acc, cnn_val_acc, mobilenet_train_acc, mobilenet_val_acc, resnet_train_acc, resnet_val_acc, plots_save_dir):
    table_data = [
        ["模型", "最终训练准确率 (%)", "最终验证准确率 (%)"],
        ["CNN", f'{cnn_train_acc[-1]:.2f}', f'{cnn_val_acc[-1]:.2f}'],
        ["MobileNet", f'{mobilenet_train_acc[-1]:.2f}', f'{mobilenet_val_acc[-1]:.2f}'],
        ["ResNet", f'{resnet_train_acc[-1]:.2f}', f'{resnet_val_acc[-1]:.2f}']
    ]

    # 确定每列的最大宽度
    column_widths = [max(len(str(item)) for item in col) for col in zip(*table_data)]

    table_string = "\n模型训练结果对比：\n"
    # 打印表头
    header = " | ".join(f'{item:<{column_widths[i]}}' for i, item in enumerate(table_data[0]))
    table_string += header + "\n"
    table_string += "-" * len(header) + "\n"

    # 打印数据行
    for row in table_data[1:]:
        row_str = " | ".join(f'{item:<{column_widths[i]}}' for i, item in enumerate(row))
        table_string += row_str + "\n"
    
    print(table_string)

    # 将表格保存到文件
    table_filepath = os.path.join(plots_save_dir, 'model_accuracy_comparison_table.txt')
    with open(table_filepath, 'w', encoding='utf-8') as f:
        f.write(table_string)
    print(f"模型准确率对比表格保存到: {table_filepath}")

def main():
    # 定义数据集路径
    train_data_path = "D:/fruit/fruitDate/train"
    val_data_path = "D:/fruit/fruitDate/test"

    # 定义模型和图表的保存路径
    base_results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'results')
    model_save_dir = os.path.join(base_results_dir, 'trained_models')
    plots_save_dir = os.path.join(base_results_dir, 'plots')

    # 确保保存目录存在
    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(plots_save_dir, exist_ok=True)

    # 定义训练轮数
    num_epochs = 30 # 您可以根据需要调整此值

    print("开始训练所有模型...")

    # 训练 CNN 模型
    cnn_train_acc, cnn_val_acc = train_cnn_model(train_data_path, val_data_path, model_save_dir, plots_save_dir, num_epochs)

    # 训练 MobileNet 模型
    mobilenet_train_acc, mobilenet_val_acc = train_mobilenet_model(train_data_path, val_data_path, model_save_dir, plots_save_dir, num_epochs)

    # 训练 ResNet 模型
    resnet_train_acc, resnet_val_acc = train_resnet_model(train_data_path, val_data_path, model_save_dir, plots_save_dir, num_epochs)

    print("所有模型训练完成。")

    # 绘制模型准确率对比图
    plot_model_accuracies_comparison(cnn_train_acc, cnn_val_acc, mobilenet_train_acc, mobilenet_val_acc, resnet_train_acc, resnet_val_acc, plots_save_dir)

    # 打印并保存模型准确率对比表格
    print_accuracy_table(cnn_train_acc, cnn_val_acc, mobilenet_train_acc, mobilenet_val_acc, resnet_train_acc, resnet_val_acc, plots_save_dir)

if __name__ == '__main__':
    main() 