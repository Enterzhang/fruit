import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from collections import defaultdict
from sklearn.metrics import confusion_matrix

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def plot_grouped_confusion_matrix(y_true, y_pred, class_names, plots_save_dir, model_name):
    num_classes = len(class_names)
    if num_classes == 0:
        print("没有类别可以绘制混淆矩阵。")
        return

    # 定义每个子图显示的类别数量
    classes_per_subplot = 12
    num_subplots = (num_classes + classes_per_subplot - 1) // classes_per_subplot

    # 确定子图布局（例如：每行3个子图）
    cols = 3
    rows = (num_subplots + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 10, rows * 8))
    axes = axes.flatten() # 将axes转换为一维数组，方便迭代

    for i in range(num_subplots):
        start_idx = i * classes_per_subplot
        end_idx = min((i + 1) * classes_per_subplot, num_classes)
        
        # 获取当前子图的类别索引和名称
        current_indices = list(range(start_idx, end_idx))
        current_class_names = class_names[start_idx:end_idx]

        # 过滤y_true和y_pred，只包含当前子图的标签
        # 这里需要将原始标签映射到新的子图标签（0到len(current_class_names)-1）
        filtered_y_true = []
        filtered_y_pred = []
        
        # 创建从原始全局索引到当前子图局部索引的映射
        global_to_local_idx = {global_idx: local_idx for local_idx, global_idx in enumerate(current_indices)}

        for true_label, pred_label in zip(y_true, y_pred):
            if true_label in current_indices and pred_label in current_indices:
                filtered_y_true.append(global_to_local_idx[true_label])
                filtered_y_pred.append(global_to_local_idx[pred_label])

        if len(filtered_y_true) > 0:
            # 重新计算针对子集的混淆矩阵
            sub_conf_matrix = confusion_matrix(filtered_y_true, filtered_y_pred, labels=list(range(len(current_class_names))))

            ax = axes[i] # 使用循环索引i作为子图索引
            sns.heatmap(sub_conf_matrix, annot=True, fmt='d', cmap='Blues',
                        xticklabels=current_class_names, yticklabels=current_class_names, ax=ax)
            ax.set_title(f'{model_name} 混淆矩阵 (类别 {start_idx+1}-{end_idx})')
            ax.set_ylabel('真实标签')
            ax.set_xlabel('预测标签')
        else:
            # 如果没有数据，仍然需要标题和标签来指示哪个子图是空的
            ax = axes[i]
            ax.set_title(f'{model_name} 混淆矩阵 (类别 {start_idx+1}-{end_idx}) - 无数据')
            ax.set_xlabel('预测标签')
            ax.set_ylabel('真实标签')
            ax.set_xticks([]) # 隐藏刻度
            ax.set_yticks([]) # 隐藏刻度

    # 隐藏未使用的子图
    for j in range(num_subplots, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()

    # 生成文件名和完整路径
    filename = f"{model_name.lower().replace(' ', '_')}_grouped_confusion_matrix.png"
    filepath = os.path.join(plots_save_dir, filename)

    plt.savefig(filepath)
    plt.close()
    print(f"""{model_name} 分组混淆矩阵保存到: {filepath}""") 