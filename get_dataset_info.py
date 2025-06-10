import os
import collections

def get_dataset_info(base_path):
    info = collections.defaultdict(lambda: {"classes": 0, "images": 0})
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif') # 可根据需要添加其他图片格式

    for subset in ['train', 'test']:
        subset_path = os.path.join(base_path, subset)
        if not os.path.isdir(subset_path):
            print(f"Warning: {subset_path} does not exist or is not a directory.")
            continue

        class_count = 0
        image_count = 0
        
        # 统计类别数量
        for entry in os.listdir(subset_path):
            entry_path = os.path.join(subset_path, entry)
            if os.path.isdir(entry_path):
                class_count += 1
                # 统计每个类别下的图片数量
                for root, _, files in os.walk(entry_path):
                    for file in files:
                        if file.lower().endswith(image_extensions):
                            image_count += 1
        
        info[subset]["classes"] = class_count
        info[subset]["images"] = image_count
    
    return info

if __name__ == "__main__":
    dataset_base_path = 'D:/fruit/fruitDate' # 根据您的实际路径设置

    dataset_info = get_dataset_info(dataset_base_path)

    print("\n--- 数据集统计信息 ---")
    print("|" + "-"*15 + "|" + "-"*15 + "|" + "-"*15 + "|")
    print("| {:<13} | {:<13} | {:<13} |".format("数据集部分", "类别数量", "图片数量"))
    print("|" + "-"*15 + "|" + "-"*15 + "|" + "-"*15 + "|")

    total_classes = 0
    total_images = 0

    for subset_name in ['train', 'test']:
        classes = dataset_info[subset_name]["classes"]
        images = dataset_info[subset_name]["images"]
        print(f"| {subset_name:<13} | {classes:<13} | {images:<13} |")
        total_classes = max(total_classes, classes) # 类别数量取最大值，因为train和test类别应该相同
        total_images += images

    print("|" + "-"*15 + "|" + "-"*15 + "|" + "-"*15 + "|")
    print(f"| {'总计':<13} | {total_classes:<13} | {total_images:<13} |")
    print("|" + "-"*15 + "|" + "-"*15 + "|" + "-"*15 + "|") 