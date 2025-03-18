import os
import json
from collections import defaultdict

def get_image_paths(root_dir):
    grouped_paths = {}
    
    # 遍历第一层目录（按数字递增的文件夹）
    for first_level in sorted(os.listdir(root_dir), key=lambda x: int(x) if x.isdigit() else float('inf')):
        first_level_path = os.path.join(root_dir, first_level)
        
        if not os.path.isdir(first_level_path):
            continue
        
        # labeled 文件夹路径
        labeled_path = os.path.join(first_level_path, "labeled")
        bbox_path = os.path.join(first_level_path, "bbox")
        
        if not os.path.isdir(labeled_path) or not os.path.isdir(bbox_path):
            continue
        
        # 获取 labeled 文件夹中的所有 PNG 图片（排除 main.png）
        image_files = [
            img for img in os.listdir(labeled_path)
            if img.endswith(".png") and img != "main.png"
        ]
        
        # 计算 bbox 文件夹中对应的预测 box 存在率
        total_images = len(image_files)
        images_with_boxes = 0
        
        for img in image_files:
            bbox_file = os.path.join(bbox_path, f"pred_{img.replace('.png', '.txt')}")
            if os.path.exists(bbox_file):
                with open(bbox_file, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        images_with_boxes += 1
        
        # 计算比率
        bbox_ratio = images_with_boxes / total_images if total_images > 0 else 0
        
        # 存入字典，使用第一层文件夹的路径作为 key
        grouped_paths[first_level_path] = bbox_ratio
    
    # 按照比率从大到小排序
    sorted_paths = dict(sorted(grouped_paths.items(), key=lambda item: item[1], reverse=True))
    
    return sorted_paths

# 示例用法
root_directory = "data/main_map_test/main_map_test"  # 请替换为你的根目录路径
bbox_ratios = get_image_paths(root_directory)

# 保存到 JSON 文件
output_file = "bbox_ratios.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(bbox_ratios, f, indent=4, ensure_ascii=False)

# 输出结果
for key, ratio in bbox_ratios.items():
    print(f"Folder {key}: {ratio:.2%} images have bounding boxes")