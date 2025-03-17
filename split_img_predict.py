import os
import cv2
import numpy as np
import subprocess
import argparse

def split_image(image_path, output_folder, tile_size=(512, 512)):
    """
    按照 tile_size 进行切分，并保存到 output_folder。
    """
    image = cv2.imread(image_path)
    h, w, _ = image.shape
    tile_w, tile_h = tile_size
    
    num_cols = (w + tile_w - 1) // tile_w  # 计算需要多少列
    num_rows = (h + tile_h - 1) // tile_h  # 计算需要多少行
    
    sub_images = []
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for i in range(num_rows):
        for j in range(num_cols):
            x1, y1 = j * tile_w, i * tile_h
            x2, y2 = min((j + 1) * tile_w, w), min((i + 1) * tile_h, h)
            
            sub_img = image[y1:y2, x1:x2]
            sub_img_name = f"sub_{i}_{j}.jpg"
            sub_img_path = os.path.join(output_folder, sub_img_name)
            cv2.imwrite(sub_img_path, sub_img)
            
            sub_images.append((sub_img_path, x1, y1))
    
    return sub_images

def run_text_detection(script_path, sub_image_path, output_folder, result_folder="data/split_main_map_test/results",
                       model_path="model/craft_mlt_25k.pth", mag_ratio=10, canvas_size=2048):
    """
    调用 CRAFT 文本检测代码。
    """
    subprocess.run([
        "python", script_path,
        "--test_folder", os.path.dirname(sub_image_path),
        "--output_folder", output_folder,
        "--result_folder", result_folder,
        "--trained_model", model_path,
        "--mag_ratio", str(mag_ratio),
        "--canvas_size", str(canvas_size),
    ])

def transform_bounding_boxes(bbox_file, x_offset, y_offset):
    """
    读取 bbox 文件并转换坐标。
    """
    transformed_bboxes = []
    with open(bbox_file, "r") as f:
        for line in f:
            coords = list(map(int, line.strip().split(",")))
            for i in range(0, len(coords), 2):
                coords[i] += x_offset  # x 坐标转换
                coords[i + 1] += y_offset  # y 坐标转换
            transformed_bboxes.append(",".join(map(str, coords)))
    return transformed_bboxes

def merge_bounding_boxes(sub_images, output_folder, merged_output_file):
    """
    读取所有子图检测的 bounding box 结果，并合并到原图坐标。
    """
    all_bboxes = []
    
    for sub_img_path, x_offset, y_offset in sub_images:
        sub_img_name = os.path.basename(sub_img_path).replace(".jpg", ".txt")
        bbox_file = os.path.join(output_folder, sub_img_name)
        
        if os.path.exists(bbox_file):
            all_bboxes.extend(transform_bounding_boxes(bbox_file, x_offset, y_offset))
    
    with open(merged_output_file, "w") as f:
        f.write("\n".join(all_bboxes))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split image and run text detection")
    parser.add_argument("--image_path", type=str, required=True, help="Path to input image")
    parser.add_argument("--script_path", type=str, required=True, help="Path to text detection script")
    parser.add_argument("--tile_size", type=str, default="512,512", help="Tile size for splitting, e.g., 512,512")
    parser.add_argument("--output_folder", type=str, default="./sub_images", help="Folder to save sub-images")
    parser.add_argument("--tmp_folder", type=str, default="./tmp_result", help="Folder to save detection results")
    parser.add_argument("--merged_output", type=str, default="merged_bboxes.txt", help="Merged bounding box output file")
    
    args = parser.parse_args()
    tile_size = tuple(map(int, args.tile_size.split(",")))
    
    # 1. 切分图片
    sub_images = split_image(args.image_path, args.output_folder, tile_size)
    
    # 2. 对每个子图运行文本检测
    for sub_img_path, _, _ in sub_images:
        run_text_detection(args.script_path, sub_img_path, args.tmp_folder)
    
    # 3. 合并检测结果
    merge_bounding_boxes(sub_images, args.tmp_folder, args.merged_output)
    
    print(f"Merged bounding boxes saved to {args.merged_output}")
