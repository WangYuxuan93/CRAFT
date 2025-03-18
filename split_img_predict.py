import os
import cv2
import numpy as np
import subprocess
import argparse

def split_image(image_path, image_output_folder, tile_size=(512, 512)):
    """
    按照 tile_size 进行切分，并保存到 output_folder。
    """
    #base_name = os.path.splitext(os.path.basename(image_path))[0]
    #image_output_folder = os.path.join(output_folder, base_name)
    #os.makedirs(image_output_folder, exist_ok=True)
    
    image = cv2.imread(image_path)
    h, w, _ = image.shape
    tile_w, tile_h = tile_size
    
    num_cols = (w + tile_w - 1) // tile_w  # 计算需要多少列
    num_rows = (h + tile_h - 1) // tile_h  # 计算需要多少行
    
    sub_images = []
    for i in range(num_rows):
        for j in range(num_cols):
            x1, y1 = j * tile_w, i * tile_h
            x2, y2 = min((j + 1) * tile_w, w), min((i + 1) * tile_h, h)
            
            sub_img = image[y1:y2, x1:x2]
            sub_img_name = f"sub_{i}_{j}.jpg"
            sub_img_path = os.path.join(image_output_folder, sub_img_name)
            cv2.imwrite(sub_img_path, sub_img)
            
            sub_images.append((sub_img_path, x1, y1))
    
    return sub_images, image_output_folder

def run_text_detection(script_path, image_output_folder, tmp_folder, model_path="model/craft_mlt_25k.pth", mag_ratio=10, canvas_size=2048):
    """
    调用 CRAFT 文本检测代码。
    """
    subprocess.run([
        "python", script_path,
        "--test_folder", image_output_folder,
        "--output_folder", tmp_folder,
        "--result_folder", tmp_folder,
        "--trained_model", model_path,
        "--mag_ratio", str(mag_ratio),
        "--canvas_size", str(canvas_size),
    ])

def merge_bounding_boxes(sub_images, result_folder, merged_output_file):
    """
    读取所有子图检测的 bounding box 结果，并合并到原图坐标。
    """
    all_bboxes = []
    
    for sub_img_path, x_offset, y_offset in sub_images:
        sub_img_name = "pred_" + os.path.basename(sub_img_path).replace(".jpg", ".txt")
        bbox_file = os.path.join(result_folder, sub_img_name)
        
        if os.path.exists(bbox_file):
            with open(bbox_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if len(line) == 0:
                        continue
                    coords = list(map(int, line.strip().split(",")))
                    for i in range(0, len(coords), 2):
                        coords[i] += x_offset  # x 坐标转换
                        coords[i + 1] += y_offset  # y 坐标转换
                    all_bboxes.append(",".join(map(str, coords)))
    
    with open(merged_output_file, "w") as f:
        f.write("\n".join(all_bboxes))

def draw_bounding_boxes(image_path, merged_output_file, output_image_path):
    """
    在原始图像上绘制所有合并后的 bounding boxes。
    """
    image = cv2.imread(image_path)
    with open(merged_output_file, "r") as f:
        for line in f:
            coords = list(map(int, line.strip().split(",")))
            pts = np.array(coords, np.int32).reshape((-1, 1, 2))
            cv2.polylines(image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
    cv2.imwrite(output_image_path, image)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a folder of images for text detection")
    parser.add_argument("--image_folder", type=str, required=True, help="Path to input image folder")
    parser.add_argument("--script_path", type=str, required=True, help="Path to text detection script")
    parser.add_argument("--tile_size", type=str, default="512,512", help="Tile size for splitting, e.g., 512,512")
    parser.add_argument("--output_folder", type=str, default="./sub_images", help="Folder to save sub-images")
    parser.add_argument("--tmp_folder", type=str, default="./tmp_result", help="Folder to save detection results")
    parser.add_argument("--merged_output_folder", type=str, default="./merged_outputs", help="Folder to save merged bounding box files")
    parser.add_argument("--output_image_folder", type=str, default="./merged_visualization", help="Folder to save output images with bounding boxes")
    
    args = parser.parse_args()
    tile_size = tuple(map(int, args.tile_size.split(",")))

    os.makedirs(args.merged_output_folder, exist_ok=True)
    os.makedirs(args.output_image_folder, exist_ok=True)
    
    for image_name in os.listdir(args.image_folder):
        image_path = os.path.join(args.image_folder, image_name)
        if not os.path.isfile(image_path):
            continue
        
        base_name = os.path.splitext(image_name)[0]
        image_output_folder = os.path.join(args.output_folder, base_name)
        os.makedirs(image_output_folder, exist_ok=True)
        image_tmp_folder = os.path.join(args.tmp_folder, base_name)
        os.makedirs(image_tmp_folder, exist_ok=True)
        
        merged_output_file = os.path.join(args.merged_output_folder, f"pred_{base_name}.txt")
        output_image_path = os.path.join(args.output_image_folder, f"{base_name}.jpg")
        
        sub_images, image_output_folder = split_image(image_path, image_output_folder, tile_size)
        run_text_detection(args.script_path, image_output_folder, image_tmp_folder)
        merge_bounding_boxes(sub_images, image_tmp_folder, merged_output_file)
        draw_bounding_boxes(image_path, merged_output_file, output_image_path)
        
        print(f"Processed {image_name}, results saved to {merged_output_file} and {output_image_path}")
