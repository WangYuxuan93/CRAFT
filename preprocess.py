import os
import json
import cv2
import matplotlib.pyplot as plt

def normalize_point(x, y, width, height):
    return round(x / width, 6), round(y / height, 6)

def bbox_to_segmentation_line(label, bbox, width, height):
    x1, x2, y1, y2 = bbox["x1"], bbox["x2"], bbox["y1"], bbox["y2"]
    points = [
        (x1, y1), (x2, y1),
        (x2, y2), (x1, y2)
    ]
    normalized_points = [normalize_point(x, y, width, height) for x, y in points]
    coord_str = " ".join(f"{x} {y}" for x, y in normalized_points)
    return f"{label} {coord_str}"

def draw_and_save(folder_path, output_path, debug=False):
    # 创建输出子文件夹
    images_dir = os.path.join(output_path, "images")
    labels_dir = os.path.join(output_path, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    for subfolder in sorted(os.listdir(folder_path)):
        subfolder_path = os.path.join(folder_path, subfolder)
        if os.path.isdir(subfolder_path) and subfolder.startswith("map_"):
            try:
                number = int(subfolder.split("_")[1])
                filename_base = f"{number:05d}"
            except:
                print(f"跳过非法子文件夹名: {subfolder}")
                continue

            json_path = os.path.join(subfolder_path, "meta.json")
            tif_path = os.path.join(subfolder_path, "main_l.tif")

            if not os.path.exists(json_path) or not os.path.exists(tif_path):
                print(f"缺少文件于 {subfolder_path}，跳过。")
                continue

            # 读取 meta.json
            with open(json_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            legend_box = meta.get("legend_pixel_coordinates")
            map_frame_box = meta.get("map_frame_pixel_coordinates")

            if not legend_box or not map_frame_box:
                print(f"{subfolder} 中缺少必要的坐标信息，跳过。")
                continue

            # 读取图像
            img = cv2.imread(tif_path)
            if img is None:
                print(f"无法读取图像 {tif_path}，跳过。")
                continue

            height, width = img.shape[:2]

            # 保存原图（不画框）
            output_img_path = os.path.join(images_dir, f"{filename_base}.jpg")
            cv2.imwrite(output_img_path, img)

            # 保存标签为 TXT
            output_txt_path = os.path.join(labels_dir, f"{filename_base}.txt")
            with open(output_txt_path, "w") as f:
                f.write(bbox_to_segmentation_line(0, legend_box, width, height) + "\n")
                f.write(bbox_to_segmentation_line(1, map_frame_box, width, height) + "\n")

            if debug:
                # 显示调试图（含框）
                img_copy = img.copy()
                cv2.rectangle(img_copy, (legend_box["x1"], legend_box["y1"]),
                            (legend_box["x2"], legend_box["y2"]),
                            color=(0, 0, 255), thickness=3)
                cv2.rectangle(img_copy, (map_frame_box["x1"], map_frame_box["y1"]),
                            (map_frame_box["x2"], map_frame_box["y2"]),
                            color=(255, 0, 0), thickness=3)

                img_rgb = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
                plt.figure(figsize=(12, 10))
                plt.imshow(img_rgb)
                plt.title(f"Bounding Boxes in {subfolder}")
                plt.axis('off')
                plt.show()

                print(f"保存图像: {output_img_path}；标签: {output_txt_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert tif+json dataset to YOLO-style segmentation format.")
    parser.add_argument("input_folder", help="Path to top-level folder containing map_n subfolders.")
    parser.add_argument("output_folder", help="Path to output folder for images and labels.")
    args = parser.parse_args()

    draw_and_save(args.input_folder, args.output_folder)
