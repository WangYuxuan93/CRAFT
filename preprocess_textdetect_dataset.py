import os
import argparse
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from utils.box_util import cal_affinity_boxes
from utils.gaussian import GaussianGenerator
from converts.synthText_convert import *
from dataset.textdetect_dataset import get_text_detect_char_box, get_affinity_boxes_list

def cache_all_data_to_pt(images_dir, labels_dir, output_file="cache/all_data.pt"):
    image_names = sorted(os.listdir(images_dir))
    label_names = sorted(os.listdir(labels_dir))

    gaussian_generator = GaussianGenerator()

    region_scores_list = []
    affinity_scores_list = []
    sc_map_list = []
    image_name_list = []

    for img_name, label_name in tqdm(zip(image_names, label_names), total=len(image_names)):
        img_path = os.path.join(images_dir, img_name)
        label_path = os.path.join(labels_dir, label_name)

        image = Image.open(img_path)
        w, h = image.size
        heat_map_size = (h, w)

        char_boxes_by_word, _ = get_text_detect_char_box(label_path)
        char_boxes_list, affinity_boxes_list = get_affinity_boxes_list(char_boxes_by_word)

        region_scores = gaussian_generator.gen(heat_map_size, char_boxes_list) * 255
        affinity_scores = gaussian_generator.gen(heat_map_size, affinity_boxes_list) * 255
        sc_map = np.ones(heat_map_size, dtype=np.float32) * 255

        region_scores_list.append(torch.from_numpy(region_scores.astype(np.uint8)))
        affinity_scores_list.append(torch.from_numpy(affinity_scores.astype(np.uint8)))
        sc_map_list.append(torch.from_numpy(sc_map.astype(np.uint8)))
        image_name_list.append(img_name)

    # 打包为字典，一次性保存
    data = {
        "region_scores": region_scores_list,
        "affinity_scores": affinity_scores_list,
        "sc_maps": sc_map_list,
        "image_names": image_name_list
    }

    torch.save(data, output_file)
    print(f"All data saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess text detection data and cache it in PyTorch .pt format")
    parser.add_argument("--images_dir", type=str, required=True, help="Path to the folder containing input images")
    parser.add_argument("--labels_dir", type=str, required=True, help="Path to the folder containing label files")
    parser.add_argument("--output_file", type=str, default="cache/all_data.pt", help="Path to save the output .pt cache file")

    args = parser.parse_args()

    cache_all_data_to_pt(
        images_dir=args.images_dir,
        labels_dir=args.labels_dir,
        output_file=args.output_file
    )