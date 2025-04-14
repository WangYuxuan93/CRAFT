import os
import argparse
import torch
import numpy as np
from PIL import Image
from multiprocessing import Pool, cpu_count
import logging

from utils.box_util import cal_affinity_boxes
from utils.gaussian import GaussianGenerator
from converts.synthText_convert import *
from dataset.textdetect_dataset import get_text_detect_char_box, get_affinity_boxes_list

def setup_logger(log_file="preprocess_individual.log"):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode='w', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

gaussian_generator = None
def init_worker():
    global gaussian_generator
    gaussian_generator = GaussianGenerator()

def process_and_save_single_sample(args):
    idx, img_path, label_path, output_dir = args
    img_name = os.path.basename(img_path)
    save_path = os.path.join(output_dir, f"{idx:06d}.pt")

    if os.path.exists(save_path):
        return f"Skip {idx:06d}.pt (already exists)"

    try:
        image = Image.open(img_path).convert("RGB")
        w, h = image.size
        heat_map_size = (h, w)

        char_boxes_by_word, _ = get_text_detect_char_box(label_path)
        char_boxes_list, affinity_boxes_list = get_affinity_boxes_list(char_boxes_by_word)

        region_scores = gaussian_generator.gen(heat_map_size, char_boxes_list) * 255
        affinity_scores = gaussian_generator.gen(heat_map_size, affinity_boxes_list) * 255
        sc_map = np.ones(heat_map_size, dtype=np.float32) * 255

        data = {
            "img_name": img_name,
            "region_scores": torch.from_numpy(region_scores.astype(np.uint8)),
            "affinity_scores": torch.from_numpy(affinity_scores.astype(np.uint8)),
            "sc_map": torch.from_numpy(sc_map.astype(np.uint8)),
        }

        torch.save(data, save_path)
        return f"Saved {save_path}"
    except Exception as e:
        return f"Error processing {img_name}: {e}"

def process_all_samples(images_dir, labels_dir, output_dir, num_workers=None):
    image_names = sorted([f for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".png"))])
    label_names = sorted([f for f in os.listdir(labels_dir) if f.endswith(".txt")])
    assert len(image_names) == len(label_names), "Image and label count mismatch!"

    os.makedirs(output_dir, exist_ok=True)

    img_paths = [os.path.join(images_dir, name) for name in image_names]
    label_paths = [os.path.join(labels_dir, name) for name in label_names]
    tasks = [(i, img_paths[i], label_paths[i], output_dir) for i in range(len(img_paths))]

    if num_workers is None:
        total_cpu = cpu_count()
        num_workers = max(1, min(16, total_cpu - 1))
        logging.info(f"Auto-selected num_workers = {num_workers} (Total CPU cores = {total_cpu})")

    logging.info(f"Starting per-sample preprocessing with {num_workers} workers...")

    with Pool(processes=num_workers, initializer=init_worker) as pool:
        for i, result in enumerate(pool.imap_unordered(process_and_save_single_sample, tasks), 1):
            if i % 100 == 0:
                logging.info(f"[{i}/{len(tasks)}] {result}")
            elif "Error" in result:
                logging.warning(result)

    logging.info("Processing complete.")

if __name__ == "__main__":
    setup_logger()

    parser = argparse.ArgumentParser(description="Preprocess text detection dataset: one sample = one file")
    parser.add_argument("--images_dir", type=str, required=True, help="Directory of input images")
    parser.add_argument("--labels_dir", type=str, required=True, help="Directory of text labels (.txt)")
    parser.add_argument("--output_dir", type=str, default="cache/train_single", help="Directory to save per-sample .pt files")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of parallel workers")

    args = parser.parse_args()

    process_all_samples(
        images_dir=args.images_dir,
        labels_dir=args.labels_dir,
        output_dir=args.output_dir,
        num_workers=args.num_workers
    )
