import os
import argparse
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import logging

from utils.box_util import cal_affinity_boxes
from utils.gaussian import GaussianGenerator
from converts.synthText_convert import *
from dataset.textdetect_dataset import get_text_detect_char_box, get_affinity_boxes_list  # ← 替换为你的实际模块路径

# ---------------------- Logging Setup ----------------------
def setup_logger(log_file="preprocess.log"):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode='w', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

# ---------------------- Worker Init ------------------------
gaussian_generator = None
def init_worker():
    global gaussian_generator
    gaussian_generator = GaussianGenerator()

# ------------------- Single Sample Processor ----------------
def process_single_sample(args):
    img_path, label_path = args
    img_name = os.path.basename(img_path)

    try:
        logging.info(f"Processing: {img_name}")
        image = Image.open(img_path)
        w, h = image.size
        heat_map_size = (h, w)

        char_boxes_by_word, _ = get_text_detect_char_box(label_path)
        char_boxes_list, affinity_boxes_list = get_affinity_boxes_list(char_boxes_by_word)

        region_scores = gaussian_generator.gen(heat_map_size, char_boxes_list) * 255
        affinity_scores = gaussian_generator.gen(heat_map_size, affinity_boxes_list) * 255
        sc_map = np.ones(heat_map_size, dtype=np.float32) * 255

        return {
            "img_name": img_name,
            "region_scores": torch.from_numpy(region_scores.astype(np.uint8)),
            "affinity_scores": torch.from_numpy(affinity_scores.astype(np.uint8)),
            "sc_map": torch.from_numpy(sc_map.astype(np.uint8)),
        }
    except Exception as e:
        logging.error(f"Error processing {img_name}: {e}")
        return None

# -------------------- Main Preprocessing ---------------------
def cache_all_data_parallel(images_dir, labels_dir, output_file, num_workers=None):
    image_names = sorted([f for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".png"))])
    label_names = sorted([f for f in os.listdir(labels_dir) if f.endswith(".txt")])
    assert len(image_names) == len(label_names), "Image and label count mismatch!"

    img_paths = [os.path.join(images_dir, name) for name in image_names]
    label_paths = [os.path.join(labels_dir, name) for name in label_names]
    tasks = list(zip(img_paths, label_paths))

    if num_workers is None:
        total_cpu = cpu_count()
        num_workers = max(1, min(16, total_cpu - 1))
        logging.info(f"Auto-selected num_workers = {num_workers} (Total CPU cores = {total_cpu})")

    logging.info(f"Starting parallel preprocessing with {num_workers} workers...")

    with Pool(processes=num_workers, initializer=init_worker) as pool:
        results = list(pool.imap(process_single_sample, tasks))

    results = [r for r in results if r is not None]
    logging.info(f"Successfully processed {len(results)} / {len(tasks)} samples.")

    data = {
        "image_names": [r["img_name"] for r in results],
        "region_scores": [r["region_scores"] for r in results],
        "affinity_scores": [r["affinity_scores"] for r in results],
        "sc_maps": [r["sc_map"] for r in results],
    }

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    torch.save(data, output_file)
    logging.info(f"All data saved to: {output_file}")

# -------------------------- CLI -----------------------------
if __name__ == "__main__":
    setup_logger("preprocess.log")

    parser = argparse.ArgumentParser(description="Parallel preprocessing and caching of text detection dataset")
    parser.add_argument("--images_dir", type=str, required=True, help="Directory of input images")
    parser.add_argument("--labels_dir", type=str, required=True, help="Directory of text labels (.txt)")
    parser.add_argument("--output_file", type=str, default="cache/all_data.pt", help="Path to output .pt cache file")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of parallel workers to use (default: auto-detect, max 16)")

    args = parser.parse_args()

    cache_all_data_parallel(
        images_dir=args.images_dir,
        labels_dir=args.labels_dir,
        output_file=args.output_file,
        num_workers=args.num_workers
    )
