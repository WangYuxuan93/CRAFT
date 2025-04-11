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

# -------------------- Save Chunks ---------------------
def cache_data_in_chunks(all_results, output_dir, chunk_size=1000):
    os.makedirs(output_dir, exist_ok=True)
    total = len(all_results)
    chunk_count = (total + chunk_size - 1) // chunk_size

    for chunk_idx in range(chunk_count):
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, total)
        chunk = all_results[start:end]

        data = {
            "image_names": [r["img_name"] for r in chunk],
            "region_scores": [r["region_scores"] for r in chunk],
            "affinity_scores": [r["affinity_scores"] for r in chunk],
            "sc_maps": [r["sc_map"] for r in chunk],
        }

        chunk_path = os.path.join(output_dir, f"train_chunk_{chunk_idx}.pt")
        torch.save(data, chunk_path)
        logging.info(f"Saved chunk {chunk_idx}: {chunk_path} ({end - start} samples)")

# -------------------- Main Pipeline ---------------------
def cache_all_data_parallel(images_dir, labels_dir, output_dir, num_workers=None, chunk_size=1000):
    image_names = sorted([f for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".png"))])
    label_names = sorted([f for f in os.listdir(labels_dir) if f.endswith(".txt")])
    assert len(image_names) == len(label_names), "Image and label count mismatch!"

    img_paths = [os.path.join(images_dir, name) for name in image_names]
    label_paths = [os.path.join(labels_dir, name) for name in label_names]
    tasks = list(zip(img_paths, label_paths))

    if num_workers is None:
        total_cpu = cpu_count()
        num_workers = max(1, min(15, total_cpu - 1))
        logging.info(f"Auto-selected num_workers = {num_workers} (Total CPU cores = {total_cpu})")

    logging.info(f"Starting parallel preprocessing with {num_workers} workers...")

    results = []
    with Pool(processes=num_workers, initializer=init_worker) as pool:
        for i, result in enumerate(pool.imap_unordered(process_single_sample, tasks), 1):
            if result is not None:
                results.append(result)
            if i % 100 == 0 or i == len(tasks):
                logging.info(f"Processed {i} / {len(tasks)} samples")

    logging.info(f"Successfully processed {len(results)} / {len(tasks)} samples.")
    cache_data_in_chunks(results, output_dir, chunk_size)

# -------------------------- CLI -----------------------------
if __name__ == "__main__":
    setup_logger("preprocess.log")

    parser = argparse.ArgumentParser(description="Parallel preprocessing and chunked caching of text detection dataset")
    parser.add_argument("--images_dir", type=str, required=True, help="Directory of input images")
    parser.add_argument("--labels_dir", type=str, required=True, help="Directory of text labels (.txt)")
    parser.add_argument("--output_dir", type=str, default="cache", help="Directory to store .pt chunks")
    parser.add_argument("--chunk_size", type=int, default=1000, help="Number of samples per .pt chunk")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of parallel workers to use (default: auto-detect, max 16)")

    args = parser.parse_args()

    cache_all_data_parallel(
        images_dir=args.images_dir,
        labels_dir=args.labels_dir,
        output_dir=args.output_dir,
        num_workers=args.num_workers,
        chunk_size=args.chunk_size
    )
