import os
import numpy as np
import argparse
from skimage.draw import polygon
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from collections import defaultdict

def expand_box(coords, scale=1.1, image_shape=None):
    center = np.mean(coords, axis=0)
    expanded_coords = np.round(center + (coords - center) * scale).astype(int)

    if image_shape is not None:
        expanded_coords[:, 0] = np.clip(expanded_coords[:, 0], 0, image_shape[0] - 1)
        expanded_coords[:, 1] = np.clip(expanded_coords[:, 1], 0, image_shape[1] - 1)

    return expanded_coords

def read_txt_file(file_path, is_gold=True, debug=False):
    class_coord_dict = defaultdict(list)

    with open(file_path, 'r', encoding="utf-8") as f:
        for line in f:
            if len(line.strip()) == 0:
                continue
            parts = line.strip().split(",")
            coords = np.array(list(map(float, parts[:8]))).reshape(4, 2)
            if is_gold:
                cls = parts[-2].strip()
                class_coord_dict[cls].append(coords)
            else:
                class_coord_dict['pred'].append(coords)

    return class_coord_dict

def read_folder(folder_path, is_gold=True):
    folder_data = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            class_coord_dict = read_txt_file(file_path, is_gold)
            folder_data[filename] = class_coord_dict
    return folder_data

def calc_classwise_coverage(class_coords_dict, pred_coords_list, image_shape, scale=1.1):
    coverage_results = {}
    pred_mask = np.zeros(image_shape, dtype=np.uint8)
    pred_coords_list = [expand_box(coords, scale=scale, image_shape=image_shape) for coords in pred_coords_list]

    for coords in pred_coords_list:
        rr, cc = polygon(coords[:, 0], coords[:, 1], image_shape)
        pred_mask[rr, cc] = 1

    for cls, coords_list in class_coords_dict.items():
        if cls == 'pred':
            continue
        cls_mask = np.zeros(image_shape, dtype=np.uint8)
        for coords in coords_list:
            rr, cc = polygon(coords[:, 0], coords[:, 1], image_shape)
            cls_mask[rr, cc] = 1

        intersection = np.logical_and(cls_mask, pred_mask).sum()
        cls_area = np.sum(cls_mask)
        coverage = intersection / cls_area if cls_area > 0 else 0.0
        coverage_results[cls] = {'coverage': coverage, 'area': cls_area, 'count': len(coords_list)}

    return coverage_results

def calc_mask_iou(gt_coords_list, pred_coords_list, image_shape, scale=1.1, debug=False):
    pred_coords_list = [expand_box(coords, scale=scale, image_shape=image_shape) for coords in pred_coords_list]

    gt_mask = np.zeros(image_shape, dtype=np.uint8)
    pred_mask = np.zeros(image_shape, dtype=np.uint8)

    for coords in gt_coords_list:
        rr, cc = polygon(coords[:, 0], coords[:, 1], image_shape)
        gt_mask[rr, cc] = 1

    for coords in pred_coords_list:
        rr, cc = polygon(coords[:, 0], coords[:, 1], image_shape)
        pred_mask[rr, cc] = 1

    if debug:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(gt_mask.T, cmap='gray')
        plt.title("Ground Truth Mask")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(pred_mask.T, cmap='gray')
        plt.title("Prediction Mask")
        plt.axis('off')
        plt.show()

    intersection = np.logical_and(gt_mask, pred_mask).sum()
    union = np.logical_or(gt_mask, pred_mask).sum()

    gt_mask_area = np.sum(gt_mask)
    pred_mask_area = np.sum(pred_mask)

    coverage_ratio = intersection / gt_mask_area if gt_mask_area > 0 else 0.0
    pred_extra_area = (pred_mask_area - intersection) / pred_mask_area if pred_mask_area > 0 else 0.0

    return (intersection / float(union) if union > 0 else 0.0, gt_mask_area, pred_mask_area, coverage_ratio, pred_extra_area)

def evaluate_text_detection(gold_folder: str, pred_folder: str, img_folder: str, scale: float = 1.1, iou_threshold: float = 0.5, debug=False) -> dict:
    gold_data = read_folder(gold_folder, is_gold=True)
    pred_data = read_folder(pred_folder, is_gold=False)

    ious_list = []
    total_tp = 0
    gt_mask_area_list = []
    pred_mask_area_list = []
    coverage_ratio_list = []
    pred_extra_ratio_list = []

    classwise_coverage_all = defaultdict(list)
    classwise_area = defaultdict(float)
    classwise_count = defaultdict(int)

    for img_id, gt_class_coords in tqdm(gold_data.items()):
        pred_id = img_id.replace('gt_img', 'pred_img')
        if pred_id not in pred_data:
            print(f"预测文件 {pred_id} 不存在，跳过该图像。")
            continue

        pred_coords = pred_data[pred_id]['pred']

        img_filename = img_id.replace('gt_img_', 'img_').replace('.txt', '.jpg')
        img_path = os.path.join(img_folder, img_filename)
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            image_shape = (img.shape[1], img.shape[0])
        else:
            print(f"警告: 找不到图片 {img_path}，跳过该图像。")
            continue

        all_gt_coords = []
        for cls, coords_list in gt_class_coords.items():
            all_gt_coords.extend(coords_list)

        mask_iou, gt_area, pred_area, coverage_ratio, pred_extra_ratio = \
            calc_mask_iou(all_gt_coords, pred_coords, image_shape=image_shape, scale=scale, debug=debug)

        ious_list.append(mask_iou)
        gt_mask_area_list.append(gt_area)
        pred_mask_area_list.append(pred_area)
        coverage_ratio_list.append(coverage_ratio)
        pred_extra_ratio_list.append(pred_extra_ratio)
        total_tp += int(mask_iou >= iou_threshold)

        coverage_results = calc_classwise_coverage(gt_class_coords, pred_coords, image_shape=image_shape, scale=scale)
        for cls, stats in coverage_results.items():
            classwise_coverage_all[cls].append(stats['coverage'])
            classwise_area[cls] += stats['area']
            classwise_count[cls] += stats['count']

    overall_acc = total_tp / len(gold_data) if len(gold_data) > 0 else 0.0

    iou_bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    iou_histogram, _ = np.histogram(ious_list, bins=iou_bins)
    iou_percentage = (iou_histogram / len(ious_list)) if len(ious_list) > 0 else [0] * len(iou_histogram)

    result = {
        'Average IoU (per file)': np.mean(ious_list),
        'Overall Acc': overall_acc,
        'Average Coverage Ratio': np.mean(coverage_ratio_list),
        'Average Prediction Extra Area Ratio': np.mean(pred_extra_ratio_list),
        'Average Ground Truth Mask Area': np.mean(gt_mask_area_list),
        'Average Prediction Mask Area': np.mean(pred_mask_area_list),
        'IoU 0-20%': iou_percentage[0],
        'IoU 21-40%': iou_percentage[1],
        'IoU 41-60%': iou_percentage[2],
        'IoU 61-80%': iou_percentage[3],
        'IoU 81-100%': iou_percentage[4]
    }

    total_area = sum(classwise_area.values())
    for cls in classwise_coverage_all:
        result[f'{cls} Coverage'] = np.mean(classwise_coverage_all[cls])
        result[f'{cls} Area Ratio'] = classwise_area[cls] / total_area if total_area > 0 else 0.0
        result[f'{cls} Box Ratio'] = classwise_count[cls] / sum(classwise_count.values()) if sum(classwise_count.values()) > 0 else 0.0

    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CRAFT Evaluation')
    parser.add_argument('--gold_folder', default='result/gold_labels', type=str, help='Gold file folder')
    parser.add_argument('--pred_folder', default='result/pred_labels', type=str, help='Prediction file folder')
    parser.add_argument('--img_folder', default='data/char_lvl/valid_images', type=str, help='image file folder')
    parser.add_argument('--scale', default=1, type=float, help='char box expanding scale')
    parser.add_argument('--iou_threshold', default=0.5, type=float, help='IoU threshold')
    args = parser.parse_args()

    result = evaluate_text_detection(args.gold_folder, args.pred_folder, args.img_folder, args.scale, iou_threshold=args.iou_threshold)
    print(result)
    print("\n".join(["{}:{:.2f}".format(k, v) for k, v in result.items()]))
