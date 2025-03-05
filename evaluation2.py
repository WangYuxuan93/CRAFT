import os
import numpy as np
import argparse
from skimage.draw import polygon
import matplotlib.pyplot as plt
import cv2

def expand_box(coords, scale=1.1, image_shape=None):
    """
    按比例扩展预测框。
    - coords: 形状为 (N,2) 的 numpy 数组，表示预测框的坐标。
    - scale: 扩展比例，默认为 1.1，即扩大 10%。
    """
    center = np.mean(coords, axis=0)
    expanded_coords = np.round(center + (coords - center) * scale).astype(int)
    
    # 限制坐标不超出边界
    if image_shape is not None:
        expanded_coords[:, 0] = np.clip(expanded_coords[:, 0], 0, image_shape[0] - 1)
        expanded_coords[:, 1] = np.clip(expanded_coords[:, 1], 0, image_shape[1] - 1)
    
    #print ("coords:", coords)
    #print ("expanded_coords:", expanded_coords)
    return expanded_coords

def calc_mask_iou(gt_coords_list, pred_coords_list, image_shape, scale=1.1, debug=False):
    """
    计算 mask 级别的 IoU，并统计 mask 的面积。
    
    参数：
    - gt_coords_list: 地面真值的多边形坐标列表，每个元素是一个 (N,2) 的数组
    - pred_coords_list: 预测框的多边形坐标列表，每个元素是一个 (N,2) 的数组
    
    返回：
    - mask_ious: 计算出的 IoU
    - gt_mask_area: 地面真值 mask 的面积
    - pred_mask_area: 预测 mask 的面积
    """
    def calculate_image_shape(coords_list):
        """
        计算图像的宽度和高度（image shape）。
        根据所有地面真值和预测框的坐标，找出最大和最小坐标来推断图像大小。
        """
        max_x, max_y = float('-inf'), float('-inf')

        for coords in coords_list:
            max_x = max(max_x, np.max(coords[:, 0]))
            max_y = max(max_y, np.max(coords[:, 1]))

        return int(max_x) + 1, int(max_y) + 1
    
    # 预处理预测框，按比例扩展
    pred_coords_list = [expand_box(coords, scale=scale, image_shape=image_shape) for coords in pred_coords_list]

    # 计算图像尺寸
    #image_shape = calculate_image_shape(gt_coords_list + pred_coords_list)

    #print ("image shape:", image_shape)
    # 创建空白 mask
    gt_mask = np.zeros(image_shape, dtype=np.uint8)
    pred_mask = np.zeros(image_shape, dtype=np.uint8)

    # 填充地面真值 mask
    for coords in gt_coords_list:
        rr, cc = polygon(coords[:, 0], coords[:, 1], image_shape)
        gt_mask[rr, cc] = 1
    
    # 填充预测框 mask
    for coords in pred_coords_list:
        rr, cc = polygon(coords[:, 0], coords[:, 1], image_shape)
        pred_mask[rr, cc] = 1
    
    if debug:
        # **可视化 mask**
        plt.figure(figsize=(10, 5))

        # 画地面真值 mask
        plt.subplot(1, 2, 1)
        plt.imshow(gt_mask.T, cmap='gray')
        plt.title("Ground Truth Mask")
        plt.axis('off')

        # 画预测 mask
        plt.subplot(1, 2, 2)
        plt.imshow(pred_mask.T, cmap='gray')
        plt.title("Prediction Mask")
        plt.axis('off')

        plt.show()

    # 计算 IoU
    intersection = np.logical_and(gt_mask, pred_mask).sum()
    union = np.logical_or(gt_mask, pred_mask).sum()
    
    # 计算 mask 面积
    gt_mask_area = np.sum(gt_mask)
    pred_mask_area = np.sum(pred_mask)

    # 计算未覆盖的比例
    uncovered_ratio = (gt_mask_area - intersection) / gt_mask_area if gt_mask_area > 0 else 0.0

    pred_extra_area = (pred_mask_area - intersection) / pred_mask_area if pred_mask_area > 0 else 0.0

    return (intersection / float(union) if union > 0 else 0.0, gt_mask_area, pred_mask_area, uncovered_ratio, pred_extra_area)
    #return (intersection / float(union) if union > 0 else 0.0, gt_mask_area, pred_mask_area, uncovered_ratio)


def read_txt_file(file_path, is_gold=True, debug=False):
    """ 读取txt文件的坐标数据 """
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            if len(line.strip()) == 0: 
                continue
            parts = line.strip().split(",")
            coords = np.array(list(map(float, parts[:8]))).reshape(4, 2)  # 解析坐标
            data.append(coords)
    return data

def read_folder(folder_path, is_gold=True):
    """ 读取文件夹中的所有txt文件，返回每个文件的坐标数据 """
    data_dict = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            data_dict[filename] = read_txt_file(file_path, is_gold)
    return data_dict


def evaluate_text_detection(gold_folder: str, pred_folder: str, img_folder:str, scale:float = 1.1, iou_threshold: float = 0.5, debug=True) -> dict:
    """
    计算文本检测的 IoU, Precision, Recall 和 F1 值，同时统计 mask 面积。
    """
    gold_data = read_folder(gold_folder, is_gold=True)
    pred_data = read_folder(pred_folder, is_gold=False)
    
    ious_list = []
    total_tp = 0
    gt_mask_area_list = []
    pred_mask_area_list = []
    uncovered_ratio_list = []
    pred_extra_ratio_list = []

    for img_id, gt_data in gold_data.items():
        pred_id = img_id.replace('gt_img', 'pred_img')
        if pred_id not in pred_data:
            print(f"预测文件 {pred_id} 不存在，跳过该图像。")
            continue

        pred_data_coords = pred_data[pred_id]
        if debug:
            print ("image file:", img_id)
        
        img_filename = img_id.replace('gt_img_', 'img_').replace('.txt', '.jpg')
        img_path = os.path.join(img_folder, img_filename)
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            image_shape = (img.shape[1], img.shape[0])  # (width, height)
            #print ("image shape:", image_shape)
        else:
            print(f"警告: 找不到图片 {img_path}，跳过该图像。")
            continue

        mask_iou, gt_area, pred_area, uncovered_ratio, pred_extra_ratio = calc_mask_iou(gt_data, pred_data_coords, image_shape=image_shape, scale=scale, debug=debug)
        ious_list.append(mask_iou)
        gt_mask_area_list.append(gt_area)
        pred_mask_area_list.append(pred_area)
        uncovered_ratio_list.append(uncovered_ratio)
        pred_extra_ratio_list.append(pred_extra_ratio)
        #total_gt_mask_area += gt_area
        #total_pred_mask_area += pred_area
        
        tp = int(mask_iou >= iou_threshold)
        total_tp += tp
    
    overall_acc = total_tp / len(gold_data) if len(gold_data) > 0 else 0.0

    result = {
        'Average IoU (per file)': np.mean(ious_list) if len(ious_list) > 0 else 0.0,
        'Overall Acc': overall_acc,
        'Average Uncovered Gold Area Ratio': np.mean(uncovered_ratio_list) if len(uncovered_ratio_list) > 0 else 0.0,
        'Average Prediction Extra Area Ratio': np.mean(pred_extra_ratio_list) if len(pred_extra_ratio_list) > 0 else 0.0,
        'Average Ground Truth Mask Area': np.mean(gt_mask_area_list) if len(gt_mask_area_list) > 0 else 0.0,
        'Average Prediction Mask Area': np.mean(pred_mask_area_list) if len(pred_mask_area_list) > 0 else 0.0,
    }

    return result


def calc_mask_iou_without_image_shape(gt_coords_list, pred_coords_list, debug=False):
    """
    计算 mask 级别的 IoU，并统计 mask 的面积。
    
    参数：
    - gt_coords_list: 地面真值的多边形坐标列表，每个元素是一个 (N,2) 的数组
    - pred_coords_list: 预测框的多边形坐标列表，每个元素是一个 (N,2) 的数组
    
    返回：
    - mask_ious: 计算出的 IoU
    - gt_mask_area: 地面真值 mask 的面积
    - pred_mask_area: 预测 mask 的面积
    """
    def calculate_image_shape(coords_list):
        """
        计算图像的宽度和高度（image shape）。
        根据所有地面真值和预测框的坐标，找出最大和最小坐标来推断图像大小。
        """
        max_x, max_y = float('-inf'), float('-inf')

        for coords in coords_list:
            max_x = max(max_x, np.max(coords[:, 0]))
            max_y = max(max_y, np.max(coords[:, 1]))

        return int(max_x) + 1, int(max_y) + 1

    # 计算图像尺寸
    image_shape = calculate_image_shape(gt_coords_list + pred_coords_list)

    #print ("image shape:", image_shape)
    # 创建空白 mask
    gt_mask = np.zeros(image_shape, dtype=np.uint8)
    pred_mask = np.zeros(image_shape, dtype=np.uint8)

    # 填充地面真值 mask
    for coords in gt_coords_list:
        rr, cc = polygon(coords[:, 0], coords[:, 1], image_shape)
        gt_mask[rr, cc] = 1
    
    # 填充预测框 mask
    for coords in pred_coords_list:
        rr, cc = polygon(coords[:, 0], coords[:, 1], image_shape)
        pred_mask[rr, cc] = 1
    
    if debug:
        # **可视化 mask**
        plt.figure(figsize=(10, 5))

        # 画地面真值 mask
        plt.subplot(1, 2, 1)
        plt.imshow(gt_mask.T, cmap='gray')
        plt.title("Ground Truth Mask")
        plt.axis('off')

        # 画预测 mask
        plt.subplot(1, 2, 2)
        plt.imshow(pred_mask.T, cmap='gray')
        plt.title("Prediction Mask")
        plt.axis('off')

        plt.show()

    # 计算 IoU
    intersection = np.logical_and(gt_mask, pred_mask).sum()
    union = np.logical_or(gt_mask, pred_mask).sum()
    
    # 计算 mask 面积
    gt_mask_area = np.sum(gt_mask)
    pred_mask_area = np.sum(pred_mask)

    # 计算未覆盖的比例
    uncovered_ratio = (gt_mask_area - intersection) / gt_mask_area if gt_mask_area > 0 else 0.0

    return (intersection / float(union) if union > 0 else 0.0, gt_mask_area, pred_mask_area, uncovered_ratio)

def eval_text_detection(gold_data, pred_data, iou_threshold=0.5):
    # 存储评估结果
    ious_list = []
    gt_mask_area_list = []
    pred_mask_area_list = []
    uncovered_ratio_list = []
    total_tp = 0

    # 遍历每个地面真值文件
    for gt_data, pred_data_coords in zip(gold_data, pred_data):

        mask_iou, gt_area, pred_area, uncovered_ratio = calc_mask_iou_without_image_shape(gt_data, pred_data_coords)
        ious_list.append(mask_iou)
        gt_mask_area_list.append(gt_area)
        pred_mask_area_list.append(pred_area)
        uncovered_ratio_list.append(uncovered_ratio)
        
        # 计算 Precision, Recall 和 F1 值
        tp = int(mask_iou >= iou_threshold)  # IoU 大于阈值为真阳性

        # 累加整体的TP, FP, FN
        total_tp += tp
    
    # 计算整体PRF
    overall_acc = total_tp / len(gold_data)

    # 汇总结果
    result = {
        'iou': np.mean(ious_list),
        'acc': overall_acc,
        'gt_mask_area': np.mean(gt_mask_area_list),
        'pred_mask_area': np.mean(pred_mask_area_list),
        'uncovered_ratio': np.mean(uncovered_ratio_list)
    }

    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CRAFT Evaluation')
    parser.add_argument('--gold_folder', default='result/gold_labels', type=str, help='Gold file folder')
    parser.add_argument('--pred_folder', default='result/pred_labels', type=str, help='Prediction file folder')
    parser.add_argument('--img_folder', default='data/char_lvl/valid_images', type=str, help='image file folder')
    parser.add_argument('--scale', default=1.1, type=float, help='char box expanding scale')
    parser.add_argument('--iou_threshold', default=0.5, type=float, help='IoU threshold')
    args = parser.parse_args()

    result = evaluate_text_detection(args.gold_folder, args.pred_folder, args.img_folder, args.scale, iou_threshold=args.iou_threshold)
    print(result)
    print("\n".join(["{}:{:.2f}".format(k, v) for k, v in result.items()]))
