import os
import numpy as np
from shapely.geometry import Polygon
from scipy.optimize import linear_sum_assignment
import argparse

def calc_iou(gt_poly, pred_poly):
    """计算两多边形之间的IoU"""
    if gt_poly.intersects(pred_poly) and gt_poly.area > 0 and pred_poly.area > 0:
        intersection = gt_poly.intersection(pred_poly).area
        union = gt_poly.union(pred_poly).area
        return intersection / union
    return 0.0

def read_txt_file(file_path, is_gold=True, debug=False):
    """
    从txt文件读取坐标数据，每行包含四个角的坐标，并且：
    - 对于地面真值（`gold`），还包含语言和文本字段。
    - 对于预测结果（`pred`），只有坐标字段。
    
    参数:
    - file_path: txt文件的路径
    - is_gold: 是否为地面真值数据，如果是则包含语言和字母文本，否则仅包含坐标
    
    返回:
    - data: 包含坐标数据的列表，每个元素是一个四元组（坐标）和语言/文本（如果是`gold`）
    """
    data = []
    if debug:
        print (file_path)
    with open(file_path, 'r') as f:
        for line in f:
            if len(line.strip())==0: continue
            parts = line.strip().split(",")
            if debug:
                print (parts)
            coords = list(map(float, parts[:8]))  # 前8个是坐标
            if is_gold:
                lang = parts[8]  # 语言
                text = parts[9]  # 文本
                data.append(coords)
            else:
                data.append(coords)  # 预测框只包含坐标
    return data

def read_folder(folder_path, is_gold=True):
    """
    读取文件夹中的所有txt文件，返回每个文件对应的坐标数据。
    
    参数:
    - folder_path: 文件夹路径
    - is_gold: 是否为地面真值数据
    
    返回:
    - data_dict: 文件名为键，坐标数据为值的字典
    """
    data_dict = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            # 使用文件名中的id作为键
            data_dict[filename] = read_txt_file(file_path, is_gold)
    return data_dict

def evaluate_text_detection(gold_folder: str, pred_folder: str, iou_threshold: float = 0.5, debug=False) -> dict:
    """
    计算文本检测的IoU, Precision, Recall 和 F1值。
    
    Arguments:
    - gold_folder : 地面真值文件夹路径
    - pred_folder : 预测结果文件夹路径
    
    Returns:
    - result : dict, 包括 IoU, Precision, Recall 和 F1值
    """
    # 读取地面真值和预测框的数据
    gold_data = read_folder(gold_folder, is_gold=True)
    pred_data = read_folder(pred_folder, is_gold=False)
    
    # 存储评估结果
    ious_list = []
    precision_list = []
    recall_list = []
    f1_list = []

    # 统计整体的真阳性、假阳性和假阴性
    total_tp = total_fp = total_fn = 0

    # 遍历每个地面真值文件
    for img_id, gt_data in gold_data.items():
        # 对应的预测结果文件名
        pred_id = img_id.replace('gt_img', 'pred_img')
        
        if pred_id not in pred_data:
            print(f"预测文件 {pred_id} 不存在，跳过该图像。")
            continue

        # 获取对应的预测框坐标
        pred_data_coords = pred_data[pred_id]
        
        # 转换为多边形对象
        #print (gt_data)
        gt_polygons = [Polygon([coords[:2], coords[2:4], coords[4:6], coords[6:8]]) for coords in gt_data]
        pred_polygons = [Polygon([coords[:2], coords[2:4], coords[4:6], coords[6:8]]) for coords in pred_data_coords]

        # 计算每对地面真值和预测框的IoU
        ious = np.zeros((len(gt_polygons), len(pred_polygons)), dtype=np.float64)
        for i, gt_poly in enumerate(gt_polygons):
            for j, pred_poly in enumerate(pred_polygons):
                ious[i, j] = calc_iou(gt_poly, pred_poly)
        if debug:
            print ("IoU:", ious)
        # 计算得分矩阵（如果得分与IoU相等，可以直接用IoU）
        scores = ious.copy()  # 使用IoU作为得分，也可以根据需要替换为其他得分计算方法

        # 定义一个可行性矩阵（允许所有匹配）
        allowed = np.ones_like(ious, dtype=bool)

        # 使用匈牙利算法进行匹配
        row_ind, col_ind = linear_sum_assignment(scores, maximize=True)
        if debug:
            print (f"row_ind: {row_ind}, col_ind: {col_ind}")

        # 找出匹配的地面真值和预测框
        matches_gt = row_ind
        matches_pred = col_ind
        matches_ious = ious[matches_gt, matches_pred]
        #print (matches_ious)
        if debug:
            print (f"matches_ious: {matches_ious}")

        # 统计真阳性(TP), 假阳性(FP), 假阴性(FN)
        tp = np.sum(matches_ious >= iou_threshold)  # 真阳性数目
        fp = len(pred_polygons) - tp     # 假阳性数目
        fn = len(gt_polygons) - tp       # 假阴性数目
        if debug:
            print (f"tp: {tp}")

        # 计算Precision, Recall和F1值
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # 存储每个图像的评估结果
        if len(matches_ious) > 0:
            ious_list.append(np.mean(matches_ious))
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

        # 累加整体的TP, FP, FN
        total_tp += tp
        total_fp += fp
        total_fn += fn
    
    # 计算整体PRF
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0

    # 汇总结果
    result = {
        'Average IoU (per file)': np.mean(ious_list),
        'Average Precision (per file)': np.mean(precision_list),
        'Average Recall (per file)': np.mean(recall_list),
        'Average F1 (per file)': np.mean(f1_list),
        'Overall Precision': overall_precision,
        'Overall Recall': overall_recall,
        'Overall F1': overall_f1
    }

    return result


parser = argparse.ArgumentParser(description='CRAFT Evaluation')
parser.add_argument('--gold_folder', default='result/gold_labels', type=str, help='Gold file folder')
parser.add_argument('--pred_folder', default='result/pred_labels', type=str, help='Prediction file folder')
parser.add_argument('--iou_threshold', default=0.5, type=float, help='test interval')
args = parser.parse_args()


# 计算并输出结果
result = evaluate_text_detection(args.gold_folder, args.pred_folder, iou_threshold=args.iou_threshold)
print(result)
info = "\n".join(["{}:{:.2f}".format(x,y) for x, y in result.items()])
print (info)