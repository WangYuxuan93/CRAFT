import os
import numpy as np
from shapely.geometry import Polygon
from scipy.optimize import linear_sum_assignment
import argparse
from skimage.draw import polygon
import matplotlib.pyplot as plt

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

def calc_mask_iou(gt_polygons, pred_polygons, debug=False):
    """
    计算 mask 级别的 IoU。
    根据地面真值和预测框的多边形坐标计算图像大小，并生成 mask 进行 IoU 计算。
    
    参数：
    - gt_polygons: 地面真值的多边形列表
    - pred_polygons: 预测框的多边形列表
    
    返回：
    - mask_ious: 计算出来的每对多边形的 IoU
    - image_shape: 图像的形状
    """
    # 自动推断图像的尺寸
    def calculate_image_shape(gt_polygons, pred_polygons):
        """
        自动计算图像的宽度和高度（image shape）。
        根据所有地面真值和预测框的坐标，找出最大和最小坐标来推断图像大小。

        参数:
        - gt_polygons: 地面真值的多边形列表
        - pred_polygons: 预测框的多边形列表
        
        返回：
        - image_shape: (height, width) 图像的形状
        """
        max_x, max_y = float('-inf'), float('-inf')

        # 获取地面真值和预测框的坐标范围
        for poly in gt_polygons:
            max_x = max(max_x, *[coord[0] for coord in poly.exterior.coords])
            max_y = max(max_y, *[coord[1] for coord in poly.exterior.coords])

        for poly in pred_polygons:
            max_x = max(max_x, *[coord[0] for coord in poly.exterior.coords])
            max_y = max(max_y, *[coord[1] for coord in poly.exterior.coords])

        # 计算图像大小，假设至少包含所有框
        image_width = int(max_x) + 1
        image_height = int(max_y) + 1

        return (image_width, image_height)

    # 计算当前图像的尺寸
    image_shape = calculate_image_shape(gt_polygons, pred_polygons)

    # 创建空白的图像来表示 mask
    # 这里因为np初始化的时候0维是x轴，所以后面画的时候从横的变成竖的，但结果不影响
    gt_mask = np.zeros(image_shape, dtype=np.uint8)
    pred_mask = np.zeros(image_shape, dtype=np.uint8)
    #print ("image_shape:", image_shape)
    #print ("gt_polygons:", gt_polygons)
    #print ("pred_polygons:", pred_polygons)
    # 填充地面真值的 mask
    for poly in gt_polygons:
        if poly.is_valid:  # 如果是有效的多边形
            rr, cc = polygon(*zip(*list(poly.exterior.coords)))
            #print (rr, cc)
            rr, cc = rr[rr <= image_shape[0]], cc[cc <= image_shape[1]]  # 防止超出边界
            gt_mask[rr, cc] = 1
    
    # 填充预测框的 mask
    for poly in pred_polygons:
        if poly.is_valid:
            rr, cc = polygon(*zip(*list(poly.exterior.coords)))
            rr, cc = rr[rr <= image_shape[0]], cc[cc <= image_shape[1]]  # 防止超出边界
            pred_mask[rr, cc] = 1

    # 计算 mask 的交并比 (IoU)
    intersection = np.logical_and(gt_mask, pred_mask).sum()
    union = np.logical_or(gt_mask, pred_mask).sum()

    if debug:
        # 可视化 gt_mask 和 pred_mask
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(gt_mask.T, cmap='gray')
        axes[0].set_title('Ground Truth Mask')
        axes[0].axis('off')

        axes[1].imshow(pred_mask.T, cmap='gray')
        axes[1].set_title('Prediction Mask')
        axes[1].axis('off')
        plt.show()

    return intersection / float(union) if union > 0 else 0.0

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

        mask_iou = calc_mask_iou(gt_polygons, pred_polygons)
        ious_list.append(mask_iou)
        
        # 计算 Precision, Recall 和 F1 值
        tp = int(mask_iou >= iou_threshold)  # IoU 大于阈值为真阳性

        # 累加整体的TP, FP, FN
        total_tp += tp
    
    # 计算整体PRF
    overall_acc = total_tp / len(gold_data)

    # 汇总结果
    result = {
        'Average IoU (per file)': np.mean(ious_list),
        'Overall Acc': overall_acc
    }

    return result


def eval_text_detection(gold_data, pred_data, iou_threshold=0.5):
    # 存储评估结果
    ious_list = []
    total_tp = 0

    # 遍历每个地面真值文件
    for gt_data, pred_data_coords in zip(gold_data, pred_data):

        # 转换为多边形对象
        #print (gt_data)
        gt_polygons = [Polygon([coords[:2], coords[2:4], coords[4:6], coords[6:8]]) for coords in gt_data]
        pred_polygons = [Polygon([coords[:2], coords[2:4], coords[4:6], coords[6:8]]) for coords in pred_data_coords]

        mask_iou = calc_mask_iou(gt_polygons, pred_polygons)
        ious_list.append(mask_iou)
        
        # 计算 Precision, Recall 和 F1 值
        tp = int(mask_iou >= iou_threshold)  # IoU 大于阈值为真阳性

        # 累加整体的TP, FP, FN
        total_tp += tp
    
    # 计算整体PRF
    overall_acc = total_tp / len(gold_data)

    # 汇总结果
    result = {
        'iou': np.mean(ious_list),
        'acc': overall_acc
    }

    return result

if __name__ == '__main__':
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