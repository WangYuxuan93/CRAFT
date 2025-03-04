import os
import numpy as np
import argparse
from skimage.draw import polygon
import matplotlib.pyplot as plt

def calc_mask_iou(gt_coords_list, pred_coords_list, debug=False):
    """
    计算 mask 级别的 IoU，替代 shapely 的多边形计算。
    
    参数：
    - gt_coords_list: 地面真值的多边形坐标列表，每个元素是一个 (N,2) 的数组
    - pred_coords_list: 预测框的多边形坐标列表，每个元素是一个 (N,2) 的数组
    
    返回：
    - mask_ious: 计算出的 IoU
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

    
    #print ("gt_coords_list:", gt_coords_list)
    #print ("pred_coords_list:", pred_coords_list)

    # 计算图像尺寸
    image_shape = calculate_image_shape(gt_coords_list + pred_coords_list)
    #print ("image_shape:", image_shape)

    # 创建空白 mask
    gt_mask = np.zeros(image_shape, dtype=np.uint8)
    pred_mask = np.zeros(image_shape, dtype=np.uint8)

    # 填充地面真值 mask
    for coords in gt_coords_list:
        #rr, cc = polygon(coords[:, 1], coords[:, 0], image_shape)
        rr, cc = polygon(coords[:, 0], coords[:, 1], image_shape)
        #print (rr, cc)
        gt_mask[rr, cc] = 1
    #exit()
    
    # 填充预测框 mask
    for coords in pred_coords_list:
        #rr, cc = polygon(coords[:, 1], coords[:, 0], image_shape)
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

    if debug:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(gt_mask.T, cmap='gray')
        axes[0].set_title('Ground Truth Mask')
        axes[0].axis('off')

        axes[1].imshow(pred_mask.T, cmap='gray')
        axes[1].set_title('Prediction Mask')
        axes[1].axis('off')
        plt.show()

    return intersection / float(union) if union > 0 else 0.0


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


def evaluate_text_detection(gold_folder: str, pred_folder: str, iou_threshold: float = 0.5, debug=False) -> dict:
    """
    计算文本检测的IoU, Precision, Recall 和 F1值
    """
    gold_data = read_folder(gold_folder, is_gold=True)
    pred_data = read_folder(pred_folder, is_gold=False)
    
    ious_list = []
    total_tp = 0

    for img_id, gt_data in gold_data.items():
        pred_id = img_id.replace('gt_img', 'pred_img')
        if pred_id not in pred_data:
            print(f"预测文件 {pred_id} 不存在，跳过该图像。")
            continue

        pred_data_coords = pred_data[pred_id]

        mask_iou = calc_mask_iou(gt_data, pred_data_coords)
        ious_list.append(mask_iou)
        
        tp = int(mask_iou >= iou_threshold)
        total_tp += tp
    
    overall_acc = total_tp / len(gold_data) if len(gold_data) > 0 else 0.0

    result = {
        'Average IoU (per file)': np.mean(ious_list) if len(ious_list) > 0 else 0.0,
        'Overall Acc': overall_acc
    }

    return result


def eval_text_detection(gold_data, pred_data, iou_threshold=0.5):
    # 存储评估结果
    ious_list = []
    total_tp = 0

    # 遍历每个地面真值文件
    for gt_data, pred_data_coords in zip(gold_data, pred_data):

        mask_iou = calc_mask_iou(gt_data, pred_data_coords)
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
    parser.add_argument('--iou_threshold', default=0.5, type=float, help='IoU threshold')
    args = parser.parse_args()

    result = evaluate_text_detection(args.gold_folder, args.pred_folder, iou_threshold=args.iou_threshold)
    print(result)
    print("\n".join(["{}:{:.2f}".format(k, v) for k, v in result.items()]))
