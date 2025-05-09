from torchvision.ops import nms
import torch
import cv2
import numpy as np

def polygon_to_bbox(box):
    x, y, w, h = cv2.boundingRect(np.array(box).astype(np.int32))
    return [x, y, x + w, y + h]

def is_mostly_covered(inner_box, outer_box, cover_threshold=0.9):
    """
    判断 inner_box 是否有一定比例（如90%）被 outer_box 覆盖

    参数：
    - inner_box: 被测试是否被覆盖的多边形
    - outer_box: 另一个用于覆盖检测的多边形
    - cover_threshold: 被覆盖面积比例的阈值，默认0.9

    返回：
    - True：inner_box 超过阈值被 outer_box 覆盖
    - False：否则不认为被包含
    """
    inner = np.array(inner_box, dtype=np.float32)
    outer = np.array(outer_box, dtype=np.float32)

    # 计算相交区域
    retval, intersect_poly = cv2.intersectConvexConvex(inner, outer)
    if retval is None or retval <= 0:
        return False  # 没有交集

    # inner_box 的面积
    area_inner = cv2.contourArea(inner)
    if area_inner == 0:
        return False

    # 交集面积占比
    if retval / area_inner >= cover_threshold:
        return True
    else:
        return False

def merge_boxes(boxes1, boxes2, iou_threshold=0.7, cover_threshold=0.9):
    """
    合并两个模型的检测框：
    1. 删除完全包含的框（只保留大的）
    2. NMS 去重，主模型框优先（通过分数控制）

    返回最终框列表
    """
    all_boxes = list(boxes1) + list(boxes2)
    num_boxes = len(all_boxes)

    # 构造模型来源对应的得分：主模型分数高，Zero-Shot 分数低
    scores = [1.0] * len(boxes1) + [0.5] * len(boxes2)

    # Step 1: 过滤完全包含关系的框
    to_remove = set()
    for i in range(num_boxes):
        if i in to_remove:
            continue
        for j in range(num_boxes):
            if i == j or j in to_remove:
                continue
            box_i = all_boxes[i]
            box_j = all_boxes[j]

            if is_mostly_covered(box_i, box_j, cover_threshold=cover_threshold):
                to_remove.add(i)
            elif is_mostly_covered(box_j, box_i, cover_threshold=cover_threshold):
                to_remove.add(j)

    filtered_boxes = [box for idx, box in enumerate(all_boxes) if idx not in to_remove]
    filtered_scores = [score for idx, score in enumerate(scores) if idx not in to_remove]

    if not filtered_boxes:
        return []

    # Step 2: 执行 NMS（主模型得分高 → 优先保留）
    rect_boxes = [polygon_to_bbox(box) for box in filtered_boxes]
    boxes_tensor = torch.tensor(rect_boxes, dtype=torch.float32)
    scores_tensor = torch.tensor(filtered_scores, dtype=torch.float32)

    keep_indices = nms(boxes_tensor, scores_tensor, iou_threshold)
    final_boxes = [filtered_boxes[i] for i in keep_indices]

    return final_boxes