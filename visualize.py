import cv2
import numpy as np

def resize_mask_to_input_size(mask, ratio, image, image_resized, debug=False):
    """
    根据给定的 ratio 放大 mask，并去除 padding 部分，使其与输入图像尺寸匹配。

    参数:
    - mask: 输入的二值化文本区域 mask。
    - ratio: 缩放比例的倒数。
    - target_h: 输入图像的高度。
    - target_w: 输入图像的宽度。

    返回:
    - resized_mask: 调整为与输入图像相同尺寸的 mask。
    """
    target_h32, target_w32 = image_resized.shape[:2]
    if debug:
        print ("target 32:", target_h32, target_w32)

    resized_mask32 = cv2.resize(mask, (target_w32, target_h32), interpolation=cv2.INTER_LINEAR)
    # 使用 ratio 的倒数来计算 mask 应该被放大的尺寸
    img_height, img_weight = image.shape[:2]
    target_h, target_w = int(img_height * ratio), int(img_weight * ratio)
    if debug:
        print ("target:", target_h, target_w)

    resized_mask_clean = resized_mask32[:target_h, :target_w]

    # 调整 mask 尺寸
    resized_mask = cv2.resize(resized_mask_clean, (img_weight, img_height), interpolation=cv2.INTER_LINEAR)

    #print (target_h, target_w, target_mask_w, target_mask_h, ratio)
    if debug:
        print (resized_mask.shape)

    # 去除 padding 部分，确保 mask 尺寸与输入图像一致
    #resized_mask = resized_mask[:target_h, :target_w]

    return resized_mask

def generate_text_mask(score_text, low_text, image, image_resized, ratio, interpolation=cv2.INTER_LINEAR, sigma=2):
    """
    生成基于文本得分矩阵的二值化 mask，低于 low_text 的像素将被过滤掉。

    参数:
    - score_text: 输入的文本得分矩阵，表示每个像素属于文本的概率，值在 [0, 1] 范围内。
    - low_text: 用于二值化的阈值，决定哪些区域是文本，哪些区域是背景。
    - input_image: 输入图像，用于确定输出 mask 的大小。
    - square_size: 输出图像的最大尺寸。
    - interpolation: 图像重采样时使用的插值方法。
    - sigma: 高斯滤波的标准差，用来平滑图像

    返回:
    - mask: 二值化后的文本区域 mask，文本区域为 255，其他区域为 0。
    - resized_mask: 调整为与输入图像相同尺寸的 mask。
    """
    # 对 text_score 进行高斯平滑
    blurred_text_score = cv2.GaussianBlur(score_text, (0, 0), sigma)

    # 使用 low_text 进行二值化，过滤低得分区域
    mask = np.where(blurred_text_score >= low_text, 255, 0).astype(np.uint8)

    # 调整 mask 的尺寸并去除 padding
    final_resized_mask = resize_mask_to_input_size(mask, ratio, image, image_resized)

    return final_resized_mask

def overlay_mask_on_image(input_image, text_mask, alpha=0.5):
    """
    将 text_mask 以透明度 alpha 叠加在 input_image 上。

    参数:
    - input_image: 原始图像。
    - text_mask: 文本区域 mask，值为 0 或 255。
    - alpha: 透明度，取值范围 0 到 1，默认是 0.5。

    返回:
    - output_image: 叠加后的图像。
    """
    # 确保 mask 是三通道图像，以便与输入图像叠加
    text_mask_colored = cv2.cvtColor(text_mask, cv2.COLOR_GRAY2BGR)

    # 叠加 mask 和输入图像
    overlay = cv2.addWeighted(input_image, 1 - alpha, text_mask_colored, alpha, 0)

    return overlay

"""
def overlay_boxes_on_image(image, boxes, alpha=0.5):
    # 确保图像为彩色（避免灰度图的错误绘制）
    if len(image.shape) == 2 or image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # 在图像上绘制框
    for box in boxes:
        poly = np.array(box).astype(np.int32).reshape((-1, 1, 2))  # 转换为多边形格式
        #print ("poly in draw:", poly)
        cv2.polylines(image, [poly], isClosed=True, color=(0, 0, 255), thickness=1)

    return image
"""

def overlay_boxes_on_image(image, boxes, alpha=0.5):
    if len(image.shape) == 2 or image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    for box in boxes:
        poly = np.array(box).astype(np.int32).reshape((-1, 1, 2))
        
        # 创建一层透明图层
        overlay = image.copy()

        # 先在 overlay 上画粗线
        cv2.polylines(overlay, [poly], isClosed=True, color=(0, 0, 255), thickness=2)

        # 将 overlay 叠加回原图，实现半透明效果
        image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    return image


def overlay_mask_and_boxes(input_image, mask, boxes, alpha=0.5):
    """
    将 mask 和预测框叠加到输入图像上。

    参数:
    - input_image: 原始图像。
    - mask: 生成的文本区域 mask，值为 0 或 255。
    - boxes: 检测的框，格式为 [num_boxes, 4, 2] 的多边形顶点坐标。
    - alpha: 透明度，默认为 0.5。

    返回:
    - overlay_image: 包含 mask 和框的叠加图像。
    """
    # 将 mask 转为彩色并与原图叠加
    mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    overlay_image = cv2.addWeighted(input_image, 1 - alpha, mask_colored, alpha, 0)

    # 在叠加图上绘制框
    overlay_image_with_boxes = overlay_boxes_on_image(overlay_image, boxes)

    return overlay_image_with_boxes