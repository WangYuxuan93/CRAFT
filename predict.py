"""
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-
import os
import time
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import cv2
import numpy as np
from tqdm import tqdm

from utils import file_utils, craft_utils, imgproc

from net.craft import CRAFT
from eval import copyStateDict
from evaluation2 import read_txt_file
from torchvision.ops import nms

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

def test_net(net, image, text_threshold, link_threshold, low_text, cuda):
    t0 = time.time()

    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, args.canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=args.mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    # forward pass
    with torch.no_grad():
        y, _ = net(x)

    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    boxes = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text)
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)

    t1 = time.time() - t1

    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    if args.show_time : print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return boxes, ret_score_text

def test_net_v2(net, image, text_threshold, link_threshold, low_text, cuda, canvas_size, mag_ratio, refine_net=None, output_char_box=True, debug=False):
    t0 = time.time()

    # resize
    if debug:
        print ("original img size:", image.shape)
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio
    if debug:
        print ("img_resized:", img_resized.shape)

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    if debug:
        print ("x:", x.shape)
    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

    # refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0,:,:,0].cpu().data.numpy()

    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    boxes = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, output_char_box=output_char_box)
    if debug:
        print ("score_text:", score_text.shape)
    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)

    t1 = time.time() - t1

    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    #if args.show_time : print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return boxes, ret_score_text, score_text, target_ratio, img_resized

def test_net_v3(net, image, text_threshold, link_threshold, low_text, cuda, target_size=768, refine_net=None, output_char_box=True, debug=False):
    t0 = time.time()

    # resize
    if debug:
        print ("original img size:", image.shape)
    #img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=mag_ratio)
    target_w, target_h = target_size, target_size
    img_resized = cv2.resize(image, (target_w, target_h), interpolation = cv2.INTER_LINEAR)
    ratio_h = image.shape[0] / target_h
    ratio_w = image.shape[1] / target_w
    target_ratio = ratio_h
    #ratio_h = ratio_w = 1 / target_ratio
    if debug:
        print ("img_resized:", img_resized.shape)

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    if debug:
        print ("x:", x.shape)
    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

    # refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0,:,:,0].cpu().data.numpy()

    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    boxes = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, output_char_box=output_char_box)
    if debug:
        print ("score_text:", score_text.shape)
    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)

    t1 = time.time() - t1

    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    #if args.show_time : print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return boxes, ret_score_text, score_text, target_ratio, img_resized

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

def load_model(model_path, cuda=False):
    net = CRAFT()
    #checkpoint = torch.load(model_path, map_location=device)
    checkpoint = torch.load(model_path)

    if 'model_state_dict' in checkpoint:
        # 去掉 "module." 前缀
        new_state_dict = {}
        for k, v in checkpoint['model_state_dict'].items():
            if k.startswith("module."):
                new_state_dict[k[7:]] = v  # 去掉 "module."
            else:
                new_state_dict[k] = v
        
        if cuda:
            # 加载去掉 "module." 的 state_dict
            net.load_state_dict(new_state_dict)
            net = net.cuda()
        else:
            net.load_state_dict(new_state_dict)
    else:
        if cuda:
            net.load_state_dict(copyStateDict(torch.load(model_path)))
            net = net.cuda()
            net = torch.nn.DataParallel(net)
            cudnn.benchmark = False
        else:
            net.load_state_dict(copyStateDict(torch.load(model_path, map_location='cpu')))
    net.eval()
    return net

def load_models(trained_model_path, zeroshot_model_path=None, use_cuda=False):
    """
    加载主模型和可选的 zero-shot 模型。

    参数：
    - trained_model_path: 主模型路径
    - zeroshot_model_path: zero-shot 模型路径（可选）
    - use_cuda: 是否使用 GPU

    返回：
    - net: 主模型
    - zeroshot_net: zero-shot 模型（或 None）
    """
    print(f"Loading main model from {trained_model_path}")
    net = load_model(trained_model_path, cuda=use_cuda)

    zeroshot_net = None
    if zeroshot_model_path:
        print(f"Loading zero-shot model from {zeroshot_model_path}")
        zeroshot_net = load_model(zeroshot_model_path, cuda=use_cuda)

    return net, zeroshot_net

def load_text_detect(images_path, labels_path):
    image_names = os.listdir(images_path)
    label_names = os.listdir(labels_path)
    image_names.sort()
    label_names.sort()
    return image_names, label_names

def inference(net, test_folder, text_threshold=0.5, low_text=0.4, link_threshold=0.4, cuda=False, canvas_size=1280, mag_ratio=1.5):
    images_path = os.path.join(test_folder, "valid_images")
    labels_path = os.path.join(test_folder, "valid_labels")
    image_names, label_names = load_text_detect(images_path, labels_path)

    net.eval()
    pred_bbox_list = []
    gold_bbox_list = []
    for gold_path, image_path in tqdm(zip(label_names, image_names), total=len(image_names), desc="Processing", ncols=80):
        #print("Test image {:d}/{:d}: {:s}".format(k+1, len(image_list), image_path), end='\r')
        image = imgproc.loadImage(os.path.join(images_path, image_path))

        bboxes, ret_score_text, score_text, target_ratio, img_resized = test_net_v2(net, image, text_threshold, link_threshold, low_text, cuda, canvas_size, mag_ratio)
        #print ("bboxes:", bboxes)
        pred_bbox = []
        for i, box in enumerate(bboxes):
            poly = np.array(box).astype(np.int32)#.reshape((-1))
            pred_bbox.append(poly)
        #print ("pred bbox:", pred_bbox)
        pred_bbox_list.append(pred_bbox)
        gold_bbox = read_txt_file(os.path.join(labels_path, gold_path), is_gold=True)
        #print ("gold_bbox:", gold_bbox)
        gold_bbox_list.append(gold_bbox)
    
    return gold_bbox_list, pred_bbox_list

def expand_box(coords, scale=1.1, image_shape=None):
    """
    按比例扩展预测框。
    - coords: 形状为 (N,2) 的 numpy 数组，表示预测框的坐标。
    - scale: 扩展比例，默认为 1.1，即扩大 10%。
    """
    center = np.mean(coords, axis=0)
    #print ("image shape:", image_shape)
    #print ("center:", center)
    #print ("(coords - center) * scale:",(coords - center) * scale)
    #print ("center + (coords - center) * scale:", center + (coords - center) * scale)
    expanded_coords = np.round(center + (coords - center) * scale).astype(int)
    
    # 限制坐标不超出边界
    if image_shape is not None:
        expanded_coords[:, 0] = np.clip(expanded_coords[:, 0], 0, image_shape[1] - 1)
        expanded_coords[:, 1] = np.clip(expanded_coords[:, 1], 0, image_shape[0] - 1)
    
    #print ("coords:", coords)
    #print ("expanded_coords:", expanded_coords)
    #exit()
    return expanded_coords

# ----------------- API -----------------

def predict_image_with_boxes(image_input,
                             model=None,
                             model_path='final_net_param.pth',
                             text_threshold=0.3,
                             low_text=0.3,
                             link_threshold=0.4,
                             canvas_size=4096,
                             mag_ratio=1.5,
                             target_size=768,
                             use_target_size=False,
                             scale=1.0,
                             use_cuda=False,
                             output_char_box=True,
                             debug=False):
    """
    单张图片文本检测接口，返回检测框。
    
    参数:
    - image_input: 图像路径或 OpenCV 图像 (numpy.ndarray)
    - model_path: 模型路径
    - 其他参数参考 argparse
    
    返回:
    - boxes: 检测框列表，每个框是形如 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] 的点集
    """
    if model is None:
        # 1. 加载模型
        device = 'cuda' if use_cuda and torch.cuda.is_available() else 'cpu'
        net = CRAFT()
        print('Loading weights from checkpoint (' + model_path + ')')
        
        net = load_model(model_path, net, cuda=use_cuda)
        net.eval()
    else:
        net = model

    # 2. 读取图像
    if isinstance(image_input, str):
        image = imgproc.loadImage(image_input)
    elif isinstance(image_input, np.ndarray):
        image = image_input
    else:
        raise ValueError("image_input 应该是文件路径字符串或OpenCV图像")

    # 3. 推理
    if use_target_size:
        bboxes, ret_score_text, score_text, target_ratio, img_resized = test_net_v3(net, image, text_threshold, link_threshold, low_text, use_cuda, target_size, output_char_box=output_char_box, debug=debug)
    else:
        bboxes, ret_score_text, score_text, target_ratio, img_resized = test_net_v2(net, image, text_threshold, link_threshold, low_text, use_cuda, canvas_size, mag_ratio, output_char_box=output_char_box, debug=debug)

    # 4. 扩展 box（如果需要）
    if scale != 1:
        image_shape = image.shape
        bboxes = [expand_box(coords, scale=scale, image_shape=image_shape) for coords in bboxes]

    return bboxes, ret_score_text, score_text, target_ratio, img_resized


def predict_legend_box(image, model_path="model/td-bs8_8gpu-v1/finetuned_epoch_9_iter700.pth", scale=1, use_cuda=True):
    bboxes, ret_score_text, score_text, target_ratio, img_resized = predict_image_with_boxes(
            image_input=image,
            model=None,
            model_path=model_path,
            text_threshold=0.3,
            low_text=0.3,
            link_threshold=0.4,
            canvas_size=4096,
            mag_ratio=10,
            target_size=768,
            use_target_size=False,
            scale=scale,
            use_cuda=use_cuda
        )
    return bboxes

def predict_main_map_box(image, model_path="model/main_map-bs8_8gpu-v1/finetuned_epoch_7_iter460.pth", scale=1, use_cuda=True):
    bboxes, ret_score_text, score_text, target_ratio, img_resized = predict_image_with_boxes(
            image_input=image,
            model=None,
            model_path=model_path,
            text_threshold=0.3,
            low_text=0.3,
            link_threshold=0.4,
            canvas_size=4096,
            mag_ratio=1,
            target_size=768,
            use_target_size=True,
            scale=scale,
            use_cuda=use_cuda
        )
    return bboxes

# ----------------- API -----------------

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


def infer_single_image(
    image,
    net,
    zeroshot_net,
    use_target_size=False,
    target_size=768,
    canvas_size=1280,
    mag_ratio=1.5,
    text_threshold=0.7,
    link_threshold=0.4,
    low_text=0.4,
    use_cuda=False,
    output_char_box=True,
    merge_iou_threshold=0.7,
    merge_cover_threshold=0.9,
    scale=1.0
):
    """
    使用主模型和（可选）Zero-shot 模型对单张图像进行推理。

    返回：
    - bboxes: 合并后的检测框
    - score_text: 融合得分图
    - img_resized: 缩放后的图像
    - target_ratio: 缩放比率
    """
    if use_target_size:
        bboxes1, _, score_text1, target_ratio, img_resized = test_net_v3(
            net, image, text_threshold, link_threshold, low_text,
            use_cuda, target_size, output_char_box=output_char_box)

        bboxes2 = []
        if zeroshot_net is not None:
            bboxes2, _, score_text2, _, _ = test_net_v3(
                zeroshot_net, image, text_threshold, link_threshold, low_text,
                use_cuda, target_size, output_char_box=output_char_box)
    else:
        bboxes1, _, score_text1, target_ratio, img_resized = test_net_v2(
            net, image, text_threshold, link_threshold, low_text,
            use_cuda, canvas_size, mag_ratio, output_char_box=output_char_box)

        bboxes2 = []
        if zeroshot_net is not None:
            bboxes2, _, score_text2, _, _ = test_net_v2(
                zeroshot_net, image, text_threshold, link_threshold, low_text,
                use_cuda, canvas_size, mag_ratio, output_char_box=output_char_box)

    # 合并框
    bboxes = merge_boxes(bboxes1, bboxes2, iou_threshold=merge_iou_threshold, cover_threshold=merge_cover_threshold)
    score_text = np.maximum(score_text1, score_text2) if zeroshot_net is not None else score_text1

    # 扩框（可选）
    if scale != 1:
        bboxes = [expand_box(box, scale=scale, image_shape=image.shape) for box in bboxes]

    return bboxes, score_text, img_resized, target_ratio



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CRAFT Text Detection')
    parser.add_argument('--trained_model', default='final_net_param.pth', type=str, help='pretrained model')
    parser.add_argument('--text_threshold', default=0.3, type=float, help='text confidence threshold')
    parser.add_argument('--low_text', default=0.3, type=float, help='text low-bound score')
    parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
    parser.add_argument('--cuda', default=False, type=str2bool, help='Use cuda to train model')
    parser.add_argument('--canvas_size', default=4096, type=int, help='image size for inference')
    parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')
    parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
    parser.add_argument('--test_folder', default='/home/brooklyn/ICDAR/icdar2013/test_images/', type=str, help='folder path to input images')
    parser.add_argument('--result_folder', default='./result/', type=str, help='folder path to save result images')
    parser.add_argument('--output_folder', default='./result/pred_labels', type=str, help='folder path to save prediction file')
    parser.add_argument('--only_pred_file', default=False, action='store_true', help='Only output prediction file to output folder')
    parser.add_argument('--target_size', default=768, type=int, help='image size for inference')
    parser.add_argument('--use_target_size', default=False, type=str2bool, help='resize the image to target size')
    parser.add_argument('--scale', default=1, type=float, help='box expanding scale')
    parser.add_argument('--output_word_box', default=False, action='store_true', help='output word bbox')
    args = parser.parse_args()

    output_char_box = not args.output_word_box
    
    """ For test images in a folder """
    image_list, _, _ = file_utils.get_files(args.test_folder)
    print (image_list)
    #测试结果保存路径
    result_folder = args.result_folder
    if not os.path.isdir(result_folder):
        os.mkdir(result_folder)

    print (args.only_pred_file)
    # load net
    net = CRAFT()     # initialize
    print('Loading weights from checkpoint (' + args.trained_model + ')')
    
    net = load_model(args.trained_model, net, cuda=args.cuda)
    net.eval()

    t = time.time()
    #print("net.eval")
    #print(image_list)
    # load data
    for image_path in tqdm(image_list):
        #print("Test image {:d}/{:d}: {:s}".format(k+1, len(image_list), image_path), end='\r')
        image = imgproc.loadImage(image_path)

        #bboxes, polys, score_text = test_net(net, image, args.text_threshold, args.link_threshold, args.low_text, args.cuda, args.poly, refine_net)
        #if args.use_target_size:
        #    bboxes, ret_score_text, score_text, target_ratio, img_resized = test_net_v3(net, image, args.text_threshold, args.link_threshold, args.low_text, args.cuda, args.target_size)
        #else:
        #    bboxes, ret_score_text, score_text, target_ratio, img_resized = test_net_v2(net, image, args.text_threshold, args.link_threshold, args.low_text, args.cuda, args.canvas_size, args.mag_ratio)
        
        #if args.scale != 1:
        #    image_shape = image.shape
            #print ("image shape:",image_shape)
            #print ("origin bboxes:",bboxes)
        #    bboxes = [expand_box(coords, scale=args.scale, image_shape=image_shape) for coords in bboxes]
            #print ("expanded bboxes:",bboxes)

        bboxes, ret_score_text, score_text, target_ratio, img_resized = predict_image_with_boxes(
            image_input=image,
            model=net,
            model_path=args.trained_model,
            text_threshold=args.text_threshold,
            low_text=args.low_text,
            link_threshold=args.link_threshold,
            canvas_size=args.canvas_size,
            mag_ratio=args.mag_ratio,
            target_size=args.target_size,
            use_target_size=args.use_target_size,
            scale=args.scale,
            use_cuda=args.cuda,
            output_char_box=output_char_box
        )
        if not args.only_pred_file:
            # save score text
            filename, file_ext = os.path.splitext(os.path.basename(image_path))

            real_mask = generate_text_mask(score_text, args.low_text, image, img_resized, target_ratio)
            real_mask_file = result_folder + "/" + filename + '_mask.png'
        
            cv2.imwrite(real_mask_file, real_mask)

            #overlay_image = overlay_mask_on_image(image, real_mask, alpha=0.5)
            #overlay_file = result_folder + "/overlay_" + filename + '_mask.jpg'
            #cv2.imwrite(overlay_file, overlay_image)

            #heatmap_overlay_image = overlay_mask_on_image(image, real_mask, alpha=0.5)
            #heatmap_overlay_file = result_folder + "/" + filename + '_mask_overlay.jpg'
            #cv2.imwrite(heatmap_overlay_file, heatmap_overlay_image)

            box_image = overlay_boxes_on_image(image, bboxes)
            box_image_file = result_folder + "/" + filename + '_box_overlay.jpg'
            cv2.imwrite(box_image_file, box_image)

            mask_file = result_folder + "/res_" + filename + '_heatmap.jpg'
            cv2.imwrite(mask_file, ret_score_text)

            mask_and_box_image = overlay_mask_and_boxes(image, real_mask, bboxes, alpha=0.5)
            mask_and_box_image_file = result_folder + "/" + filename + '_mask_and_box_overlay.jpg'
            cv2.imwrite(mask_and_box_image_file, mask_and_box_image)

        file_utils.saveResult(image_path, image[:,:,::-1], bboxes, dirname=args.output_folder)

    print("elapsed time : {}s".format(time.time() - t))
