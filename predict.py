"""
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-
import os
import time
import argparse

from collections import OrderedDict
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import cv2
import numpy as np
from tqdm import tqdm

from utils import file_utils, craft_utils, imgproc

from net.craft import CRAFT
from merge import merge_boxes

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

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
                             craft_net=None,
                             zeroshot_craft_net=None,
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
                             merge_iou_threshold=0.7,
                             merge_cover_threshold=0.9,
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

    net = craft_net
    zeroshot_net = zeroshot_craft_net

    # 2. 读取图像
    if isinstance(image_input, str):
        image = imgproc.loadImage(image_input)
    elif isinstance(image_input, np.ndarray):
        image = image_input
    else:
        raise ValueError("image_input 应该是文件路径字符串或OpenCV图像")

    # 3. 推理
    bboxes, score_text, img_resized, target_ratio = infer_single_image(image=image,
                                                                        net=net,
                                                                        zeroshot_net=zeroshot_net,
                                                                        use_target_size=use_target_size,
                                                                        target_size=target_size,
                                                                        canvas_size=canvas_size,
                                                                        mag_ratio=mag_ratio,
                                                                        text_threshold=text_threshold,
                                                                        link_threshold=link_threshold,
                                                                        low_text=low_text,
                                                                        use_cuda=use_cuda,
                                                                        output_char_box=output_char_box,
                                                                        merge_iou_threshold=merge_iou_threshold,
                                                                        merge_cover_threshold=merge_cover_threshold,
                                                                        scale=scale)
    bboxes = [
        [ [int(round(x)), int(round(y))] for (x, y) in box ]
        for box in bboxes
    ]

    return bboxes, score_text, target_ratio, img_resized


def predict_legend_box(image, craft_net=None, scale=1, use_cuda=True):
    bboxes, score_text, target_ratio, img_resized = predict_image_with_boxes(
            image_input=image,
            craft_net=craft_net,
            zeroshot_craft_net=None,
            text_threshold=0.3,
            low_text=0.3,
            link_threshold=0.4,
            canvas_size=4096,
            mag_ratio=10,
            target_size=768,
            use_target_size=False,
            scale=scale,
            use_cuda=use_cuda,
            output_char_box=True,
            merge_iou_threshold=0.7,
            merge_cover_threshold=0.9
        )
    return bboxes

class craft_predictor(object):
    def __init__(self, model_path="model/after_zero-0430_main_map-bs16_8gpu-v2/finetuned_epoch_9_iter80.pth", use_cuda=True):
        self.model_path = model_path
        self.use_cuda = use_cuda

    def load_craft_model(self):
        try:
            self.craft_net = load_model(
                self.model_path,
                cuda=self.use_cuda)
            return True
        except:
            return False

    def predict_main_map_box(self, image, scale=1, use_cuda=True):
        try:
            assert self.craft_net is not None
        except:
            print ("Failed loading CRAFT model.")
        bboxes, score_text, target_ratio, img_resized = predict_image_with_boxes(
                image_input=image,
                craft_net=self.craft_net,
                zeroshot_craft_net=None,
                text_threshold=0.3,
                low_text=0.3,
                link_threshold=0.4,
                canvas_size=4096,
                mag_ratio=1,
                target_size=768,
                use_target_size=True,
                scale=scale,
                use_cuda=self.use_cuda,
                output_char_box=False,
                merge_iou_threshold=0.7,
                merge_cover_threshold=0.9
            )
        return bboxes

# ----------------- API -----------------

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

def safe_imwrite(filename, image):
    ext = os.path.splitext(filename)[1]
    success, buffer = cv2.imencode(ext, image)
    if success:
        buffer.tofile(filename)  # 正确处理中文路径
        return True
    else:
        print(f"[ERROR] Failed to encode image: {filename}")
        return False