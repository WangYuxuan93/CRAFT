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

def test_net_v2(net, image, text_threshold, link_threshold, low_text, cuda, canvas_size, mag_ratio, refine_net=None, debug=False):
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
    boxes = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text)
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

def test_net_v3(net, image, text_threshold, link_threshold, low_text, cuda, target_size=768, refine_net=None, debug=False):
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
    boxes = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text)
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

def overlay_boxes_on_image(image, boxes, alpha=0.5):
    """
    将预测的框叠加到图像上。

    参数:
    - image: 输入的原始图像。
    - boxes: 检测的框，格式为 [num_boxes, 4, 2] 的多边形顶点坐标。
    - alpha: 透明度，默认为 0.5。

    返回:
    - image_with_boxes: 绘制了框的图像。
    """
    # 确保图像为彩色（避免灰度图的错误绘制）
    if len(image.shape) == 2 or image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # 在图像上绘制框
    for box in boxes:
        poly = np.array(box).astype(np.int32).reshape((-1, 1, 2))  # 转换为多边形格式
        cv2.polylines(image, [poly], isClosed=True, color=(0, 0, 255), thickness=1)

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

def load_model(model_path, net, device="cpu"):
    checkpoint = torch.load(model_path, map_location=device)
    # 去掉 "module." 前缀
    new_state_dict = {}
    for k, v in checkpoint['model_state_dict'].items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v  # 去掉 "module."
        else:
            new_state_dict[k] = v

    # 加载去掉 "module." 的 state_dict
    net.load_state_dict(new_state_dict)
    return net

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CRAFT Text Detection')
    parser.add_argument('--trained_model', default='final_net_param.pth', type=str, help='pretrained model')
    parser.add_argument('--text_threshold', default=0.5, type=float, help='text confidence threshold')
    parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
    parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
    parser.add_argument('--cuda', default=False, type=str2bool, help='Use cuda to train model')
    parser.add_argument('--canvas_size', default=1280, type=int, help='image size for inference')
    parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')
    parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
    parser.add_argument('--test_folder', default='/home/brooklyn/ICDAR/icdar2013/test_images/', type=str, help='folder path to input images')
    parser.add_argument('--result_folder', default='./result/', type=str, help='folder path to save result images')
    parser.add_argument('--output_folder', default='./result/pred_labels', type=str, help='folder path to save prediction file')
    parser.add_argument('--only_pred_file', default=False, action='store_true', help='Only output prediction file to output folder')
    parser.add_argument('--target_size', default=768, type=int, help='image size for inference')
    parser.add_argument('--use_target_size', default=False, type=str2bool, help='resize the image to target size')
    args = parser.parse_args()


    """ For test images in a folder """
    image_list, _, _ = file_utils.get_files(args.test_folder)

    #测试结果保存路径
    result_folder = args.result_folder
    if not os.path.isdir(result_folder):
        os.mkdir(result_folder)

    print (args.only_pred_file)
    # load net
    net = CRAFT()     # initialize
    print('Loading weights from checkpoint (' + args.trained_model + ')')
    
    checkpoint = torch.load(args.trained_model)
    if 'model_state_dict' in checkpoint:
        if args.cuda:
            net = load_model(args.trained_model, net)
            net = net.cuda()
        else:
            net = load_model(args.trained_model, net, device="cpu")
    else:
        if args.cuda:
            net.load_state_dict(copyStateDict(torch.load(args.trained_model)))
            net = net.cuda()
            net = torch.nn.DataParallel(net)
            cudnn.benchmark = False
        else:
            net.load_state_dict(copyStateDict(torch.load(args.trained_model, map_location='cpu')))

    net.eval()
    t = time.time()
    #print("net.eval")
    #print(image_list)
    # load data
    for k, image_path in enumerate(image_list):
        #print("Test image {:d}/{:d}: {:s}".format(k+1, len(image_list), image_path), end='\r')
        image = imgproc.loadImage(image_path)

        #bboxes, polys, score_text = test_net(net, image, args.text_threshold, args.link_threshold, args.low_text, args.cuda, args.poly, refine_net)
        if args.use_target_size:
            bboxes, ret_score_text, score_text, target_ratio, img_resized = test_net_v3(net, image, args.text_threshold, args.link_threshold, args.low_text, args.cuda, args.target_size)
        else:
            bboxes, ret_score_text, score_text, target_ratio, img_resized = test_net_v2(net, image, args.text_threshold, args.link_threshold, args.low_text, args.cuda, args.canvas_size, args.mag_ratio)
        
        if not args.only_pred_file:
            # save score text
            filename, file_ext = os.path.splitext(os.path.basename(image_path))

            real_mask = generate_text_mask(score_text, args.low_text, image, img_resized, target_ratio)
            real_mask_file = result_folder + "/" + filename + '_mask.png'
        
            cv2.imwrite(real_mask_file, real_mask)

            #overlay_image = overlay_mask_on_image(image, real_mask, alpha=0.5)
            #overlay_file = result_folder + "/overlay_" + filename + '_mask.jpg'
            #cv2.imwrite(overlay_file, overlay_image)

            heatmap_overlay_image = overlay_mask_on_image(image, real_mask, alpha=0.5)
            heatmap_overlay_file = result_folder + "/" + filename + '_mask_overlay.jpg'
            cv2.imwrite(heatmap_overlay_file, heatmap_overlay_image)

            mask_file = result_folder + "/res_" + filename + '_heatmap.jpg'
            cv2.imwrite(mask_file, ret_score_text)

            mask_and_box_image = overlay_mask_and_boxes(image, real_mask, bboxes, alpha=0.5)
            mask_and_box_image_file = result_folder + "/" + filename + '_mask_and_box_overlay.jpg'
            cv2.imwrite(mask_and_box_image_file, mask_and_box_image)

        file_utils.saveResult(image_path, image[:,:,::-1], bboxes, dirname=args.output_folder)

    print("elapsed time : {}s".format(time.time() - t))
