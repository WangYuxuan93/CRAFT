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
from visualize import generate_text_mask, generate_text_mask, overlay_boxes_on_image, overlay_mask_and_boxes
from merge import merge_boxes

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
                             trained_model_path='final_net_param.pth',
                             zeroshot_model_path=None,
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
    
    net, zeroshot_net = load_models(
        trained_model_path=trained_model_path,
        zeroshot_model_path=zeroshot_model_path if hasattr(args, 'zeroshot_model') else None,
        use_cuda=args.cuda
    )

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
    

    return bboxes, score_text, target_ratio, img_resized


def predict_legend_box(image, trained_model_path="model/td-bs8_8gpu-v1/finetuned_epoch_9_iter700.pth", scale=1, use_cuda=True):
    bboxes, score_text, target_ratio, img_resized = predict_image_with_boxes(
            image_input=image,
            trained_model_path=trained_model_path,
            zeroshot_model_path=None,
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

def predict_main_map_box(image, trained_model_path="model/after_zero-0430_main_map-bs16_8gpu-v2/finetuned_epoch_9_iter80.pth", scale=1, use_cuda=True):
    bboxes, ret_score_text, score_text, target_ratio, img_resized = predict_image_with_boxes(
            image_input=image,
            trained_model_path=trained_model_path,
            zeroshot_model_path='model/craft_mlt_25k.pth',
            text_threshold=0.3,
            low_text=0.3,
            link_threshold=0.4,
            canvas_size=4096,
            mag_ratio=1,
            target_size=768,
            use_target_size=True,
            scale=scale,
            use_cuda=use_cuda,
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CRAFT Text Detection')
    parser.add_argument('--zeroshot_model', default=None, type=str, help='path to the zeroshot model')
    parser.add_argument('--merge_iou_threshold', default=0.7, type=float, help='merge iou threshold for nms')
    parser.add_argument('--merge_cover_threshold', default=0.9, type=float, help='merge cover threshold for nms')
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

    #测试结果保存路径
    result_folder = args.result_folder
    os.makedirs(result_folder, exist_ok=True)

    # load net
    net, zeroshot_net = load_models(
        trained_model_path=args.trained_model,
        zeroshot_model_path=args.zeroshot_model if hasattr(args, 'zeroshot_model') else None,
        use_cuda=args.cuda
    )

    t = time.time()
    # load data
    for image_path in tqdm(image_list):
        #print("Test image {:d}/{:d}: {:s}".format(k+1, len(image_list), image_path), end='\r')
        image = imgproc.loadImage(image_path)

        bboxes, score_text, img_resized, target_ratio = infer_single_image(image=image,
                                                                                net=net,
                                                                                zeroshot_net=zeroshot_net,
                                                                                use_target_size=args.use_target_size,
                                                                                target_size=args.target_size,
                                                                                canvas_size=args.canvas_size,
                                                                                mag_ratio=args.mag_ratio,
                                                                                text_threshold=args.text_threshold,
                                                                                link_threshold=args.link_threshold,
                                                                                low_text=args.low_text,
                                                                                use_cuda=args.cuda,
                                                                                output_char_box=not args.output_word_box,
                                                                                merge_iou_threshold=args.merge_iou_threshold,
                                                                                merge_cover_threshold=args.merge_cover_threshold,
                                                                                scale=args.scale)
        
        if not args.only_pred_file:
            # save score text
            filename, file_ext = os.path.splitext(os.path.basename(image_path))

            real_mask = generate_text_mask(score_text, args.low_text, image, img_resized, target_ratio)
            real_mask_file = result_folder + "/" + filename + '_mask.png'
        
            cv2.imwrite(real_mask_file, real_mask)

            box_image = overlay_boxes_on_image(image, bboxes)
            box_image_file = result_folder + "/" + filename + '_box_overlay.jpg'
            cv2.imwrite(box_image_file, box_image)

            #mask_file = result_folder + "/res_" + filename + '_heatmap.jpg'
            #cv2.imwrite(mask_file, ret_score_text)

            mask_and_box_image = overlay_mask_and_boxes(image, real_mask, bboxes, alpha=0.5)
            mask_and_box_image_file = result_folder + "/" + filename + '_mask_and_box_overlay.jpg'
            cv2.imwrite(mask_and_box_image_file, mask_and_box_image)

        file_utils.saveResult(image_path, image[:,:,::-1], bboxes, dirname=args.output_folder)

    print("elapsed time : {}s".format(time.time() - t))
