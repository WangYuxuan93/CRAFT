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

from collections import defaultdict
from visualize import generate_text_mask, generate_text_mask, overlay_boxes_on_image
from predict import infer_single_image, load_models, safe_imwrite

def get_image_paths_v0(root_dir):
    grouped_paths = defaultdict(list)
    
    # 遍历第一层目录（按数字递增的文件夹）
    for first_level in sorted(os.listdir(root_dir), key=lambda x: int(x) if x.isdigit() else float('inf')):
        first_level_path = os.path.join(root_dir, first_level)
        
        if not os.path.isdir(first_level_path):
            continue
        
        # labeled 文件夹路径
        #labeled_path = os.path.join(first_level_path, "labeled")
        labeled_path = os.path.join(first_level_path, "image")
        if not os.path.isdir(labeled_path):
            continue
        
        # 获取 labeled 文件夹中的main.png
        image_files = [
            os.path.join(labeled_path, img) for img in os.listdir(labeled_path)
            if img.endswith("tif") or img.endswith("png")
            #if img in ["main_no_legend.png", "main_line_no_legend.png"]
        ]
        
        #assert len(image_files) == 2
        # 存入字典，使用第一层文件夹的路径作为 key
        if image_files:
            grouped_paths[first_level].extend(image_files)
    
    return dict(grouped_paths)

def get_image_paths(root_dir):
    grouped_paths = defaultdict(list)
    
    # 遍历第一层目录
    for first_level in os.listdir(root_dir):
        first_level_path = os.path.join(root_dir, first_level)
        
        if not os.path.isdir(first_level_path):
            continue
        
        # image 文件夹路径
        labeled_path = os.path.join(first_level_path, "image")
        if not os.path.isdir(labeled_path):
            continue
        
        # 获取 image 文件夹中的图像文件
        image_files = [
            os.path.join(labeled_path, img) for img in os.listdir(labeled_path)
            if img.endswith("tif") or img.endswith("png")
        ]
        
        # 存入字典，使用第一层文件夹的路径作为 key
        if image_files:
            grouped_paths[first_level].extend(image_files)
    
    return dict(grouped_paths)

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

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
    parser.add_argument('--canvas_size', default=1280, type=int, help='image size for inference')
    parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')
    parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
    parser.add_argument('--test_folder', default='/home/brooklyn/ICDAR/icdar2013/test_images/', type=str, help='folder path to input images')
    parser.add_argument('--result_folder', default='./result/', type=str, help='folder path to save result images')
    parser.add_argument('--output_folder', default='./result/pred_labels', type=str, help='folder path to save prediction file')
    parser.add_argument('--only_pred_file', default=False, action='store_true', help='Only output prediction file to output folder')
    parser.add_argument('--target_size', default=768, type=int, help='image size for inference')
    parser.add_argument('--use_target_size', default=False, type=str2bool, help='resize the image to target size')
    parser.add_argument('--scale', default=1, type=float, help='box expanding scale')
    parser.add_argument('--output_to_origin_folder', default=False, action='store_true', help='Whether output to the original folder')
    parser.add_argument('--output_word_box', default=False, action='store_true', help='output word bbox')
    args = parser.parse_args()

    output_char_box = not args.output_word_box

    """ For test images in a folder """
    image_dict = get_image_paths(args.test_folder)

    # load net
    net, zeroshot_net = load_models(
        trained_model_path=args.trained_model,
        zeroshot_model_path=args.zeroshot_model if hasattr(args, 'zeroshot_model') else None,
        use_cuda=args.cuda
    )

    t = time.time()
    # load data
    for img_id, img_paths in tqdm(image_dict.items()):
        for k, image_path in enumerate(img_paths):
            #print("Test image {:d}/{:d}: {:s}".format(k+1, len(image_list), image_path), end='\r')
            #print (image_path)
            image = imgproc.loadImage(image_path)
            if args.output_to_origin_folder:
                img_dir = os.path.join(args.test_folder, img_id)
                result_folder = os.path.join(img_dir, "visualization")
                output_folder = os.path.join(img_dir, "bbox")
            else:
                result_folder = os.path.join(args.result_folder, img_id)
                result_folder = os.path.join(result_folder, "visualization")
                output_folder = os.path.join(args.output_folder, img_id)
                output_folder = os.path.join(output_folder, "bbox")
            
            if not os.path.isdir(result_folder):
                os.makedirs(result_folder)
            if not os.path.isdir(output_folder):
                os.makedirs(output_folder)

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

                model_name = os.path.basename(os.path.dirname(args.trained_model))
                #print (model_name)
                #exit()
                scale_value = args.scale  # 获取scale的值
                box_type = "wordbox" if args.output_word_box else "charbox"

                # 在文件名中加入模型名和scale值作为前缀
                real_mask_file = os.path.join(result_folder, f"{box_type}_{filename}_mask_{model_name}_sacle-{scale_value}.png")
                box_image_file = os.path.join(result_folder, f"{box_type}_{filename}_box_overlay_{model_name}_sacle-{scale_value}.png")

                real_mask = generate_text_mask(score_text, args.low_text, image, img_resized, target_ratio)

                safe_imwrite(real_mask_file, real_mask)

                box_image = overlay_boxes_on_image(image, bboxes)
                safe_imwrite(box_image_file, box_image)


            file_utils.saveResult(image_path, image[:,:,::-1], bboxes, dirname=output_folder)

    print("elapsed time : {}s".format(time.time() - t))
