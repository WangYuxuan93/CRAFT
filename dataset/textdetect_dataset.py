import torch
from PIL import Image
import os
import numpy as np
from utils.img_util import img_normalize, load_image
from utils.box_util import cal_affinity_boxes
from utils.gaussian import GaussianGenerator
from converts.synthText_convert import *

def load_text_detect(images_path, labels_path):
    image_names = os.listdir(images_path)
    label_names = os.listdir(labels_path)
    image_names.sort()
    label_names.sort()
    return image_names, label_names

def get_text_detect_char_box(label_path):

    fh = open(label_path, 'r')
    char_boxes = []
    chars = []
    for line in fh:
        line = line.rstrip()
        line = line.split(',')
        box = np.array(line[:8], dtype=int)
        #转换格式
        box = np.float32(
            [[box[0], box[1]], [box[2], box[3]], [box[4], box[5]],
             [box[6], box[7]]])
        char = line[-1]
        char_boxes.append(box)
        chars.append(char)
    return char_boxes, chars

#输入一张图片的字符边框列表，字符串列表，输出affinity边框列表
def get_affinity_boxes_list(char_boxes):
    """

    :param char_boxes_array: 字符边框矩阵
    :param wordsList: 从SynthText/gt.mat中读取到的文字列表
    :return: 字符边框列表和字间边框列表
    """
    # 字符索引，确定word中字符个数
    affinity_boxes_list = list()
    char_boxes_list = list()
    affinity_boxes_list.append(cal_affinity_boxes(char_boxes))
    char_boxes_list.append(char_boxes)
    affinity_boxes_list = list(chain.from_iterable(affinity_boxes_list))
    char_boxes_list = list(chain.from_iterable(char_boxes_list))
    return char_boxes_list, affinity_boxes_list

#重写dataset类
class TextDetectDataset(torch.utils.data.Dataset):
    def __init__(self, image_transform=None, label_transform=None, target_transform=None, images_dir=None, labels_dir=None):
        super(TextDetectDataset, self).__init__() #继承父类构造方法

        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.image_names, self.label_names = load_text_detect(self.images_dir, self.labels_dir)

        self.image_transform = image_transform
        self.label_transform = label_transform
        self.target_transform = target_transform
        self.sc_map = torch.ones(1)

    def __len__(self):
        return len(self.image_names)

    def draw_box(self, image, word_boxes, color="red"):
        from PIL import Image, ImageDraw
        import matplotlib.pyplot as plt
        if image.mode != 'RGB':
            image = image.convert('RGB')
        draw = ImageDraw.Draw(image)

        # 遍历word_boxes，绘制bounding box
        for box in word_boxes:
            # 获取四个角的坐标
            top_left = tuple(box[0])
            top_right = tuple(box[1])
            bottom_right = tuple(box[2])
            bottom_left = tuple(box[3])
            
            # 绘制矩形框
            draw.line([top_left, top_right, bottom_right, bottom_left, top_left], fill=color, width=2)

        # 显示绘制后的图片
        plt.imshow(image)
        plt.axis('off')  # 不显示坐标轴
        plt.show()

    # label应为高斯热力图
    def __getitem__(self, idx, debug=False):
        fn = self.image_names[idx]
        image = Image.open(os.path.join(self.images_dir, fn))
        
        label_name = self.label_names[idx]
        char_boxes, chars = get_text_detect_char_box(os.path.join(self.labels_dir, label_name))
    
        char_boxes_list, affinity_boxes_list = get_affinity_boxes_list(char_boxes)
        if debug:
            print ("char_boxes_list: ",char_boxes_list)
            self.draw_box(image, char_boxes_list, color="red")
            print ("affinity_boxes_list: ",affinity_boxes_list)
            self.draw_box(image, affinity_boxes_list, color="blue")

        width, height = image.size
        heat_map_size = (height, width)
        region_scores = self.get_region_scores(heat_map_size, char_boxes_list) * 255
        affinity_scores = self.get_region_scores(heat_map_size, affinity_boxes_list) * 255
        sc_map = np.ones(heat_map_size, dtype=np.float32) * 255
        #numpy.ndarray转为PIL.Image
        region_scores = Image.fromarray(np.uint8(region_scores))
        affinity_scores = Image.fromarray(np.uint8(affinity_scores))
        sc_map = Image.fromarray(np.uint8(sc_map))
        if self.image_transform is not None:
            image = self.image_transform(image)
            #print (image.shape)

        if self.label_transform is not None:
            region_scores = self.label_transform(region_scores)
            affinity_scores = self.label_transform(affinity_scores)
            sc_map = self.label_transform(sc_map)
            #print (region_scores.shape)
        return image, region_scores, affinity_scores, sc_map

    #获取图片的高斯热力图
    def get_region_scores(self, heat_map_size, char_boxes_list):
        # 高斯热力图
        gaussian_generator = GaussianGenerator()
        region_scores = gaussian_generator.gen(heat_map_size, char_boxes_list)
        return region_scores

