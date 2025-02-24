import torch
from PIL import Image
import os
import numpy as np
from utils.gaussian import GaussianGenerator
from converts.synthText_convert import *
import matplotlib.pyplot as plt

#重写dataset类
class SynthDataset(torch.utils.data.Dataset):
    def __init__(self, image_transform=None, label_transform=None, target_transform=None, file_path=None, image_dir=None):
        super(SynthDataset, self).__init__() #继承父类构造方法

        #图片名和标签数据（不是标签名）
        # 加载syntnText数据集
        imnames, charBB, txt = load_synthText(file_path)
        self.imnames = imnames
        self.charBB = charBB
        self.txt = txt
        self.image_dir = image_dir #训练数据文件夹地址
        self.image_transform = image_transform
        self.label_transform = label_transform
        self.target_transform = target_transform
        self.sc_map = torch.ones(1)
    def __len__(self):
        return len(self.imnames)
    
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

    def apply_colormap(self, heatmap):
        """
        将灰度热力图应用颜色映射，转换为彩色热力图
        :param heatmap: 输入的灰度热力图（numpy数组）
        :return: 彩色热力图（RGBA格式）
        """
        # 使用matplotlib的jet颜色映射
        return plt.cm.jet(heatmap)  # jet是一个常用的颜色映射

    def overlay_heatmap(self, image, heatmap, alpha=0.5):
        """
        将热力图叠加到图像上，并将热力图变为彩色。

        Parameters:
        - image (PIL.Image): 原图像
        - heatmap (PIL.Image): 高斯热力图或亲和力热力图
        - alpha (float): 热力图的透明度，0.0表示完全透明，1.0表示完全不透明

        Returns:
        - result_image (PIL.Image): 叠加后的图像
        """
        # 确保 heatmap 是 numpy 数组格式，用于颜色映射
        if isinstance(heatmap, Image.Image):
            heatmap = np.array(heatmap)  # 如果 heatmap 是 PIL.Image 格式，则转换为 numpy 数组
        
        # 将热力图转换为彩色图像
        heatmap_rgb = self.apply_colormap(heatmap)  # 将灰度热力图转换为彩色热力图
        
        # 将彩色热力图（RGBA格式）转换为 numpy 数组
        heatmap_rgb_array = (heatmap_rgb[:, :, :3] * 255).astype(np.uint8)  # 丢弃 alpha 通道并缩放到 [0, 255]
        
        # 将 numpy 数组转换回 PIL.Image 对象
        heatmap_rgb = Image.fromarray(heatmap_rgb_array)
        
        # 叠加热力图到原图
        result_image = Image.blend(image.convert("RGB"), heatmap_rgb, alpha)
        
        result_image.show()  # 显示最终结果图像

    # label应为高斯热力图
    def __getitem__(self, idx, debug=False):
        imname = self.imnames[idx].item()
        image = Image.open(os.path.join(self.image_dir, imname))

        #numpy ndarray格式
        char_boxes_array = np.array(self.charBB[idx])
        char_boxes_array = char_boxes_array.swapaxes(0,2)
        #生成affinity边框列表
        word_lines = self.txt[idx]
        word_list = get_wordsList(word_lines)   #文字列表
        char_boxes_list, affinity_boxes_list = get_affinity_boxes_list(char_boxes_array, word_list)
        if debug:
            self.draw_box(image, char_boxes_list, color="red")
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

        #self.draw_box(image, char_boxes_list, color="red")
        #self.overlay_heatmap(image, region_scores, alpha=0.5)

        if self.image_transform is not None:
            image = self.image_transform(image)

        if self.label_transform is not None:
            region_scores = self.label_transform(region_scores)
            affinity_scores = self.label_transform(affinity_scores)
            sc_map = self.label_transform(sc_map)
        return image, region_scores, affinity_scores, sc_map

    #获取图片的高斯热力图
    def get_region_scores(self, heat_map_size, char_boxes_list):
        # 高斯热力图
        gaussian_generator = GaussianGenerator()
        region_scores = gaussian_generator.gen(heat_map_size, char_boxes_list)
        return region_scores

