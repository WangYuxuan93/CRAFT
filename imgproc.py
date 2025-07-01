"""  
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-
import numpy as np
from skimage import io
import cv2
import matplotlib.pyplot as plt
from tifffile import TiffFile, TiffFileError
from PIL import Image

def load_tif_to_rgb(path):
    with TiffFile(path) as tif:
        page = tif.pages[0]
        image = page.asarray()
        
        if page.colormap is not None:
            # Paletted image
            palette = page.colormap  # shape: (3, N)
            palette = (palette / palette.max() * 255).astype(np.uint8)
            if image.max() >= palette.shape[1]:
                raise ValueError("Palette index exceeds colormap range.")

            h, w = image.shape
            rgb_image = np.zeros((h, w, 3), dtype=np.uint8)
            for c in range(3):
                rgb_image[..., c] = palette[c][image]
            return rgb_image
        else:
            # Non-paletted image, likely RGB or RGBA
            if image.ndim == 2:
                # Grayscale -> stack to RGB
                return np.stack([image]*3, axis=-1)
            elif image.shape[2] == 3:
                # Already RGB
                return image.astype(np.uint8)
            elif image.shape[2] == 4:
                # RGBA -> drop alpha
                return image[:, :, :3].astype(np.uint8)
            else:
                raise ValueError(f"Unsupported image shape: {image.shape}")

def loadImage(img_file, debug=False):
    if img_file.endswith(".tif"):
        try:
            img = load_tif_to_rgb(img_file)
        except (TiffFileError, ValueError, IndexError):
            print (f"Failed loading {img_file}")
            img = Image.open(img_file).convert("RGB")
            img = np.array(img)
    else:
        img = io.imread(img_file)
    
    if debug:
        print (img_file)
        print("Image shape: {}, dtype: {}".format(img.shape, img.dtype))
    
    if img.shape[0] == 2: img = img[0]
    if len(img.shape) == 2 : img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:   img = img[:,:,:3]
    #if img.ndim == 3 and img.shape[2] == 4:
    #    img = img[:, :, :3]

    # normalize if needed
    if img.dtype != np.uint8:
        img = img.astype(np.float32)
        img -= img.min()
        img /= img.max()
        img *= 255
        img = img.astype(np.uint8)

    img = np.array(img)

    if debug:
        plt.imshow(img)
        plt.title("Loaded Image")
        plt.axis('off')
        plt.show()

    return img

def normalizeMeanVariance(in_img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
    # should be RGB order
    img = in_img.copy().astype(np.float32)

    img -= np.array([mean[0] * 255.0, mean[1] * 255.0, mean[2] * 255.0], dtype=np.float32)
    img /= np.array([variance[0] * 255.0, variance[1] * 255.0, variance[2] * 255.0], dtype=np.float32)
    return img

def denormalizeMeanVariance(in_img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
    # should be RGB order
    img = in_img.copy()
    img *= variance
    img += mean
    img *= 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def resize_aspect_ratio(img, square_size, interpolation, mag_ratio=1):
    height, width, channel = img.shape

    # magnify image size
    target_size = mag_ratio * max(height, width)

    # set original image size
    if target_size > square_size:
        target_size = square_size
    
    ratio = target_size / max(height, width)    

    target_h, target_w = int(height * ratio), int(width * ratio)
    proc = cv2.resize(img, (target_w, target_h), interpolation = interpolation)


    # make canvas and paste image
    target_h32, target_w32 = target_h, target_w
    if target_h % 32 != 0:
        target_h32 = target_h + (32 - target_h % 32)
    if target_w % 32 != 0:
        target_w32 = target_w + (32 - target_w % 32)
    resized = np.zeros((target_h32, target_w32, channel), dtype=np.float32)
    resized[0:target_h, 0:target_w, :] = proc
    target_h, target_w = target_h32, target_w32

    size_heatmap = (int(target_w/2), int(target_h/2))

    return resized, ratio, size_heatmap

def cvt2HeatmapImg(img):
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    return img
