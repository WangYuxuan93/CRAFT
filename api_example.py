import os
import argparse
import cv2
from predict import predict_legend_box, predict_main_map_box, overlay_boxes_on_image
from utils import imgproc


def draw_and_save(image, boxes, save_path):
    """绘制预测框并保存图像"""
    image_with_boxes = overlay_boxes_on_image(image.copy(), boxes)
    cv2.imwrite(save_path, image_with_boxes)
    print(f"Saved visualization to {save_path}")


def process_folder(image_folder, predictor_func, output_folder, label):
    """处理指定类型的图像文件夹"""
    image_list = sorted([
        os.path.join(image_folder, f) for f in os.listdir(image_folder)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif'))
    ])
    print(f"\n[{label}] Found {len(image_list)} images in {image_folder}")

    for image_path in image_list:
        image = imgproc.loadImage(image_path)
        boxes = predictor_func(image)
        print(f"{label} - {os.path.basename(image_path)}: {len(boxes)} boxes")

        if output_folder:
            os.makedirs(output_folder, exist_ok=True)
            filename = os.path.splitext(os.path.basename(image_path))[0]
            out_path = os.path.join(output_folder, f"{filename}_{label}.jpg")
            draw_and_save(image, boxes, out_path)


def main(args):
    if args.type == 'legend':
        predictor = lambda img: predict_legend_box(img, trained_model_path=args.model_path, scale=args.scale, use_cuda=args.cuda)
        label = 'legend'
    elif args.type == 'mainmap':
        predictor = lambda img: predict_main_map_box(img, trained_model_path=args.model_path, scale=args.scale, use_cuda=args.cuda)
        label = 'mainmap'
    else:
        raise ValueError("type must be 'legend' or 'mainmap'")

    process_folder(
        image_folder=args.image_folder,
        predictor_func=predictor,
        output_folder=args.output_dir if args.save else None,
        label=label
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, required=True, choices=['legend', 'mainmap'],
                        help="Specify the prediction type: 'legend' or 'mainmap'")
    parser.add_argument('--image_folder', type=str, required=True, help='Path to the folder containing images')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model weights')
    parser.add_argument('--scale', type=float, default=1.0, help='scaling factor (default = 1.0)')
    parser.add_argument('--cuda', action='store_true', help='Use GPU for inference if available')
    parser.add_argument('--save', action='store_true', help='Whether to save the visualized result images')
    parser.add_argument('--output_dir', type=str, default='results/', help='Directory to save output images')
    args = parser.parse_args()

    main(args)