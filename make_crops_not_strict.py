import argparse
parser = argparse.ArgumentParser(
                    prog='Crops maker',
                    description='cuts pictures to crops, from one folder to another')
parser.add_argument('-i', '--input-path', default='.')
parser.add_argument('-o', '--output-path', required=False)
parser.add_argument('-c', '--crop-size', default=256, type=int)

args = parser.parse_args()

import os
from PIL import Image
import torch

import sys

print(args)
input_folder = args.input_path
if args.output_path is None:
    output_folder = os.path.join(input_folder, '../input_crops')
else:
    output_folder = args.output_path

crop_size = args.crop_size

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def add_coordinates(filename, i, j):
    idx = filename.rfind('.')
    return filename[:idx] + f"_{i}_{j}" + filename[idx:]

def adjuct_crop(crop, size):
    if crop > size:
        return size
    if crop * 2 > size:
        return size // 2
    if crop * 3 > size:
        return size // 3
    return crop
        

def crop_and_save_image(img_path, output_folder):
    image = Image.open(img_path)
    
    width, height = image.size
    crop_size_w = adjuct_crop(crop_size, width)
    crop_size_h = adjuct_crop(crop_size, height)

    for i in range(0, height, crop_size_h):
        for j in range(0, width, crop_size_w):
            if i + crop_size_h <= height and j + crop_size_w <= width:
                cropped_img = image.crop((j, i, j + crop_size_w, i + crop_size_h))
                save_path = os.path.join(output_folder, add_coordinates(os.path.basename(img_path), i // crop_size_h, j // crop_size_w))
                cropped_img.save(save_path)

# Iterate over all images in the input folder
for image_file in os.listdir(input_folder):
    if image_file.endswith('.jpg') or image_file.endswith('.png'):
        image_path = os.path.join(input_folder, image_file)
        print(image_path)
        crop_and_save_image(image_path, output_folder)