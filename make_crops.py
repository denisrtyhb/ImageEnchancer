import argparse
from tqdm import tqdm
parser = argparse.ArgumentParser(
                    prog='Crops maker',
                    description='''
input: folder of images

output: folder of patches with size exactly crop_size''')
parser.add_argument('-i', '--input-path', default='.')
parser.add_argument('-o', '--output-path', required=False)
parser.add_argument('-c', '--crop-size', default=256, type=int)

args = parser.parse_args()

import os
from PIL import Image
import torch

import sys

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

def crop_and_save_image(img_path, output_folder):
    image = Image.open(img_path)
    
    width, height = image.size
    num_crops = (width // crop_size) * (height // crop_size)

    for i in range(0, height, crop_size):
        for j in range(0, width, crop_size):
            if i + crop_size <= height and j + crop_size <= width:
                cropped_img = image.crop((j, i, j + crop_size, i + crop_size))
                save_path = os.path.join(output_folder, add_coordinates(os.path.basename(img_path), i // crop_size, j // crop_size))
                cropped_img.save(save_path)

# Iterate over all images in the input folder
for image_file in tqdm(os.listdir(input_folder), desc="Cropping images"):
    if image_file.endswith('.jpg') or image_file.endswith('.png'):
        image_path = os.path.join(input_folder, image_file)
        crop_and_save_image(image_path, output_folder)