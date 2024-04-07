import argparse
from tqdm import tqdm
parser = argparse.ArgumentParser(
                    prog='Crops maker',
                    description='''
input: folder of images

output: folder of patches with size exactly crop_size''')

parser.add_argument('-n', '--noisy-path', default='.')
parser.add_argument('-r', '--restored-path', default='.')
parser.add_argument('-g', '--ground-truth-path', default=None, required=False)
parser.add_argument('-m', '--metrics-path', default='.')
# parser.add_argument('-t', '--type', nargs='+', help='Types of metrics to calculate')

args = parser.parse_args()

import os
from PIL import Image
import torch

import cv2
import numpy as np
import sys

noisy_folder = args.noisy_path
restored_folder = args.restored_path
gt_folder = args.ground_truth_path
output_folder = args.metrics_path


if output_folder is None:
    output_folder = os.path.join(restored_folder, '..')

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

from torchvision.transforms.functional import pil_to_tensor

def brightness(image):
    return torch.mean(image).item()

import torch
import torch.nn.functional as F

def sharpness(image_tensor):
    # Convert the image tensor to grayscale
    if image_tensor.shape[0] == 3:
        image_gray = 0.2989 * image_tensor[0, :, :] + 0.5870 * image_tensor[1, :, :] + 0.1140 * image_tensor[2, :, :]
    else:
        image_gray = image_tensor

    # Calculate the Laplacian of the grayscale image
    matrix = torch.Tensor([[1, 1, 1],
                          [1, -8, 1],
                          [1, 1, 1]]).to(torch.double)
    laplacian = F.conv2d(image_gray.unsqueeze(0).unsqueeze(0), matrix.unsqueeze(0).unsqueeze(0))

    # Compute the variance of the Laplacian
    sharpness = torch.var(laplacian)

    return sharpness.item()

unsupervised_metrics = {
    "noisy_brightness": (lambda noisy, restored: brightness(noisy)),
    "restored_brightness": (lambda noisy, restored: brightness(restored)),
    "noisy_sharpness": (lambda noisy, restored: sharpness(noisy)),
    "restored_sharpness": (lambda noisy, restored: sharpness(restored)),
}

supervised_metrics = {
    "gt sharpness": (lambda noisy, restored, reference: sharpness(ref))
}

if gt_folder is None:
    supervised_metrics = {}

import pandas as pd
df = pd.DataFrame(columns=list(unsupervised_metrics.keys()) + list(supervised_metrics.keys()))

filenames = sorted(os.listdir(restored_folder))

# Iterate over all images in the input folder
with torch.no_grad():
    for image_file in tqdm(filenames, desc="Evaluating images"):
        if image_file.endswith('.jpg') or image_file.endswith('.png'):

            noisy =  pil_to_tensor(Image.open(os.path.join(noisy_folder, image_file))).to(float)
            restored = pil_to_tensor(Image.open(os.path.join(restored_folder, image_file))).to(float)
            
            for metric, func in unsupervised_metrics.items():
                df.loc[image_file, metric] = func(noisy, restored)

            if gt_folder is None:
                continue
        
            reference_path = os.path.join(gt_folder, image_file)
            ref = pil_to_tensor(Image.open(reference_path)).to(float)
            
            for metric, func in supervised_metrics.items():
                df.loc[image_file, metric] = func(noisy, restored, ref)

print(df)
df.to_csv(os.path.join(output_folder, "metrics.csv"))