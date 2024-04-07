import argparse
parser = argparse.ArgumentParser(
                    prog='Crops maker',
                    description='''
Format before:
input_folder/
  0001_001_S6_00100_00060_3200_L/
    GT_SRGB_010.PNG
    NOISY_SRGB_010.PNG
  ...
  
Format after:
output_folder/
  input/
    0001_001_S6_00100_00060_3200_L.PNG
    0001_001_S6_00100_00060_6400_L.PNG
    ...
  ground_truth/
    0001_001_S6_00100_00060_3200_L.PNG
    0001_001_S6_00100_00060_6400_L.PNG
    ...
'''
)
parser.add_argument('-i', '--input-path', default='.')
parser.add_argument('-o', '--output-path', required=False)

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

import glob
import shutil

os.makedirs(os.path.join(output_folder, "input"), exist_ok=True)
os.makedirs(os.path.join(output_folder, "ground_truth"), exist_ok=True)

for filepath in glob.glob(os.path.join(input_folder, "*/NOISY*.PNG")):
    shutil.copy(filepath, os.path.join(output_folder, "input", filepath.split("/")[-2] + ".png"))

for filepath in glob.glob(os.path.join(input_folder, "*/GT*.PNG")):
    shutil.copy(filepath, os.path.join(output_folder, "ground_truth", filepath.split("/")[-2] + ".png"))