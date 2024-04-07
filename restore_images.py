import os
from PIL import Image

import sys

if len(sys.argv) > 1:
    path = sys.argv[1]
else:
    print("Provide folder")
    exit(0)

input_folder = path
assert path[-6:] == "_crops"
output_folder = path[:-6] # remove _crops
crop_size = 256

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def check_crop(filename):
    if not (filename.endswith('.jpg') or filename.endswith('.png')):
        return False
    if len(filename[:filename.rfind('.')].split('_')) < 3:
        return False
    return True

def update_size(sizes, filename):

    # remove extension
    filename = filename[:filename.rfind('.')] 

    parts = filename.split('_')
    image_name = '_'.join(parts[:-2])
    if image_name not in sizes:
        sizes[image_name] = [0, 0]
    sizes[image_name][0] = max(sizes[image_name][0], int(parts[-2]) + 1)
    sizes[image_name][1] = max(sizes[image_name][1], int(parts[-1]) + 1)

def construct_image(crop_folder, output_folder):
    constructed = set()
    filenames = sorted(os.listdir(crop_folder))
    sizes = dict()
    for filename in filenames:
        if not check_crop(filename):
            continue
        update_size(sizes, filename)
    full_image = None
    last_name = ""

    already_constructed = set()
    
    for filename in filenames:
        if not check_crop(filename):
            continue
        
        parts = filename[:filename.rfind('.')].split("_")
        pos_y, pos_x = map(int, parts[-2:])
        image_name = "_".join(parts[:-2])
        if image_name == "image":
            print(image_name, pos_y, pos_x, sizes[image_name])
        if image_name != last_name:
            if last_name != "":
                full_image.save(os.path.join(output_folder, f'{last_name}.jpg'))
                if last_name in already_constructed:
                    print("Trying to construct same image twice")
                already_constructed.add(last_name)
            last_name = image_name
            full_image = Image.new('RGB', (crop_size * sizes[image_name][1], crop_size * sizes[image_name][0]))

        crop_image = Image.open(os.path.join(crop_folder, filename))
        full_image.paste(crop_image, (pos_x * crop_size, pos_y * crop_size))
    
    full_image.save(os.path.join(output_folder, f'{last_name}.jpg'))

    
        
    # for filename in os.listdir(crop_folder):
    #     if filename.endswith('.jpg') or filename.endswith('.png'):
    #         full_image = Image.new('RGB', (crop_size * 10, crop_size * 10))
            
    #         parts = filename.split('_')
    #         image_name = '_'.join(parts[:-2])  # Get the common image name prefix
    #         if image_name in constructed:
    #             continue
    #         print(image_name)
    #         constructed.add(image_name)
    #         mx_row = 0
    #         mx_col = 0
    #         for crop_file in os.listdir(crop_folder):
    #             if crop_file.startswith(image_name + "_"):
    #                 parts = crop_file.split('_')
    #                 row, col = int(parts[-2]), int(parts[-1].split('.')[0])
    #                 mx_row = max(mx_row, row)
    #                 mx_col = max(mx_col, col)
                    
    #                 crop_image = Image.open(os.path.join(crop_folder, crop_file))
    #                 full_image.paste(crop_image, (col * crop_size, row * crop_size))
    #         full_image = full_image.crop((0, 0, (mx_row + 1) * crop_size, (mx_col + 1) * crop_size))
            
    #         full_image.save(os.path.join(output_folder, f'{image_name}.jpg'))
        # break

# Call the function to construct the image
construct_image(input_folder, output_folder)
