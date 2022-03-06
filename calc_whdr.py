from tqdm import tqdm

import argparse
import json
import sys
import os
from whdr import compute_whdr, load_image
import re

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_dir', type=str, default='/home/ros/datasets/intrinsic/MIT-inner-split/trainA',
                    help="input image path")
parser.add_argument('-j', '--judgement_dir', type=str, default='dataset/json',
                    help="checkpoint of MUID")
opts = parser.parse_args()

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

image_paths = sorted(os.listdir(opts.input_dir), key=natural_keys)
image_paths = [x for x in image_paths if is_image_file(x)]
t_bar = tqdm(image_paths)
t_bar.set_description('Processing')

whdr_sum = 0
whdr_min = float('inf')
whdr_max = 0
n = 0
for image_name in t_bar:
    image_pwd = os.path.join(opts.input_dir, image_name)
    judgement_pwd = os.path.join(opts.judgement_dir, os.path.splitext(os.path.basename(image_name))[0]+'.json')

    reflectance = load_image(image_pwd)
    judgement = json.load(open(judgement_pwd))
    whdr = compute_whdr(reflectance, judgement)
    whdr_sum += whdr
    n = n + 1

    if whdr < whdr_min:
        whdr_min = whdr
    elif whdr > whdr_max:
        whdr_max = whdr

print("whdr_mean={}, whdr_max={}, whdr_min={}".format(whdr_sum/n, whdr_max, whdr_min))