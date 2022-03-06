from tqdm import tqdm

import argparse
import sys
import os
import torch
from torchvision import transforms
from torch.autograd import Variable
import torchvision.utils as vutils
import torch.nn as nn
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--input_reflectance', type=str, default='dataset/edge_reflectance_dexi/',)
parser.add_argument('-s', '--input_shading', type=str, default='dataset/reflectance/',)
parser.add_argument('-t', '--target_reflectance', type=str, default='dataset/reflectance/',)
parser.add_argument('-u', '--target_shading', type=str, default='dataset/shading/',)
opts = parser.parse_args()

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


mse = nn.MSELoss()

reflectance_paths = os.listdir(opts.input_reflectance)
reflectance_paths = [x for x in reflectance_paths if is_image_file(x)]

mse_ref = []
mse_shading = []

with torch.no_grad():
    transform = transforms.Compose([transforms.Resize((256,256)),
                                    transforms.ToTensor(),
                                   ])

    t_bar = tqdm(reflectance_paths)
    t_bar.set_description('Processing')
    for image_name in t_bar:
        reflectance_pwd_in = os.path.join(opts.input_reflectance, image_name)
        shading_pwd_in = os.path.join(opts.input_shading, image_name)
        reflectance_pwd_tar = os.path.join(opts.target_reflectance, image_name)
        shading_pwd_tar = os.path.join(opts.target_shading, image_name)

        ref_in = Variable(transform(Image.open(reflectance_pwd_in).convert('RGB')).cuda())
        shading_in = Variable(transform(Image.open(shading_pwd_in).convert('RGB')).cuda())
        ref_target = Variable(transform(Image.open(reflectance_pwd_tar).convert('RGB')).cuda())
        shading_target = Variable(transform(Image.open(shading_pwd_tar)).cuda())

        mse_ref.append(mse(ref_in, ref_target))
        mse_shading.append(mse(shading_in, shading_target))

    mse_ref_ave = sum(mse_ref)/len(mse_ref)
    mse_shading_ave = sum(mse_shading)/len(mse_shading)
    ave_total = (mse_ref_ave + mse_shading_ave)/2.

    print("Ave MSE of Reflectance:"+str(mse_ref_ave))
    print("Ave MSE of Shading:"+str(mse_shading_ave))
    print("Total Ave MSE:"+str(ave_total))