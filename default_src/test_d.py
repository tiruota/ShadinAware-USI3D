"""
@author: DremTale
"""
from __future__ import print_function

from tqdm import tqdm

from utils_d import get_config
from trainer_d import UnsupIntrinsicTrainer
import argparse
from torch.autograd import Variable
import torchvision.utils as vutils
import sys
import torch
import os
from torchvision import transforms
from PIL import Image
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, default='configs/intrinsic_MIT.yaml', help="net configuration")
parser.add_argument('-i', '--test_list', type=str, default='dataset/test-input.txt',
                    help="input image path")
parser.add_argument('-o', '--output_folder', type=str, default='id_mit_train-inner-opt',
                    help="output image path")
parser.add_argument('-p', '--checkpoint', type=str, default='checkpoints/mit_inner-opt/gen_00440000.pt',
                    help="checkpoint of MUID")
parser.add_argument('-g', '--guided_folder', type=str, default='dataset/L1smooth/',
                    help="output image path")
parser.add_argument('--seed', type=int, default=10, help="random seed")
parser.add_argument('-f',"--filter_type",default="guided", help="""Which filter to choose,the guided filter (guided) orthe joint bilateral filter (bilateral).""")
parser.add_argument('-k',"--sigma_color", type=float, default=20, help="color parameter")
parser.add_argument('-s',"--sigma_spatial", type=float, default=22, help="spatial parameter")
opts = parser.parse_args()

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def apply_filter(filter_type, image, joint, sigma_color, sigma_spatial):
    """
    Apply the joint/guided filter.

    Apply the filter of the given type on the image according to the
    joint/guidance image and based on the given parameters.
    """
    if sigma_color <= 0 or sigma_spatial <= 0:
        raise ValueError("Parameters are expected to be positive.")
    if filter_type == 'bilateral':
        # using the joint bilateral filter instead
        filtered = cv2.ximgproc.jointBilateralFilter(joint,
                                                     image,
                                                     d=-1,
                                                     sigmaColor=sigma_color,
                                                     sigmaSpace=sigma_spatial)
    elif filter_type == 'guided':
        # or the guided filter
        filtered = cv2.ximgproc.guidedFilter(guide=joint,
                                             src=image,
                                             radius=int(sigma_spatial),
                                             eps=sigma_color)
    else:
        raise ValueError("filter_type must be 'bilateral' or 'guided'.")
    return filtered


wo_fea = 'wo_fea' in opts.checkpoint


torch.manual_seed(opts.seed)
torch.cuda.manual_seed(opts.seed)
if not os.path.exists(opts.output_folder+'reflectance'):
    os.makedirs(opts.output_folder+'reflectance')
if not os.path.exists(opts.output_folder+'shading'):
    os.makedirs(opts.output_folder+'shading')
if not os.path.exists(opts.output_folder+'input'):
    os.makedirs(opts.output_folder+'input')
if not os.path.exists(opts.output_folder+'reflectance_smooth'):
    os.makedirs(opts.output_folder+'reflectance_smooth')
# Load experiment setting
config = get_config(opts.config)
opts.reflect_only = False
opts.save_smooth = True

# Setup model and data loader
trainer = UnsupIntrinsicTrainer(config)

state_dict = torch.load(opts.checkpoint, map_location='cuda:0')
trainer.gen_i.load_state_dict(state_dict['i'])
trainer.gen_r.load_state_dict(state_dict['r'])
trainer.gen_s.load_state_dict(state_dict['s'])
trainer.fea_s.load_state_dict(state_dict['fs'])
trainer.fea_m.load_state_dict(state_dict['fm'])

trainer.cuda()
trainer.eval()

if 'new_size' in config:
    new_size = config['new_size']
else:
    new_size = config['new_size_i']

intrinsic_image_decompose = trainer.inference

root = config['data_root']

images = []
with open(opts.test_list, 'r') as fid:
        lines = fid.readlines()
        for line in lines:
            line = line.strip()
            images.append(os.path.join(root, line))

with torch.no_grad():
    transform = transforms.Compose([transforms.Resize((256,256)),
                                    transforms.ToTensor(),
#                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))    # Make sure the vaule range of input tensor be consistent to the training time
                                   ])

    # image_paths = os.listdir(opts.input_dir)
    # image_paths = [x for x in image_paths if is_image_file(x)]
    t_bar = tqdm(images)
    t_bar.set_description('Processing')
    for image_name in t_bar:
        # image_pwd = os.path.join(opts.input_dir, image_name)
        image_pwd = image_name

        out_root = os.path.join(opts.output_folder, image_name.split('.')[0])

        # if not os.path.exists(out_root):
        #     os.makedirs(out_root)
        
        image = Image.open(image_pwd).convert('RGB')
        w_org, h_org = image.size
        image_cuda = Variable(transform(image).unsqueeze(0).cuda())

        # Start testing
        im_reflect, im_shading = intrinsic_image_decompose(image_cuda, wo_fea)
        im_reflect = (im_reflect + 1) / 2.
        im_shading = (im_shading + 1) / 2.

        # path_reflect = os.path.join(out_root, 'output_r.jpg')
        # path_shading = os.path.join(out_root, 'output_s.jpg')
        path_reflect = os.path.join(opts.output_folder+'reflectance/', os.path.splitext(os.path.basename(image_name))[0]+'.png')
        path_shading = os.path.join(opts.output_folder+'shading/', os.path.splitext(os.path.basename(image_name))[0]+'.png')

        # vutils.save_image(im_reflect.data, path_reflect, padding=0, normalize=True)
        # vutils.save_image(im_shading.data, path_shading, padding=0, normalize=True)
        if opts.save_smooth:
            reflect = vutils.make_grid(im_reflect, normalize=True)
            reflect = reflect.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
            reflect = cv2.cvtColor(reflect, cv2.COLOR_RGB2BGR)
            h, w, _ = reflect.shape[:3]
            joint_path = opts.guided_folder + os.path.splitext(os.path.basename(image_name))[0]+'.png'
            joint = cv2.imread(joint_path)
            joint = cv2.resize(joint, dsize=(w,h))

            filtered = apply_filter(opts.filter_type, reflect, joint, opts.sigma_color, opts.sigma_spatial)
            filtered = cv2.resize(filtered, dsize=(w_org, h_org))
            cv2.imwrite(opts.output_folder + "reflectance_smooth/" + os.path.splitext(os.path.basename(image_name))[0]+'.png', filtered)

        if not opts.reflect_only:
            im_reflect = vutils.make_grid(im_reflect, normalize=True)
            im_reflect = im_reflect.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
            im_reflect = cv2.cvtColor(im_reflect, cv2.COLOR_RGB2BGR)
            im_reflect = cv2.resize(im_reflect, dsize=(w_org, h_org))
            cv2.imwrite(path_reflect, im_reflect)

            im_shading = vutils.make_grid(im_shading, normalize=True)
            im_shading = im_shading.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
            im_shading = cv2.resize(im_shading, dsize=(w_org, h_org))
            cv2.imwrite(path_shading, im_shading)

            image.save(os.path.join(opts.output_folder+'input', os.path.splitext(os.path.basename(image_name))[0]+'.png'))
        else:
            im_reflect = vutils.make_grid(im_reflect, normalize=True)
            im_reflect = im_reflect.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
            im_reflect = cv2.resize(im_reflect, dsize=(w_org, h_org))
            cv2.imwrite(path_reflect, im_reflect)