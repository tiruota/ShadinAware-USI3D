"""
@author:    DreamTale
@institute: PHI-AI Lab
"""
import argparse

import torch
import torch.backends.cudnn as cudnn
from utils import get_local_time
from trainer import UnsupIntrinsicTrainer
from utils import prepare_sub_folder, write_loss, get_config, write_2images, Timer
from utils import get_intrinsic_data_loader
import numpy as np

import os
import sys
import tensorboardX
import shutil
from torchvision import transforms

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, default='configs/configs.yaml',
                    help='Path to the config file.')
parser.add_argument('-o', '--output_path', type=str, default='checkpoints-tmp', help="outputs path")
parser.add_argument('-r', "--resume", action="store_true", default=False,
                    help='whether to resume training from the last checkpoint')
parser.add_argument('-g', '--gpu_id', type=int, default=0, help="gpu id")
opts = parser.parse_args()
opts.is_sup = False
cudnn.benchmark = True

# ┌────────────────────────────────────────────────────────────────────┐
# │                     Load experiment setting                        │ 
# └────────────────────────────────────────────────────────────────────┘

config = get_config(opts.config)

max_iter = config['max_iter']
display_size = config['display_size']

intrinsic_rate = None
if 'ablation_study' in config:
    if 'intrinsic_rate' in config['ablation_study']:
        intrinsic_rate = config['ablation_study']['intrinsic_rate']

cudnn.benchmark = True
torch.cuda.set_device(opts.gpu_id)

# ┌────────────────────────────────────────────────────────────────────┐
# │                  Setup model and data loader                       │ 
# └────────────────────────────────────────────────────────────────────┘
trainer = UnsupIntrinsicTrainer(config)
trainer.cuda()

train_loader, test_loader = get_intrinsic_data_loader(config, is_sup=opts.is_sup, rate=intrinsic_rate)
train_display_images_i = torch.stack([train_loader.dataset[i][0] for i in range(display_size)]).cuda()
train_display_images_e_i = torch.stack([train_loader.dataset[i][1] for i in range(display_size)]).cuda()
train_display_images_r = torch.stack([train_loader.dataset[i][2]['albedo'] for i in range(display_size)]).cuda()
train_display_images_e_r = torch.stack([train_loader.dataset[i][2]['edge-albedo'] for i in range(display_size)]).cuda()
train_display_images_s = torch.stack([train_loader.dataset[i][2]['shading'] for i in range(display_size)]).cuda()

# ┌────────────────────────────────────────────────────────────────────┐
# │                 Setup logger and output folders                    │ 
# └────────────────────────────────────────────────────────────────────┘
model_name = os.path.splitext(os.path.basename(opts.config))[0]
train_writer = tensorboardX.SummaryWriter(os.path.join(opts.output_path + "/logs", model_name))
output_directory = os.path.join(opts.output_path + "/outputs", model_name)
checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml'))  # copy config file to output folder

# ┌────────────────────────────────────────────────────────────────────┐
# │                         Start training                             │ 
# └────────────────────────────────────────────────────────────────────┘
iterations = trainer.resume(checkpoint_directory, param=config) if opts.resume else 0

to_pil = transforms.ToPILImage()

for epoch in range(config['n_epoch']):
    for it, (image_in, image_edge, targets) in enumerate(train_loader):
        images_i, images_e_i, images_r, images_e_r, images_s, image_m, fname_i, fname_e_i, fname_r, fname_e_r= image_in.cuda().detach(), image_edge.cuda().detach(), targets['albedo'].cuda().detach(), \
                                                targets['edge-albedo'].cuda().detach(), targets['shading'].cuda().detach(), targets['mask'].cuda().detach(), \
                                                targets['name_i'], targets['name_e_i'], targets['name_r'], targets['name_e_r']

        with Timer("<{}> [Epoch: {}] Elapsed time in update: %f".format(get_local_time(), epoch)):
            # ┌────────────────────────────────────────────────────────┐
            # │               Main training code                       │ 
            # └────────────────────────────────────────────────────────┘
            image_m = image_m > 0.1
            trainer.dis_update(images_i, images_e_i, images_r, images_e_r, images_s, config)
            trainer.gen_update(images_i, images_e_i, images_r, images_e_r, images_s, targets, config)
            torch.cuda.synchronize()

            trainer.update_learning_rate()

        # ┌────────────────────────────────────────────────────────────┐
        # │               Dump training stats in log file              │ 
        # └────────────────────────────────────────────────────────────┘
        if (iterations + 1) % config['log_iter'] == 0:
            print("<{}> Iteration: %08d/%08d".format(get_local_time()) % (iterations + 1, max_iter))
            write_loss(iterations, trainer, train_writer)
            
        # ┌────────────────────────────────────────────────────────────┐
        # │                     Write images                           │ 
        # └────────────────────────────────────────────────────────────┘
        if (iterations + 1) % config['image_save_iter'] == 0:
            with torch.no_grad():
                train_image_outputs = trainer.sample(train_display_images_i, train_display_images_e_i, train_display_images_r, train_display_images_e_r, train_display_images_s)
            write_2images(train_image_outputs, display_size, image_directory, 'train_%08d' % (iterations + 1))

        if (iterations + 1) % config['image_display_iter'] == 0:
            with torch.no_grad():
                image_outputs = trainer.sample(train_display_images_i, train_display_images_e_i, train_display_images_r, train_display_images_e_r, train_display_images_s)
            write_2images(image_outputs, display_size, image_directory, 'train_current')

        iterations = iterations + 1

    # ┌────────────────────────────────────────────────────────────────┐
    # │                   Save network weights                         │ 
    # └────────────────────────────────────────────────────────────────┘
    trainer.save(checkpoint_directory, iterations)
