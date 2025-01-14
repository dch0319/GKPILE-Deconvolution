from __future__ import print_function

import argparse
import os
import re

from networks.skip import skip
import glob
from skimage.io import imsave
import warnings
from tqdm import tqdm
from torch.optim.lr_scheduler import MultiStepLR
from utils.common_utils import *

import torch.nn.functional as F
from SSIM import SSIM
from networks.knet import Generator, ResNet18
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

parser = argparse.ArgumentParser()

parser.add_argument('--num_iter', type=int, default=5000, help='number of epochs of training')
parser.add_argument('--img_size', type=int, default=[256, 256], help='size of each image dimension')
parser.add_argument('--kernel_size', type=int, default=31, help='size of blur kernel')
parser.add_argument('--data_path', type=str, default="./datasets/lai/uniform", help='path to blurry image')
parser.add_argument('--gt_path', type=str, default="./datasets/lai/ground_truth", help='path to gt image')
parser.add_argument('--models_path', type=str, default='./models/lai', help='path to save the model file')
parser.add_argument('--save_path', type=str, default="./results/lai", help='path to save results')
parser.add_argument('--save_frequency', type=int, default=1000, help='lfrequency to save results')

opt = parser.parse_args()

print(opt)

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
dtype = torch.cuda.FloatTensor

warnings.filterwarnings("ignore")

files_source = glob.glob(os.path.join(opt.data_path, '*.png'))
files_source.sort()
save_path = opt.save_path
os.makedirs(save_path, exist_ok=True)


def get_kernel_network(kernel_size):
    netG_path = opt.models_path + '/' + 'netG_{}.pth'.format(kernel_size)
    netE_path = opt.models_path + '/' + 'netE_{}.pth'.format(kernel_size)

    netG = Generator(kernel_size).cuda()
    netE = ResNet18().cuda()

    netG_state_dict = torch.load(netG_path)
    netE_state_dict = torch.load(netE_path)

    # 判断权重文件是否使用 DataParallel
    if any(key.startswith('module.') for key in netG_state_dict.keys()):
        netG = torch.nn.DataParallel(netG)
    if any(key.startswith('module.') for key in netE_state_dict.keys()):
        netE = torch.nn.DataParallel(netE)

    netG.load_state_dict(netG_state_dict)
    netE.load_state_dict(netE_state_dict)

    for p in netG.parameters():
        p.requires_grad = False
    netG.eval()

    for p in netE.parameters():
        p.requires_grad = False
    netE.eval()

    return netE, netG


for f in files_source:
    INPUT = 'noise'
    pad = 'reflection'
    LR = 0.01
    num_iter = opt.num_iter
    reg_noise_std = 0.001

    path_to_image = f
    imgname = os.path.basename(f)
    imgname = os.path.splitext(imgname)[0]

    if imgname.find('kernel_01') != -1:
        opt.kernel_size = 31
    if (imgname.find('kernel_02') != -1) or (imgname.find('kernel_03') != -1):
        opt.kernel_size = 55
    if imgname.find('kernel_04') != -1:
        opt.kernel_size = 75

    print(f'imgname={imgname}, kernel_size={opt.kernel_size}')

    netE, netG = get_kernel_network(opt.kernel_size)

    new_path = os.path.join(opt.save_path, '%s' % imgname)
    os.makedirs(new_path, exist_ok=True)
    imgs, y = get_color_image(path_to_image, -1)  # load image and convert to np.
    img_blur = np_to_torch(imgs).type(dtype)
    pattern = r'_kernel_0\d'
    img_gt, _ = get_color_image(os.path.join(opt.gt_path, re.sub(pattern, '', imgname) + '.png'), -1)
    img_gt = img_gt.transpose(1, 2, 0)
    y = np_to_torch(y).type(dtype)

    img_size = imgs.shape
    padh, padw = opt.kernel_size - 1, opt.kernel_size - 1
    opt.img_size[0], opt.img_size[1] = img_size[1] + padh, img_size[2] + padw

    input_depth = 8

    net_input = get_noise(input_depth, INPUT, (opt.img_size[0], opt.img_size[1])).type(dtype)

    net = skip(input_depth, 3,
               num_channels_down=[128, 128, 128, 128, 128],
               num_channels_up=[128, 128, 128, 128, 128],
               num_channels_skip=[16, 16, 16, 16, 16],
               upsample_mode='bilinear',
               need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

    net = net.type(dtype)

    z = netE(y.unsqueeze(0))
    if isinstance(netG, torch.nn.DataParallel):
        w = netG.module.g1(z)
    else:
        w = netG.g1(z)
    w.requires_grad = True
    if isinstance(netG, torch.nn.DataParallel):
        out_k = netG.module.Gk(w)
    else:
        out_k = netG.Gk(w)

    # Losses
    mse = torch.nn.MSELoss().type(dtype)
    ssim = SSIM().type(dtype)

    optimizerI = torch.optim.Adam([{'params': net.parameters()}, {'params': [w], 'lr': 5e-4}], lr=LR)
    schedulerI = MultiStepLR(optimizerI, milestones=[2000, 3000, 4000], gamma=0.5)

    net_input_saved = net_input.detach().clone()

    save_path = os.path.join(new_path, 'initialization_k.png')
    out_k_np = torch_to_np(out_k)
    out_k_np = out_k_np.squeeze()
    out_k_np /= np.max(out_k_np)
    save_img_np(save_path, out_k_np)
    psnr_max = 0

    for step in tqdm(range(1, num_iter + 1)):

        # input regularization
        net_input = net_input_saved + reg_noise_std * torch.zeros(net_input_saved.shape).type_as(
            net_input_saved.data).normal_()

        # change the learning rate
        schedulerI.step(step)
        optimizerI.zero_grad()

        # get the network output
        out_x = net(net_input)
        if isinstance(netG, torch.nn.DataParallel):
            out_k = netG.module.Gk(w)  # 1,1,31,31
        else:
            out_k = netG.Gk(w)
        out_img = F.conv2d(out_x, out_k.repeat(3, 1, 1, 1), groups=3)

        if step <= 500:
            total_loss = mse(out_img, img_blur)
        else:
            total_loss = 1 - ssim(out_img, img_blur)
        total_loss.backward()
        optimizerI.step()

        # if (step + 1) % opt.save_frequency == 0:
        out_x_np = torch_to_np(out_x).transpose(1, 2, 0)
        out_x_np = out_x_np[padh // 2:padh // 2 + img_size[1], padw // 2:padw // 2 + img_size[2], 0:3]
        out_x_np = np.clip(out_x_np, 0, 1)
        psnr = compare_psnr(img_gt, out_x_np)
        # tqdm.write(f"PSNR={psnr}")
        if psnr > psnr_max:
            psnr_max = psnr
            save_path = os.path.join(new_path, 'x.png')
            save_img_np(save_path, out_x_np)
            save_path = os.path.join(new_path, 'k.png')
            out_k_np = torch_to_np(out_k)
            out_k_np = out_k_np.squeeze()
            out_k_np /= np.max(out_k_np)
            save_img_np(save_path, out_k_np)
