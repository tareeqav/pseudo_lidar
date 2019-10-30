from __future__ import print_function

import os
import time

import cv2
from skimage import io

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from matplotlib import pyplot as plt


import logger

from utils import preprocess
from models import *
import file_utils

def infer(model, imgL, imgR, img_idx):
    
    imgL = torch.FloatTensor(imgL).cuda()
    imgR = torch.FloatTensor(imgR).cuda()

    imgL, imgR= Variable(imgL), Variable(imgR)

    imgL, imgR= Variable(imgL), Variable(imgR)

    with torch.no_grad():
        output = model(imgL,imgR)
    
    output = torch.squeeze(output)
    pred_disp = output.data.cpu().numpy()

    return pred_disp

def main():

    img_idx = 0
    img_idx = str(img_idx).zfill(10)

    processed = preprocess.get_transform(augment=False)

    imgL_o = io.imread(f'./images/left/{img_idx}.png').astype('float32')
    imgR_o = io.imread(f'./images/right/{img_idx}.png').astype('float32')


    imgL = processed(imgL_o).numpy()
    imgR = processed(imgR_o).numpy()
    
    imgL = np.reshape(imgL,[1,3,imgL.shape[1],imgL.shape[2]])
    imgR = np.reshape(imgR,[1,3,imgR.shape[1],imgR.shape[2]])

    # pad to (384, 1248)
    top_pad = 384-imgL.shape[2]
    left_pad = 1248-imgL.shape[3]

    imgL = np.lib.pad(imgL,((0,0),(0,0),(top_pad,0),(0,left_pad)),mode='constant',constant_values=0)
    imgR = np.lib.pad(imgR,((0,0),(0,0),(top_pad,0),(0,left_pad)),mode='constant',constant_values=0)
    
    model = stackhourglass(192)
    model = nn.DataParallel(model, device_ids=[0])
    model.cuda()

    state_dict = torch.load('./weights/psmnet_finetune_300.tar')
    model.load_state_dict(state_dict['state_dict'])

    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    model.eval()
    pred_disp = infer(model, imgL, imgR, img_idx)
     
    top_pad   = 384-imgL_o.shape[0]
    left_pad  = 1248-imgL_o.shape[1]
    img = pred_disp[top_pad:,:-left_pad]
    io.imsave(f'images/disparity_{img_idx}.png',(img*256).astype('uint16'))
    
if __name__ == '__main__':
    main()
    