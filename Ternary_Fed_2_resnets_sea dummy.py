#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
import torch.nn as nn
from utils.config import Args
# WY's add on or modification
import utils.data_utils_wy as data_utils_wy
from tools.Fed_Operator_sea import ServerUpdate, LocalUpdate
from model.resnet_torch_sea import resnet50 as Fed_Model


# this is the code use WY's corrected FL training and evaluation part, which is the same as Ternary_Fed_2, and Tenary_Fed_wy2
# this script trains on Seagate's dataset, basing on the corrected resnet18 model

# seagate dataset has 7 classes, which is different from cifar-10, so need to specify
CLASS_NUM = 7

# define a function to make seagate's resnet50
def sea_model(pretrained_flag=True):
    # load standard resnet
    model = Fed_Model(pretrained=pretrained_flag)

    # customize the last fc layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, CLASS_NUM)

    return model

import time
  
# define the countdown func.
def countdown(t):
    while t:
        mins, secs = divmod(t, 60)
        timer = '{:02d}:{:02d}'.format(mins, secs)
        print(timer, end="\r")
        time.sleep(1)
        t -= 1
      
    print('Fire in the hole!!') 
  
      

if __name__ == '__main__':

    # check torch version
    print(torch.__version__)
    
    # check gpu usage info
    print("using gpu: ",torch.cuda.is_available())
    print("available gpu number: ",torch.cuda.device_count())
    # print("selected gpu id: ", Args.gpu_id)
    # torch.cuda.device(Args.gpu_id)
    print("current gpu id: ",torch.cuda.current_device())
    print("current gpu name: ",torch.cuda.get_device_name(torch.cuda.current_device()))
    device = 'cuda'

    # cuda instructions to show gpu memory usage status
    torch.cuda.empty_cache()

    # set the randomization seed
    torch.manual_seed(Args.seed)

    # get the data loader
    client_train_loaders, test_loader, _ = data_utils_wy.seagate_dataloader(args=Args)
    
    # set global network
    G_net = sea_model()

    print('Quantization strategy: ', Args.fedmdl)
    print('Num of global rounds: ', Args.rounds)
    print('Local batch size: ', Args.batch_size)
    print('Num of local epoch: ', Args.local_e)
    print('Learning rate: ', Args.lr)
    print('Save results: ', Args.save_record)

    # function call
    t=10
    countdown(int(t))  