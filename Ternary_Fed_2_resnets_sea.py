#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import sys
import copy
import torch
import numpy as np
from utils.config import Args
from utils.Evaluate import evaluate
# import utils.data_utils as data_utils
# from tools.Fed_Operator import ServerUpdate, LocalUpdate

# WY's add on or modification
import utils.data_utils_wy as data_utils_wy
from tools.Fed_Operator_sea import ServerUpdate, LocalUpdate
import time, csv
from itertools import zip_longest
import torch.nn as nn
from model.resnet_torch_sea import resnet50 as Fed_Model


# this is the code use WY's corrected FL training and evaluation part, which is the same as Ternary_Fed_2, and Tenary_Fed_wy2
# this script trains on Seagate's dataset, basing on the corrected resnet18 model

# seagate dataset has 7 classes, which is different from cifar-10, so need to specify
CLASS_NUM = 7

def choose_model(model, f_dict, ter_dict):
    tmp_net1 = model.load_state_dict(f_dict)
    tmp_net2 = model.load_state_dict(ter_dict)

    _, acc_1, _ = evaluate(tmp_net1, G_loss_fun, test_loader, Args)
    _, acc_2, _ = evaluate(tmp_net2, G_loss_fun, test_loader, Args)
    print('Unquantized fed model Acc: %.3f' % acc_1, 'Quntized fed model acc: %.3f' % acc_2)

    flag = False
    if np.abs(acc_1-acc_2) < 0.03:
        flag = True
        return ter_dict, flag
    else:
        return f_dict, flag


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

    # start the clock for the main script execution
    start_time_main = time.time()

    # get the data loader
    client_train_loaders, test_loader, _ = data_utils_wy.seagate_dataloader(args=Args)
    
    # set global network
    # G_net = Fed_Model(num_classes=CLASS_NUM, pretrained=True)
    G_net = Fed_Model(pretrained=True)

    # # for debug purpose, print out all the layer names or the architecture of the model
    # for name, para in G_net.named_parameters():
    #     print(name)
    # # print(G_net)

    # remove the last fc layer to do fine-tuning
    num_ftrs = G_net.fc.in_features
    # print(num_ftrs)
    G_net.fc = nn.Linear(num_ftrs, CLASS_NUM)
    
    print('Model to train: {}'.format(Args.model))
    

    # pause and print message for user to confirm the hyparameter are good to go
    answer = input("Press n to abort, press any other key to continue, then press ENTER: ")
    if answer == 'n':
        exit('\nTraining is aborted by user')
    print('\nTraining starts...\n'.format(Args.model))

    # set the number of participant clients
    m = 3

    # initialize the variable lists for the stats of accurcy
    gv_acc = []
    net_best = None
    val_acc_list, net_list = [], []
    num_s1 = 0
    # num_s2 = 0
    c_lists = [[] for i in range(Args.num_C)]
    
    # define loss for computing test acc
    G_loss_fun = torch.nn.CrossEntropyLoss()

    # copy weights
    w_glob = G_net.state_dict()

    # training starts
    G_net.train()
    
    for rounds in range(Args.rounds):
        start_time = time.time()
        w_locals = []
        client_id = np.random.choice(range(Args.num_C), m, replace=False)
        print('Round {:d} start'.format(rounds, client_id))
        num_samp = []
        
        # local update
        for idx in client_id:
            local = LocalUpdate(client_name = idx, c_round = rounds, train_iter = client_train_loaders[idx], test_iter = test_loader, wp_lists= c_lists[idx], args=Args)
            w, wp_lists = local.TFed_train(net=copy.deepcopy(G_net).to(Args.device))
            c_lists[idx] = wp_lists
            w_locals.append(copy.deepcopy(w))
            num_samp.append(len(client_train_loaders[idx].dataset))
        
        # update global weights
        w_glob, ter_glob = ServerUpdate(w_locals, num_samp)

        # load the unquantized global weights for test loss and accuracy evaluation
        G_net.load_state_dict(w_glob)

        # compute test loss and test accuracy
        g_loss, g_acc, _ = evaluate(G_net, G_loss_fun, test_loader, Args)
        gv_acc.append(g_acc)

        # download the global model weights to clients
        # this downloaded global weights is only userd as iterable for training, 
        # this downloaded global weights is not intented to be used for model publishing and prediction
        # for prediction after FL is done, the model lastly updated at server without quantization should be used
        if Args.fedmdl == 's1':
            # if performance drop of the quantized global model is less than 0.03, 
            # then clients download the quantized model
            w_glob_download, tmp_flag = choose_model(G_net, w_glob, ter_glob)
            if tmp_flag:
                # num_s2 += 1
                num_s1 += 1 # increase the number of execution of S1 strategy by 1
                print('S1')
        elif Args.fedmdl == 's2':
            # strategy s2 force to download unquantized global model
            w_glob_download = w_glob
        elif Args.fedmdl == 's3':
            # strategy s3 always download quantized globl model to clients regardless of the performance drops
            w_glob_download = ter_glob
        else:
            exit('Error: unrecognized quantization option for federated model')

        # download and load global weights after downlink quantization strategy selection
        G_net.load_state_dict(w_glob_download)

        end_time = time.time()
        time_elapsed = end_time-start_time

        print('Round {:3d}, Global loss {:.3f}, Global Acc {:.3f}, time elapsed: {:.2f}s ({:.2f}mins)'.format(rounds, g_loss, g_acc, time_elapsed, time_elapsed/60))
        
    end_time_main = time.time()
    time_elapsed_total = end_time_main - start_time_main
    print('Done! Time elapsed: {:.2f}hrs ({:.2f}mins))'.format(time_elapsed_total/3600,time_elapsed_total/60))
    
    if Args.fedmdl == 's1':
        print('Times of downloading quantized global model {:3d}/{:3d}'.format(num_s1, Args.rounds))
    elif Args.fedmdl == 's3':
        print('Times of downloading quantized global model {:3d}/{:3d}'.format(Args.rounds, Args.rounds))
    elif Args.fedmdl == 's2':
        print('Times of downloading quantized global model {:3d}/{:3d}'.format(0, Args.rounds))
    
    # WY's add on for recording results to csv files
    if Args.save_record:
        results = [torch.arange(1,Args.rounds+1).tolist(), gv_acc]
        export_data = zip_longest(*results, fillvalue = '')
        record_path_save = f'../save_sea/{Args.dataset}-{Args.model}-r{Args.rounds}-le{Args.local_e}-lb{Args.batch_size}-nc{Args.num_C}-lr{Args.lr}-{Args.fedmdl}-' + time.strftime('%y-%m-%d-%H-%M-%S.csv')
        with open(record_path_save, 'w', newline='') as file:
            writer = csv.writer(file,delimiter=',')
            writer.writerow(['Round', 'Test acc'])
            writer.writerows(export_data)