#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import sys
import copy
from numpy.core.fromnumeric import shape
import torch
import numpy as np
from torch.nn.modules.container import ModuleList
from utils.config import Args
from utils.Evaluate import evaluate,evaluate2
# import utils.data_utils as data_utils
# from tools.Fed_Operator import ServerUpdate, LocalUpdate

# WY's add on or modification
import utils.data_utils_wy as data_utils_wy
from tools.Fed_Operator_sea import ServerUpdate, LocalUpdate
import time, csv
from itertools import zip_longest
import torch.nn as nn
from model.resnet_torch_sea import Quantized_resnet
from model.resnet_torch_sea import resnet50 as Fed_Model
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler


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
    G_net = sea_model()

    # copy weights
    w_glob = G_net.state_dict()

    # for debug purpose, print out all the layer names or the architecture of the model
    # print(G_net)
    # for name, para in G_net.named_parameters():
    #     if 'conv1' in name and 'layer' not in name:
    #     # if ('conv' in name or 'downsample.0' in name) and ('layer' in name):
    #         print(name)
    
    numel_conv1=0
    numel_layer1=0
    numel_layer2=0
    numel_layer3=0
    numel_layer4=0
    numel_fc=0
    for name, para in G_net.named_parameters():
        if 'conv1' in name and 'layer' not in name:
            numel_conv1+=para.numel()
        elif ('conv' in name or 'downsample.0' in name) and ('layer1' in name):
            numel_layer1+=para.numel()
        elif ('conv' in name or 'downsample.0' in name) and ('layer2' in name):
            numel_layer2+=para.numel()
        elif ('conv' in name or 'downsample.0' in name) and ('layer3' in name):
            numel_layer3+=para.numel()
        elif ('conv' in name or 'downsample.0' in name) and ('layer4' in name):
            numel_layer4+=para.numel()
        elif 'fc.weight' in name:
            numel_fc+=para.numel()
    print('Num of params in conv1:', numel_conv1)
    print('Num of params in layer1:', numel_layer1)
    print('Num of params in layer2:', numel_layer2)
    print('Num of params in layer3:', numel_layer3)
    print('Num of params in layer4:', numel_layer4)
    print('Num of params in fc:', numel_fc)
    print('Num of params in total:',sum(p.numel() for p in G_net.parameters()))

    print('\nNow show the keys in model weights')
    for key, kernel in w_glob.items():
        print(key)

    # _,_,optimizer = Quantized_resnet(G_net,Args)
    
    # print(len(optimizer.param_groups[0]['params']))

    # for kernel in optimizer.param_groups[0]['params']:
    #     print(kernel.data.size())
    #     print(kernel.data.numel())
    # for kernel in optimizer.param_groups[1]['params']:
    #     print(kernel.data.size())
    #     print(kernel.data.numel())
    # print(optimizer.param_groups[1])
    
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
    c_lists = [[] for i in range(Args.num_C)]
    
    # define loss for computing test acc
    G_loss_fun = torch.nn.CrossEntropyLoss()

    # # Define optimizer and scheduler, reference to Seagate's configuration
    # optimizer = optim.Adam(G_net.parameters(), lr=Args.lr)
    # lmbda = lambda epoch: 0.9 # could be further fine-tuned
    # scheduler = lr_scheduler.MultiplicativeLR(optimizer, lmbda) # could be further fine-tuned

    # training starts
    G_net.train()
    
    for rounds in range(Args.rounds):
        start_time = time.time()
        w_locals = []
        client_id = np.random.choice(range(Args.num_C), m, replace=False)
        print('\nRound {:d} start'.format(rounds, client_id))
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

        # load weights of intermediate full precision (IFP) global model, for test loss and accuracy evaluation
        G_net.load_state_dict(w_glob)

        # compute test loss and test accuracy, for the IFP global model
        # g_loss, g_acc, _ = evaluate(G_net, G_loss_fun, test_loader, Args)
        g_loss, g_acc = evaluate2(G_net, test_loader, Args)
        
        # write the test accuracy of IFP global model to csv file
        gv_acc.append(g_acc)

        # download the global model weights to clients
        # this downloaded global weights is only userd as iterable for training, 
        # this downloaded global weights is not intented to be used for model publishing and prediction
        # for prediction after FL is done, the model lastly updated at server without quantization should be used
        if Args.fedmdl == 's1':
            G_net.load_state_dict(ter_glob)
            g_loss_t, g_acc_t = evaluate2(G_net, test_loader, Args)
            end_time = time.time()
            time_elapsed = end_time-start_time 
            if g_acc - g_acc_t < 0.03:
                num_s1 += 1
                print('Downloading quantized global model')
                print('Round {:3d} | {:<30s} | Acc {:.4f}, loss {:.4f}'.format(rounds, 'Global model at server', g_acc, g_loss))
                print('Round {:3d} | {:<30s} | Acc {:.4f}, loss {:.4f}'.format(rounds, 'Global model downloaded', g_acc_t, g_loss_t))
                print('Round {:3d} | {:<30s} | Acc {:.4f}'.format(rounds, 'Performance difference', g_acc-g_acc_t))
            else:
                G_net.load_state_dict(w_glob)
                print('Downloading full precision global model')
                print('Round {:3d} | {:<30s} | Acc {:.4f}, loss {:.4f}'.format(rounds, 'Global model at server', g_acc, g_loss))
            print('Round {:3d} | Time elapsed: {:.2f}s ({:.2f}mins)'.format(rounds, time_elapsed, time_elapsed/60))
        elif Args.fedmdl == 's2':
            end_time = time.time()
            time_elapsed = end_time-start_time
            print('Downloading full precision global model')
            print('Round {:3d} | {:<30s} | Acc {:.4f}, loss {:.4f}'.format(rounds, 'Global model at server', g_acc, g_loss))
            print('Round {:3d} | Time elapsed: {:.2f}s ({:.2f}mins)'.format(rounds, time_elapsed, time_elapsed/60))
        elif Args.fedmdl == 's3':
            G_net.load_state_dict(ter_glob)
            g_loss_t, g_acc_t = evaluate2(G_net, test_loader, Args)
            end_time = time.time()
            time_elapsed = end_time-start_time
            print('Downloading quantized global model')
            print('Round {:3d} | {:<30s} | Acc {:.4f}, loss {:.4f}'.format(rounds, 'Global model at server', g_acc, g_loss))
            print('Round {:3d} | {:<30s} | Acc {:.4f}, loss {:.4f}'.format(rounds, 'Global model downloaded', g_acc_t, g_loss_t))
            print('Round {:3d} | {:<30s} | Acc {:.4f}'.format(rounds, 'Performance difference', g_acc-g_acc_t))
            print('Round {:3d} | Time elapsed: {:.2f}s ({:.2f}mins)'.format(rounds, time_elapsed, time_elapsed/60))
        else:
            exit('Error: unrecognized quantization option for federated model')

        # scheduler.step()

    end_time_main = time.time()
    time_elapsed_total = end_time_main - start_time_main
    print('Done! Time elapsed: {:.2f}hrs ({:.2f}mins))'.format(time_elapsed_total/3600,time_elapsed_total/60))
    
    # if Args.fedmdl == 's1':
    #     print('Times of downloading quantized global model {:3d}/{:3d}'.format(num_s1, Args.rounds))

    
    # WY's add on for recording results to csv files
    if Args.save_record:
        results = [torch.arange(1,Args.rounds+1).tolist(), gv_acc]
        export_data = zip_longest(*results, fillvalue = '')
        record_path_save = f'../save_sea/seagate-resnet50-r{Args.rounds}-le{Args.local_e}-lb{Args.batch_size}-nc{Args.num_C}-lr{Args.lr}-{Args.fedmdl}-' + time.strftime('%y-%m-%d-%H-%M-%S.csv')
        with open(record_path_save, 'w', newline='') as file:
            writer = csv.writer(file,delimiter=',')
            writer.writerow(['Round', 'Test acc'])
            writer.writerows(export_data)