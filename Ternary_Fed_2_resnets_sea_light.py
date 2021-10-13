#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.8
import sys
import copy
import torch
import numpy as np
from utils.config import Args
from utils.Evaluate import evaluate
import utils.data_utils_wy as data_utils_wy
from tools.Fed_Operator_sea import ServerUpdate, LocalUpdate
import time, csv
from itertools import zip_longest
import torch.nn as nn
from model.resnet_torch_sea import resnet50 as Fed_Model



# seagate dataset has 7 classes
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
        print('\nRound {:d} start'.format(rounds, client_id))
        num_samp = []
        
        # local update
        for idx in client_id:
            local = LocalUpdate(client_name = idx, c_round = rounds, train_iter = client_train_loaders[idx], test_iter = test_loader, wp_lists= c_lists[idx], args=Args)
            w, wp_lists = local.TFed_train(net=copy.deepcopy(G_net).to(Args.device))
            c_lists[idx] = wp_lists
            w_locals.append(copy.deepcopy(w))
            num_samp.append(len(client_train_loaders[idx].dataset))
        
        # update global weights by model aggregation
        w_glob, ter_glob = ServerUpdate(w_locals, num_samp)

        # load weights of intermediate full precision (IFP) global model to evaluate
        # related test loss and accuracy evaluation
        G_net.load_state_dict(w_glob)

        # compute test loss and test accuracy, for the IFP global model
        g_loss, g_acc, _ = evaluate(G_net, G_loss_fun, test_loader, Args)
        
        # write the test accuracy of IFP global model to csv file
        gv_acc.append(g_acc)

        # print out the loss and accuracy for the IFP global model
        print('Round {:3d} | {:<30s} | loss {:.3f}, Acc {:.3f}'.format(rounds, 'Full-precision model', g_loss, g_acc))

        # download the global model weights to clients
        # this downloaded global weights is only used as iterable for training, 
        # this downloaded global weights is not intented to be used for model publishing and prediction
        # for prediction after FL is done, the model lastly updated at server without quantization should be used
        if Args.fedmdl == 's2':
            # strategy s2 force to download unquantized global model
            w_glob_download = w_glob
            print('Downloading full precision global model')
        elif Args.fedmdl == 's3':
            # strategy s3 always download quantized globl model to clients regardless of the performance drops
            w_glob_download = ter_glob
            print('Downloading quantized global model')
        else:
            exit('Error: unrecognized quantization option for federated model')

        # load the weights of downloaded global model to facilitate the iteration of next FL round
        G_net.load_state_dict(w_glob_download)

        # compute test loss and test accuracy again, for the actual global model to be downloaded by clients
        g_loss_d, g_acc_d, _ = evaluate(G_net, G_loss_fun, test_loader, Args)

        end_time = time.time()
        time_elapsed = end_time-start_time

        # print out the loss and accuracy for actual global model to be downloaded by clients
        print('Round {:3d} | {:<30s} | loss {:.3f}, Acc {:.3f}, time elapsed: {:.2f}s ({:.2f}mins)'.format(rounds, 'global model downladed', g_loss_d, g_acc_d, time_elapsed, time_elapsed/60))


    # main program ends here, do some summary and print out
    end_time_main = time.time()
    time_elapsed_total = end_time_main - start_time_main
    print('Done! Time elapsed: {:.2f}hrs ({:.2f}mins))'.format(time_elapsed_total/3600,time_elapsed_total/60))
    
    # record results to csv files
    if Args.save_record:
        results = [torch.arange(1,Args.rounds+1).tolist(), gv_acc]
        export_data = zip_longest(*results, fillvalue = '')
        record_path_save = f'../save_sea/seagate-resnet50-r{Args.rounds}-le{Args.local_e}-lb{Args.batch_size}-nc{Args.num_C}-lr{Args.lr}-{Args.fedmdl}-' + time.strftime('%y-%m-%d-%H-%M-%S.csv')
        with open(record_path_save, 'w', newline='') as file:
            writer = csv.writer(file,delimiter=',')
            writer.writerow(['Round', 'Test acc'])
            writer.writerows(export_data)