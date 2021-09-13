#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import sys
import copy
import torch
import numpy as np
from utils.config import Args
from utils.Evaluate import evaluate
import utils.data_utils as data_utils
from tools.Fed_Operator import ServerUpdate, LocalUpdate

# WY's add on
import time, csv
from itertools import zip_longest

if Args.model == 'MLP':
    from model.MLP import MLP as Fed_Model
elif Args.model == 'CNN':
    from model.CNN import CNN as Fed_Model
elif Args.model == 'ResNet':
    from model.resnet import ResNet18 as Fed_Model



def choose_model(f_dict, ter_dict):
    tmp_net1 = Fed_Model()
    tmp_net2 = Fed_Model()
    tmp_net1.load_state_dict(f_dict)
    tmp_net2.load_state_dict(ter_dict)

    _, acc_1, _ = evaluate(tmp_net1, G_loss_fun, test_iter, Args)
    _, acc_2, _ = evaluate(tmp_net2, G_loss_fun, test_iter, Args)
    print('Acc FedAvg: %.3f' % acc_1, 'Acc T-FedAvg: %.3f' % acc_2)

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
    print("selected gpu id: ",Args.gpu_id)
    torch.cuda.device(Args.gpu_id)
    device = 'cuda'
    print("current gpu id: ",torch.cuda.current_device())
    print("current gpu name: ",torch.cuda.get_device_name(torch.cuda.current_device()))
    
    torch.manual_seed(Args.seed)

    start_time_main = time.time()

    C_iter, train_iter, test_iter, stats = data_utils.get_dataset(args=Args)
    # build global network
    G_net = Fed_Model()
    # print(G_net)
    print("Training ",Args.model)
    G_net.train()
    G_loss_fun = torch.nn.CrossEntropyLoss()


    # copy weights
    w_glob = G_net.state_dict()

    m = max(int(Args.frac * Args.num_C), 1)

    gv_acc = []

    net_best = None
    val_acc_list, net_list = [], []
    num_s1 = 0
    # num_s2 = 0

    # training
    c_lists = [[] for i in range(Args.num_C)]
    for rounds in range(Args.rounds):
        start_time = time.time()
        w_locals = []
        client_id = np.random.choice(range(Args.num_C), m, replace=False)
        print('Round {:d} start'.format(rounds, client_id))
        num_samp = []
        for idx in client_id:
            local = LocalUpdate(client_name = idx, c_round = rounds, train_iter = C_iter[idx], test_iter = test_iter, wp_lists= c_lists[idx], args=Args)
            w, wp_lists = local.TFed_train(net=copy.deepcopy(G_net).to(Args.device))
            c_lists[idx] = wp_lists
            w_locals.append(copy.deepcopy(w))

            num_samp.append(len(C_iter[idx].dataset))
        # update global weights
        w_glob, ter_glob = ServerUpdate(w_locals, num_samp)

        # if performance of ternary model is smaller than 0.03 then send quantized ternary model back to clients
        w_glob, tmp_flag = choose_model(w_glob, ter_glob)        
        if tmp_flag:
            # num_s2 += 1
            num_s1 += 1 # increase the number of execution of S1 strategy by 1
            print('S1')

        # reload global network weights
        G_net.load_state_dict(w_glob)

        #verify accuracy on test set
        g_loss, g_acc, g_acc5 = evaluate(G_net, G_loss_fun, test_iter, Args)
        gv_acc.append(g_acc)

        end_time = time.time()
        time_elapsed = end_time-start_time

        print('Round {:3d}, Global loss {:.3f}, Global Acc {:.3f}, time elapsed: {:.2f}s ({:.2f}mins)'.format(rounds, g_loss, g_acc, time_elapsed, time_elapsed/60))
        
    end_time_main = time.time()
    time_elapsed_total = end_time_main - start_time_main
    print('Done! Time elapsed: {:.2f}hrs ({:.2f}mins))'.format(time_elapsed_total/3600,time_elapsed_total/60))
    print('Times of downloading quantized global model {:3d}/{:3d}'.format(num_s1, Args.rounds))

    # WY's add on for recording results to csv files
    if Args.save_record:
        results = [torch.arange(1,Args.rounds+1).tolist(), gv_acc]
        export_data = zip_longest(*results, fillvalue = '')
        record_path_save = f'./save/{Args.dataset}-{Args.model}-r{Args.rounds}-le{Args.local_e}-lb{Args.batch_size}-nc{Args.num_C}-lr{Args.lr}' + time.strftime('%y-%m-%d-%H-%M-%S.csv')
        with open(record_path_save, 'w', newline='') as file:
            writer = csv.writer(file,delimiter=',')
            writer.writerow(['Round', 'Test acc'])
            writer.writerows(export_data)