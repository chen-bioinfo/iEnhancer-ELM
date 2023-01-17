import os
import time
import torch
import argparse
import numpy as np
import pandas as pd
from util import * 
from model import *
from transformers import BertTokenizer
from pandas import DataFrame

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--epoches',              type=int,  default=40,  help='')
parser.add_argument('--batch_size',           type=int,  default=24,  help='')
parser.add_argument('--max_length',           type=int,  default=2000, help='')
parser.add_argument('--learning_rate',        type=float, default=1e-4, help="")
parser.add_argument('--model_path',           type=str,  default="../3-new-12w-0", help='')
parser.add_argument('--ind_filename',  type=str,  default="../dataset/enhancer_3-mer_DNABERT_ind.txt", help='')
parser.add_argument('--tra_filename',  type=str,  default="../dataset/enhancer_3-mer_DNABERT_tra.txt", help='')

args = parser.parse_args(args=[]) # 如果不使用"args=[]"，会报错

np.random.seed(6666)
random_list = np.random.randint(10000, 60000, size=(10,))

# average [3,4,5,6]
# 添加L2-正则化 average
mers = [3,4,5,6]
lambds = [0.5e-4, 0.5e-4, 0.5e-4, 0.5e-4]
learning_rates = [0.3e-5, 0.3e-5, 0.3e-5, 0.3e-5]

# cell_lines = ["GM12878", "HUVEC", "HSMM", "K562", "HEK293", "NHLF", "NHEK", "HMEC"]
cell_line = "GM12878"
log_filename = "../result/log_concate_{}.txt".format(cell_line)
log_file = open(log_filename, "w")
content = "state\ttra_loss\ttra_acc\ttra_mcc\ttra_sn\ttra_sp\ttest_loss\ttest_acc\ttest_mcc\ttest_sn\ttest_sp\n"
log_file.write(content)
log_file.flush()

for i in range(0,4):
    mer = mers[i]
    seed = random_list[i]
    lambd = lambds[i]
    learning_rate = learning_rates[i]

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    # for _ in range(1):
    args.model_path = "../{}-new-12w-0".format(mer)
    prefix = "/home/lijiahao/project/Enhancer/Enhancer-IF/dataset/process"

    args.ind_filename = os.path.join(prefix, "test-dataset(all-length)/{}-test-{}mer.txt".format(cell_line, mer)) 
    args.tra_filename = os.path.join(prefix, "train-dataset(all-length)/{}-train-{}mer.txt".format(cell_line, mer)) 
    
    tra_dataloader, val_dataloader = getData_concate(args, validation=True, training=True, shuffle=True)
    ind_dataloader = getData_concate(args, validation=False, training=False, shuffle=True)
    print("tra: {}; ind: {}".format(len(tra_dataloader), len(ind_dataloader)))
     
    args.learning_rate = learning_rate

    model = C_Bert_2FC_concatenate.from_pretrained(args.model_path, num_labels=1).to(device)
    print("lr: {}, lambd:{}, seed:{}, dropout:0.30, 缩放: 0.98 768->25->1, L2".format(learning_rate, lambd, "random"))

    epoches = args.epoches
    learning_rate = args.learning_rate
    tokenizer = BertTokenizer.from_pretrained(args.model_path)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08,)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.98)    # exponential decay 

    FGMmodel = FGM(model)

    gap = 0.1 # 0.2, 0.08, 0.05, 0.02
    acc_max = 0.0
    for epoch in range(epoches):
        
        if epoch < 10:
            optimizer.state_dict()['param_groups'][0]['lr'] = learning_rate * (epoch+1)/10.0
        else:
            scheduler.step()

        start_time = time.time()
        tra_acc, tra_mcc, tra_sn, tra_sp, tra_loss = train_finetuning_Norm_concate(tokenizer, model, tra_dataloader, optimizer, args, lambd, normal=2, kmer=mer, FGMmodel=FGMmodel)
        val_acc, val_mcc, val_sn, val_sp, val_loss = testing_finetuning_concate(tokenizer, model, val_dataloader, optimizer)
        ind_acc, ind_mcc, ind_sn, ind_sp, ind_loss = testing_finetuning_concate(tokenizer, model, ind_dataloader, optimizer)
        end = time.time()

        state = "Concate-{}mer-epoch={}-seed={}-lr={}-lambd={}\t".format(mer, epoch, seed, learning_rate ,lambd)
        tra_content = "{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t".format(tra_loss, tra_acc, tra_mcc, tra_sn, tra_sp)
        val_content = "{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t".format(val_loss, val_acc, val_mcc, val_sn, val_sp)
        ind_content = "{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\n".format(ind_loss, ind_acc, ind_mcc, ind_sn, ind_sp)

        content = state + tra_content + val_content + ind_content
        log_file.write(content)
        log_file.flush()
        print(content[0:-1], end="; Time{:.5f}\n".format(end-start_time))

        if np.abs(val_sn - val_sp) < gap and val_acc < max_acc:
            val_acc = max_acc
            torch.save(model.state_dict(), "./model/{}_8/C_Bert_2FC_concatenate_{}_{}mer_epoch={}.pt".format(cell_line, cell_line, mer, epoch))
        