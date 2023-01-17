import torch
import numpy as np
import pandas as pd
import math
from torch import optim
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from model import NewsDataset, NewDataset_finetuning

def getDataloader(dataset, args, training, eplison=0, shuffle=False):
    model_path = args.model_path
    max_length = args.max_length
    batch_size = args.batch_size
    if shuffle is None:
        shuffle = training

    dataset_text, dataset_label = list(dataset["text"]), list(dataset["label"])
    
    if training and eplison>0:
        dataset_label = label_smoothing(dataset_label, eplison, training)

    tokenizer = BertTokenizer.from_pretrained(model_path) 
    dataset_encoding = tokenizer(dataset_text, truncation = True, padding = True, max_length = max_length)

    train_dataset = NewsDataset(dataset_encoding, dataset_label)
    dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = shuffle)
    
    return dataloader

# return a dataloader
# split, is used to split validation dataset from train tNone
# validation, is used to identify where the data come from, train_data_file or ind_data_file
def getData(args, split=False, validation=False, eplison=0, shuffle=False):
    if validation:
        data_filename = args.ind_filename
        training = False
    else:
        data_filename = args.tra_filename
        training = True
        
    if split:
        df_raw = pd.read_csv(data_filename, sep="\t",header=None,names=["text","label"]) 
        tra_set, val_set = train_test_split(df_raw, stratify=df_raw['label'], test_size=0.1, random_state=42)
        tra_dataloader = getDataloader(tra_set, args, training, eplison, shuffle=shuffle)
        val_dataloader = getDataloader(val_set, args, training, eplison, shuffle=shuffle)
        
        return tra_dataloader, val_dataloader
    
    else:
        df_raw = pd.read_csv(data_filename, sep="\t",header=None,names=["text","label"]) 
        ind_dataloader = getDataloader(df_raw, args, training, eplison, shuffle=shuffle)
        return ind_dataloader

def label_smoothing(data_labels, eplison, training):
    if training:
        length = len(data_labels)
        for index in range(length):
            if data_labels[index] == 0:
                data_labels[index] = eplison
            else:
                data_labels[index] = 1-eplison
    return data_labels


def flat_accuracy(logits, label_true):
    length = len(label_true)
    
    true_count = 0
    for i in range(length):
        if logits[i] < 0.5 and label_true[i]==0:
            true_count += 1
        elif logits[i] > 0.5 and label_true[i]==1:
            true_count += 1
    return true_count/length

# caculate the confusion matrix
# and then return the evaluation criteria
def evaluation_criterion_temp(logits, label_true):
    length = label_true.shape[0]
    TP, FP, TN, FN = 0, 0, 0, 0
    for i in range(length):
        if logits[i]>0.5 and label_true[i]>0.5:
            TP += 1
        elif logits[i]>0.5 and label_true[i]<0.5:
            FP += 1
        elif logits[i]<0.5 and label_true[i]<0.5:
            TN += 1
        elif logits[i]<0.5 and label_true[i]>0.5:
            FN += 1
    
    ACC = (TP + TN)/(TP + TN + FP + FN)
    Sn = TP / (TP + FN)
    Sp = TN / (TN + FP)
    MCC = (TP*TN-FP*FN)/math.sqrt((TP+FN)*(TP+FP)*(FP+TN)*(FN+TN))
     
    return ACC, MCC, Sn, Sp

def evaluation_criterion(logits, label_true):
    length = logits.shape[0]
    label_pre = []
    label_tru = []
    
    threshold = 0.55
    
    for i in range(length):
        if logits[i] > threshold:
            label_pre.append(1)
        else:
            label_pre.append(0)
            
    for i in range(length):
        if label_true[i] > threshold:
            label_tru.append(1)
        else:
            label_tru.append(0)
    
    label_true = label_true.tolist()
    
    TN, FP, FN, TP = confusion_matrix(label_tru, label_pre).ravel()
    
    ACC = (TP + TN)/(TP + TN + FP + FN)
    Sn = TP / (TP + FN)
    Sp = TN / (TN + FP)
    MCC = (TP*TN-FP*FN)/math.sqrt((TP+FN)*(TP+FP)*(FP+TN)*(FN+TN))
    
    return ACC, MCC, Sn, Sp

def sigmoid(x):
    return 1.0/(1+np.exp(-x))

def evaluation_criterion_Sigmoid(logits, label_true):
    logits_sigmoid = []
    for logit in logits:
        logits_sigmoid.append(sigmoid(logit))
    
    logits_sigmoid = np.array(logits_sigmoid)
    
    length = logits.shape[0]
    label_pre = []
    label_tru = []
    
    threshold = 0.50
    
    for i in range(length):
        if logits_sigmoid[i] > threshold:
            label_pre.append(1)
        else:
            label_pre.append(0)
            
    for i in range(length):
        if label_true[i] > threshold:
            label_tru.append(1)
        else:
            label_tru.append(0)
    
    label_true = label_true.tolist()
    
    TN, FP, FN, TP = confusion_matrix(label_tru, label_pre).ravel()
    
    ACC = (TP + TN)/(TP + TN + FP + FN)
    Sn = TP / (TP + FN)
    Sp = TN / (TN + FP)
    MCC = (TP*TN-FP*FN)/math.sqrt((TP+FN)*(TP+FP)*(FP+TN)*(FN+TN))
    
    return ACC, MCC, Sn, Sp


def train(model, args, tra_dataloader, val_dataloader, ind_dataloader):
    total_train_loss = 0
    train_acc = 0.0

    epoches = args.number_of_epoches
    learning_rate = args.learning_rate
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08)
    total_steps = len(tra_dataloader) * 1
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = 0, # Default value in run_glue.py
                                                num_training_steps = total_steps*10)
    
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for epoch in range(epoches):
        total_train_loss = 0
        train_acc = 0.0
        for batch_data in tra_dataloader:
            # forward
            optimizer.zero_grad()
            input_ids = batch_data['input_ids'].to(device)
            attention_mask = batch_data['attention_mask'].to(device)
            labels = batch_data['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            pre_label = outputs[1]
        
            total_train_loss += loss
            train_acc += flat_accuracy(pre_label, labels)

            # gradient back
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # update the parameters
            optimizer.step()
            scheduler.step()
        
        print("epoch:%2d, trian loss:%.4f, acc:%.4f"%(epoch, total_train_loss/total_steps, train_acc/total_steps), end="; ")
        
        print("val ", end="")
        validation(model, val_dataloader)
        
        print("test ", end="")
        validation(model, ind_dataloader)
        print()
    
def validation(model, dataloader):
    model.eval()
    total_eval_accuracy = 0
    total_eval_loss = 0
    total_num = len(dataloader)
    
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for batch_data in dataloader:
        with torch.no_grad():
            # forward
            input_ids = batch_data['input_ids'].to(device)
            attention_mask = batch_data['attention_mask'].to(device)
            labels = batch_data['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        
        loss = outputs[0]
        logits = outputs[1]

        total_eval_loss += loss.item()
        label_ids = labels.to('cpu').numpy()

        total_eval_accuracy += flat_accuracy(logits, label_ids)
        
    print("loss: %.4f, acc: %.4f" % (total_eval_loss/total_num, total_eval_accuracy/total_num), end=";  ")

# train, and add the L2-normalization into the loss
# lambd is the weight, which means lambd*L2
# normal indicates the type of the demonstration number
def train_finetuning_Norm(model, tra_dataloader, optimizer, args, lambd, normal, kmer=3):
    tra_acc, tra_loss = [], []
    real_labels, pre_labels = [], []
    train_iter = len(tra_dataloader)*1.0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   
    train_loss = 0

    for batch_data in tra_dataloader:
        # forward
        outputs = train_step(model, batch_data, optimizer, lambd, normal, kmer=kmer)

        loss = outputs[0].detach()
        train_loss += loss.item()
            
        logits = outputs[1].detach()
        labels = batch_data['labels'].to(device)
            
        if len(real_labels) == 0:
            real_labels = labels
            pre_labels = logits
        else:
            real_labels = torch.cat([real_labels, labels], dim=0)
            pre_labels = torch.cat([pre_labels, logits], dim=0)
 
    tra_loss = train_loss/(train_iter)
    acc, mcc, sn, sp = evaluation_criterion(pre_labels.to('cpu').numpy(), real_labels.to('cpu').numpy())
        
    return acc, tra_loss

# one
def train_step(model, batch_data, optimizer, lambd, normal, kmer=3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.train()
    optimizer.zero_grad()

    input_ids = batch_data["input_ids"].to(device)
    attention_mask = batch_data["attention_mask"].to(device)
    labels = batch_data["labels"].to(device)

    outputs = model(input_ids, attention_mask, labels, kmer=3)
            
    loss = outputs[0]
    lambd = torch.tensor(lambd, requires_grad = True)
    
    # add normalization
    if normal == 1:   # L1-normalization
        L1_loss = torch.tensor(0.0, requires_grad = False).to(device)
        for param in model.parameters():
            L1_loss += torch.norm(param, p=1)
        loss += lambd * L1_loss    
    else:             # L2-normalization
        L2_loss = torch.tensor(0.0, requires_grad = True).to(device)
        for param in model.parameters():
            L2_loss += torch.norm(param, p=2)
        loss += lambd * L2_loss

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
    optimizer.step()
    
    return outputs


def validation_finetuning(model, dataloader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    total_iter = len(dataloader)*1.0
    test_loss, test_TP, test_FP, test_TN, test_FN = 0, 0, 0, 0, 0
    ACC, MCC, Sn, Sp = 0, 0, 0, 0
    real_labels, pre_labels = [], []
    
    for batch_data in dataloader:
        model.eval()
        with torch.no_grad():
            
            input_ids = batch_data["input_ids"].to(device)
            attention_mask = batch_data["attention_mask"].to(device)
            labels = batch_data["labels"].to(device)
            
            outputs = model(input_ids, attention_mask, labels)
        
        loss = outputs[0]
        test_loss += loss.item()
        
        logits = outputs[1]
        
        if len(real_labels) == 0:
            real_labels = labels
            pre_labels = logits
        else:
            real_labels = torch.cat([real_labels, labels], dim=0)
            pre_labels = torch.cat([pre_labels, logits], dim=0)
            
    acc, mcc, sn, sp = evaluation_criterion(pre_labels.to('cpu').numpy(), real_labels.to('cpu').numpy())
    loss = test_loss / total_iter
    
    return acc, mcc, sn, sp, loss

def prediction(model, dataloader, kmer=3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    total_iter = len(dataloader)*1.0
    real_labels, pre_labels = [], []
    
    for batch_data in dataloader:
        model.eval()
        with torch.no_grad():
            
            input_ids = batch_data["input_ids"].to(device)
            attention_mask = batch_data["attention_mask"].to(device)
            labels = batch_data["labels"].to(device)
            
            outputs = model(input_ids, attention_mask, labels, kmer)
        
        loss = outputs[0]
        logits = outputs[1]
        
        if len(real_labels) == 0:
            real_labels = labels
            pre_labels = logits
        else:
            real_labels = torch.cat([real_labels, labels], dim=0)
            pre_labels = torch.cat([pre_labels, logits], dim=0)
    
    return pre_labels.to('cpu').numpy(), real_labels.to('cpu').numpy()


def output_logit_label_atten(model, dataloader, args):
    
    """
    function: 
        return logits, real_label, attention
        
        on the one hand, return the logits of prediction of dataloader; 
        in the other hand, return the attention weight of each nucleotide in the original DNA sequence
        
        key parameter:
            model: 
                Class, the Classification based on DNABERT, and its output contains the attention of each k-mer of the sentence here
                
            dataloader: 
                Dataloader, contains the dataset to be predicted.
                
            args:
                here, we just need to use one parameter: mer.
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mer = args.mer
    
    real_labels, pre_labels = [], []
    attention_scores = np.zeros([len(dataloader)*args.batch_size, 12, args.max_length-mer+1+2, args.max_length-mer+1+2])

    for index, batch_data in enumerate(dataloader):
        model.eval()
        with torch.no_grad():
            
            input_ids = batch_data["input_ids"].to(device)
            attention_mask = batch_data["attention_mask"].to(device)
            labels = batch_data["labels"].to(device)
            
            outputs = model(input_ids, attention_mask, labels)
        
        loss = outputs[0]
        logits = outputs[1]
        attention_score = outputs[-1][-1]
        
        attention_scores[index*args.batch_size : index*args.batch_size + len(batch_data["input_ids"]), :, :, :] = attention_score.cpu().numpy()
    
    atten_scores = attention_compute(attention_scores, args)
    
    return atten_scores

    
def attention_compute(attention_scores, args):
    """
    Function:
        calculate the vector of attention for each sequence. 
        And the entry of each vector stands for the attention of one nucleotide of its original DNA sequence.
        
        attention_scores: [length_of_sequences, 12, max_length_sequence, max_length_sequence]
            is the result of attention directly from the bert
    """
    
    kmer = args.mer
    scores = np.zeros([attention_scores.shape[0], attention_scores.shape[-1]-1-2+args.mer])
    
    for index, attention_score in enumerate(attention_scores):
        atten_score = []
        
        for i in range(1, attention_score.shape[-1]-1):
            atten_score.append(float(attention_score[:, 0, i].sum()))

        for i in range(len(atten_score)-1):
            if atten_score[i+1] == 0:
                atten_score[i] = 0
                break

        counts = np.zeros([len(atten_score) + kmer-1])
        real_scores = np.zeros([len(atten_score) + kmer -1])
        
        for i, score in enumerate(atten_score):
            for j in range(kmer):
                counts[i+j] += 1.0
                real_scores[i+j] += score
        real_scores = real_scores / counts
        real_scores = real_scores / np.linalg.norm(real_scores)
        
        scores[index] = real_scores

    return scores


def output_logit_label_atten_head(model, dataloader, args):
    """
    function: 
        return logits, real_label, attention_head
        
        on the one hand, return the logits of prediction of dataloader; 
        in the other hand, return the attention weight of each nucleotide in the original DNA sequence
        
        key parameter:
            model: 
                Class, the Classification based on DNABERT, and its output contains the attention of each k-mer of the sentence here
                
            dataloader: 
                Dataloader, contains the dataset to be predicted.
                
            args:
                here, we just need to use one parameter: mer.
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mer = args.mer
    
    attention_scores = np.zeros([len(dataloader)*args.batch_size, 12, args.max_length-mer+1+2, args.max_length-mer+1+2])

    for index, batch_data in enumerate(dataloader):
        model.eval()
        with torch.no_grad():
            
            input_ids = batch_data["input_ids"].to(device)
            attention_mask = batch_data["attention_mask"].to(device)
            labels = batch_data["labels"].to(device)
            
            outputs = model(input_ids, attention_mask, labels)
   
        loss = outputs[0]
        logits = outputs[1]
        attention_score = outputs[-1][-1]   # [batch_size, 12, 200, 200]


        attention_scores[index*args.batch_size : index*args.batch_size + len(batch_data["input_ids"]), :, :, :] = attention_score.cpu().numpy() 
        
        # attention_scores: [sequence_number, 12, 200, 200]
    atten_scores_head = attention_compute_head(attention_scores, args)
    
    return  atten_scores_head
    

def attention_compute_head(attention_scores, args):
    """
    Function:
        calculate the vector of attention for each sequence. 
        And the entry of each vector stands for the attention of one nucleotide of its original DNA sequence.
        
        attention_scores: [length_of_sequences, 12, max_length_sequence, max_length_sequence]
            is the result of attention directly from the bert
    """
    head_scores = []   # [12, sequences_number, 200]
    for i in range(12):
        head_scores.append([])

    kmer = args.mer
    scores = np.zeros([attention_scores.shape[0], attention_scores.shape[-1]-1-2+args.mer])

    for _, attention_score in enumerate(attention_scores):
        # attention_score: [12, 200, 200]
        for index, score_head in enumerate(attention_score):
            atten_score = score_head[0][1:200-kmer+2]

            counts = np.zeros([len(atten_score) + kmer-1])
            real_score = np.zeros([len(atten_score) + kmer -1])

            for i, score in enumerate(atten_score):
                for j in range(kmer):
                    counts[i+j] += 1.0
                    real_score[i+j] += score

            real_score = real_score / counts
            head_scores[index].append(real_score)
        
    return head_scores


# fine-tuning，
def getDataLoader_finetuning(args, shuffle_=True):
    model_path =  args.model_path
    batch_size_ = args.batch_size
    train_filename = args.tra_filename
    test_filename  = args.ind_filename
  
    tra_encoding, tra_labels, ind_encoding, ind_labels = getEmbedding_finetuning(train_filename, test_filename, model_path)
    
    tra_input_ids = torch.tensor(tra_encoding["input_ids"])
    tra_atte_mask = torch.tensor(tra_encoding["attention_mask"])
    
    ind_input_ids = torch.tensor(ind_encoding["input_ids"])
    ind_atte_mask = torch.tensor(ind_encoding["attention_mask"])
    
    tra_dataset = NewDataset_finetuning(tra_input_ids, tra_atte_mask, tra_labels)
    ind_dataset = NewDataset_finetuning(ind_input_ids, ind_atte_mask, ind_labels)

    tra_dataloader = DataLoader(tra_dataset, batch_size = batch_size_, shuffle = shuffle_)
    ind_dataloader = DataLoader(ind_dataset, batch_size = batch_size_, shuffle = False)
    
    return tra_dataloader, ind_dataloader
    
    
def getEmbedding_finetuning(train_data_filename, independent_data_filename, model_path):
    
    train_data = pd.read_csv(train_data_filename, sep="\t", header=None,names=["text","label"])
    train_text, tra_label = list(train_data["text"]), list(train_data["label"])
    
    
    independent_data = pd.read_csv(independent_data_filename, sep="\t", header=None,names=["text","label"])
    independent_text, ind_label = list(independent_data["text"]), list(independent_data["label"])
    
    tokenizer = BertTokenizer.from_pretrained(model_path) 
    tra_encoding = tokenizer(train_text, truncation = True, padding = True, max_length = 200)
    ind_encoding = tokenizer(independent_text, truncation = True, padding = True, max_length = 200)
    
    return tra_encoding, tra_label, ind_encoding, ind_label
