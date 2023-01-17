import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset
from transformers import BertPreTrainedModel,BertModel

class NewsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    # 读取单个样本
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if isinstance(self.labels[idx], str):
            item['labels'] = self.labels[idx]
        else:
            item['labels'] = torch.tensor(self.labels[idx])
        return item
    
    def __len__(self):
        return len(self.labels)

    
class  NewDataset_finetuning(Dataset):
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels
        
    def __getitem__(self, idx):
        item = dict()
        item["input_ids"] = self.input_ids[idx]
        item["attention_mask"] = self.attention_mask[idx]
        item["labels"] = self.labels[idx]
        return item
    
    def __len__(self):
        return len(self.labels)


class  NewDataset_classifier(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        
    def __getitem__(self, idx):
        item = dict()
        item["features"] = self.features[idx]
        item["labels"] = self.labels[idx]
        return item
    
    def __len__(self):
        return len(self.labels)



class C_Bert_average_embedding(BertPreTrainedModel):
    def __init__(self, config):
        super(C_Bert_average_embedding, self).__init__(config)
        
        self.bert = BertModel(config)
        
    def forward(self, input_ids, attention_mask, numbers):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        
        embeddings = outputs[0] #(1+N, 768) CLS + each words embedding; embedding: [4, 502, 768]
        
        # embedding  = outputs[1] #(1, 768) CLS
        
        # excluding the CLS and SEP embedding, and then calculate the average of all words embedding
        for index in range(numbers.shape[0]):     #
            if numbers[index] == 0:
                token_average = torch.Tensor([0]*768)
            else:      
                token_average = embeddings[index][1:numbers[index]+1].mean(dim=-2)
            
            # print(token_average_1.shape, token_average_2.shape, token_average_3.shape, token_average_4.shape)
            # print(type(token_average_1), type(token_average_2), type(token_average_3), type(token_average_4))

            if index == 0:
                embedding = token_average
            else:
                embedding = torch.cat((embedding, token_average), dim=0)

        return embedding

class Enhancer_classifier_128(nn.Module):
    def __init__(self):
        super(Enhancer_classifier_128, self).__init__()
        self.dropout = nn.Dropout(0.30)
        
        self.classifier1 = nn.Linear(3072, 128)
        self.actTanh1 = nn.Tanh()
        self.classifier2 = nn.Linear(128, 1)
        self.actSigm = nn.Sigmoid()
    
    def forward(self, x, labels):
        x = self.dropout(x)
        
        x = self.classifier1(x)
        x = self.actTanh1(x)

        x = self.classifier2(x)
        x = self.actSigm(x)

        pre = x.view(-1).to(torch.float32)
        tru = labels.view(-1).to(torch.float32)
        loss_fct = nn.BCELoss()
        loss = loss_fct(pre, tru)

        return loss, pre  


class Enhancer_classifier_BN_128(nn.Module):
    def __init__(self):
        super(Enhancer_classifier_BN_128, self).__init__()
        self.dropout = nn.Dropout(0.30)
        
        self.classifier1 = nn.Linear(3072, 128)
        self.BN = nn.BatchNorm1d(128)
        self.actTanh1 = nn.Tanh()

        self.classifier2 = nn.Linear(128, 1)
        self.actSigm = nn.Sigmoid()
    
    def forward(self, x, labels):
        x = self.dropout(x)
        
        x = self.classifier1(x)
        x = self.BN(x)
        x = self.actTanh1(x)

        x = self.classifier2(x)
        x = self.actSigm(x)

        pre = x.view(-1).to(torch.float32)
        tru = labels.view(-1).to(torch.float32)
        loss_fct = nn.BCELoss()
        loss = loss_fct(pre, tru)

        return loss, pre  


class C_Bert_2FC_average(BertPreTrainedModel):
    def __init__(self, config):
        super(C_Bert_2FC_average, self).__init__(config)
        
        self.bert = BertModel(config)
        
        self.dropout = nn.Dropout(0.30)
        
        self.classifier1 = nn.Linear(768, 25)
        self.actTanh1 = nn.Tanh()
        self.classifier2 = nn.Linear(25, 1)
        self.actSigm = nn.Sigmoid()
 
    def forward(self, input_ids, attention_mask, labels, kmer=3):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        
        embeddings = outputs[0] #(1+N, 768) CLS + each words embedding
        embedding  = outputs[1] #(1, 768) CLS
        
        # excluding the CLS and SEP embedding, and then calculate the average of all words embedding
        output = embeddings[:,1:202-kmer]
        output = output.mean(dim=-2)
        
        output = self.dropout(output)

        output = self.classifier1(output)
        output = self.actTanh1(output)
        
        output = self.classifier2(output)
        output = self.actSigm(output)
        
        pre = output.view(-1).to(torch.float32)
        tru = labels.view(-1).to(torch.float32)
        loss_fct = nn.BCELoss()
        loss = loss_fct(pre, tru)

        return loss, pre  

    