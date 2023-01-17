import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset
from transformers import BertPreTrainedModel,BertModel

class NewsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    # read simple sample
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


class C_Bert_2FC_atten(BertPreTrainedModel):
    def __init__(self, config):
        super(C_Bert_2FC_atten, self).__init__(config)
        
        self.bert = BertModel(config)
        
        self.dropout = nn.Dropout(0.30)
        
        self.classifier1 = nn.Linear(768, 25)
        self.actTanh1 = nn.Tanh()
        self.classifier2 = nn.Linear(25, 1)
        self.actSigm = nn.Sigmoid()
 
    def forward(self, input_ids, attention_mask, labels):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        
        embeddings = outputs[0] #(2+N, 768) CLS + each words embedding SEP
        embedding  = outputs[1] #(1, 768) CLS
        
        output = self.dropout(embedding)

        output = self.classifier1(output)
        output = self.actTanh1(output)
        
        output = self.classifier2(output)
        output = self.actSigm(output)
        
        pre = output.view(-1).to(torch.float32)
        
        tru = labels.view(-1).to(torch.float32)
    
        loss_fct = nn.BCELoss()
        loss = loss_fct(pre, tru)
        
        if len(outputs) > 2:
            return loss, pre, outputs[2] 
        else:
            return loss, pre  

class C_Bert_2FC_average_atten(BertPreTrainedModel):
    def __init__(self, config):
        super(C_Bert_2FC_average_atten, self).__init__(config)
        
        self.bert = BertModel(config)
        
        self.dropout = nn.Dropout(0.30)
        
        self.classifier1 = nn.Linear(768, 25)
        self.actTanh1 = nn.Tanh()
        self.classifier2 = nn.Linear(25, 1)
        self.actSigm = nn.Sigmoid()
 
    def forward(self, input_ids, attention_mask, labels):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        
        embeddings = outputs[0] #(2+N, 768) CLS + each words embedding SEP
        embedding  = outputs[1] #(1, 768) CLS
        
        # excluding the CLS and SEP embedding, and then calculate the average of all words embedding
        output = embeddings[:,1:-1]
        output = output.mean(dim=-2)
        
        if len(outputs) > 2:
            return 1, 1, outputs[2]
        else:
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

    
    