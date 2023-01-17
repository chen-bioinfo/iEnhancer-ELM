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



class C_Bert_2FC(BertPreTrainedModel):
    def __init__(self, config):
        super(C_Bert_2FC, self).__init__(config)
        
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

        return loss, pre  

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
    
    
class C_Bert_2FC_Noise(BertPreTrainedModel):
    def __init__(self, config):
        super(C_Bert_2FC_Noise, self).__init__(config)
        
        self.bert = BertModel(config)
        
        self.dropout = nn.Dropout(0.30)
        
        self.classifier1 = nn.Linear(768, 25)
        self.actTanh1 = nn.Tanh()
        self.classifier2 = nn.Linear(25, 1)
        self.actSigm = nn.Sigmoid()
 
    def forward(self, input_ids, attention_mask, labels):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        
        embeddings = outputs[0] #(1+N, 768) CLS + each words embedding
        embedding  = outputs[1] #(1, 768) CLS
        
        # training, then add noise into the embedding
        # else, No
        if self.training:
            x1, x2 = embedding.shape

            noise = torch.zeros(x1, x2, dtype=torch.float32).to("cuda")
            noise = noise + (0.1**0.5)*torch.randn(x1, x2).to("cuda")
            
            embedding = embedding + noise
        
        output = self.dropout(embedding)

        output = self.classifier1(output)
        output = self.actTanh1(output)
        
        output = self.classifier2(output)
        output = self.actSigm(output)
        
        pre = output.view(-1).to(torch.float32)
        tru = labels.view(-1).to(torch.float32)
        loss_fct = nn.BCELoss()
        loss = loss_fct(pre, tru)

        return loss, pre  
    
    
class C_Bert_2FC_BN(BertPreTrainedModel):
    def __init__(self, config):
        super(C_Bert_2FC_BN, self).__init__(config)
        
        self.bert = BertModel(config)
        
        self.dropout = nn.Dropout(0.30)
        
        self.classifier1 = nn.Linear(768, 25)
        self.BN = nn.BatchNorm1d(25)
        self.actTanh1 = nn.Tanh()
        self.classifier2 = nn.Linear(25, 1)
        self.actSigm = nn.Sigmoid()
 
    def forward(self, input_ids, attention_mask, labels):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        
        embeddings = outputs[0] #(1+N, 768) CLS + each words embedding
        embedding  = outputs[1] #(1, 768) CLS
        
        output = self.dropout(embedding)

        output = self.classifier1(output)
        output = self.BN(output)
        output = self.actTanh1(output)
        
        output = self.classifier2(output)
        output = self.actSigm(output)
        
        pre = output.view(-1).to(torch.float32)
        tru = labels.view(-1).to(torch.float32)
        loss_fct = nn.BCELoss()
        loss = loss_fct(pre, tru)

        return loss, pre      
    

class C_Bert_average_embedding(BertPreTrainedModel):
    def __init__(self, config):
        super(C_Bert_average_embedding, self).__init__(config)
        
        self.bert = BertModel(config)
        
    def forward(self, input_ids, attention_mask, kmer=3):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        
        embeddings = outputs[0] #(1+N, 768) CLS + each words embedding
        # embedding  = outputs[1] #(1, 768) CLS
        
        # excluding the CLS and SEP embedding, and then calculate the average of all words embedding
        output = embeddings[:,1:202-kmer]
        output = output.mean(dim=-2)

        return output

class Enhancer_classifier(nn.Module):
    def __init__(self):
        super(Enhancer_classifier, self).__init__()
        self.dropout = nn.Dropout(0.30)
        
        self.classifier1 = nn.Linear(768, 25)
        self.actTanh1 = nn.Tanh()
        self.classifier2 = nn.Linear(25, 1)
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

    
class C_Bert_2FC_average_embedding(BertPreTrainedModel):
    def __init__(self, config):
        super(C_Bert_2FC_average_embedding, self).__init__(config)
        
        self.bert = BertModel(config)
        
        self.dropout = nn.Dropout(0.30)
        
        self.classifier1 = nn.Linear(768, 25)
        self.actTanh1 = nn.Tanh()
        self.classifier2 = nn.Linear(25, 1)
        self.actSigm = nn.Sigmoid()
 
    def forward(self, input_ids, attention_mask, labels, kmer=3):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        
        embeddings = outputs[0] #(batch_size, 200, 768) CLS + each words embedding + SEP + PAD;  last hidden state
        embedding  = outputs[1] #(1, 768) CLS
        embedding_all_layers = outputs[2] # all hidden states( zero layer + 12 layers )
      
        # excluding the CLS and SEP embedding, and then calculate the average of all words embedding
        output = embeddings[:,1:-1]
        output = output.mean(dim=-2)

        embedding = output
        embeddings_all = []
        for i in range(len(embedding_all_layers)):
            embeddings_all.append(embedding_all_layers[i][:, 1:-1])
            embeddings_all[i] = embeddings_all[i].mean(dim=-2)
        
        # embedding: the embedding of CLS
        # enbeddings_all: the mean embedding of all solid words in 12 layers
        # embeddings: the embedding of all tokens in the last layer
        return 1, 1, embedding, embeddings_all, embeddings
    
    
class C_Bert_2FC_average_no_Sigmoid(BertPreTrainedModel):
    def __init__(self, config):
        super(C_Bert_2FC_average_no_Sigmoid, self).__init__(config)
        
        self.bert = BertModel(config)
        
        self.dropout = nn.Dropout(0.30)
        
        self.classifier1 = nn.Linear(768, 25)
        self.actTanh1 = nn.Tanh()
        self.classifier2 = nn.Linear(25, 1)
        self.actSigm = nn.Sigmoid()
 
    def forward(self, input_ids, attention_mask, labels):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        
        embeddings = outputs[0] #(1+N, 768) CLS + each words embedding
        embedding  = outputs[1] #(1, 768) CLS
        
        # excluding the CLS and SEP embedding, and then calculate the average of all words embedding
        output = embeddings[:,1:-1]
        output = output.mean(dim=-2)
        
        output = self.dropout(output)

        output = self.classifier1(output)
        output = self.actTanh1(output)
        
        output = self.classifier2(output)
        output1 = self.actSigm(output)
        
        pre = output.view(-1).to(torch.float32)
        tru = labels.view(-1).to(torch.float32)
        loss_fct = nn.BCELoss()
        loss = loss_fct(pre, tru)

        return loss, pre  
    
    
class C_Bert_2FC_average_2(BertPreTrainedModel):
    def __init__(self, config):
        super(C_Bert_2FC_average_2, self).__init__(config)
        self.config = config
        self.bert = BertModel(config)
        
        self.dropout = nn.Dropout(0.30)
        
        self.classifier1 = nn.Linear(768, 25)
        self.actTanh1 = nn.Tanh()
        self.classifier2 = nn.Linear(25, 1)
        self.actSigm = nn.Sigmoid()
 
    def forward(self, input_ids, attention_mask, labels):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        print(len(outputs))
        embeddings = outputs[0] #(1+N, 768) CLS + each words embedding
        embedding  = outputs[1] #(1, 768) CLS
        
        
        # excluding the CLS and SEP embedding, and then calculate the average of all words embedding
        output = embeddings[:,1:-1]
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
        
        if len(outputs) > 2:
            return loss, pre, outputs[2]
        else:
            return loss, pre  

class C_Bert_2_Conv_2FC_3_4(BertPreTrainedModel):
    def __init__(self, config):
        super(C_Bert_2_Conv_2FC_3_4, self).__init__(config)
        
        self.bert = BertModel(config)
        
        self.conv1d_1 = nn.Conv1d(in_channels=768, out_channels=16, kernel_size=3, padding=0)
        self.activePReLU_1 = nn.PReLU()

        self.conv1d_2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=0)
        self.activePReLU_2 = nn.PReLU()

        self.maxpooling = nn.MaxPool1d(kernel_size=3, stride=2)
        
        self.flatten = nn.Flatten()
        
        self.dropout = nn.Dropout(0.40)
        
        self.classifier1 = nn.Linear(3072, 256)
        self.actFC1 = nn.Tanh()
        self.classifier2 = nn.Linear(256, 1)
        self.actSigm = nn.Sigmoid()
 

    def forward(self, input_ids, attention_mask, labels):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        
        embeddings = outputs[0]
        embedding  = outputs[1]
        
        output = embeddings[:, 1:-1]

        output = output.permute(0, 2, 1) # [batch_size, 199, 768]-->[batch_size, 768, 199]
        
        output = self.conv1d_1(output)
        output = self.activePReLU_1(output)
        output = self.conv1d_2(output)
        output = self.activePReLU_2(output)
  
        output = self.maxpooling(output)
        
        output = self.flatten(output)
        output = self.dropout(output)
        
        output = self.classifier1(output)
        output = self.actFC1(output)
        
        output = self.classifier2(output)
        output = self.actSigm(output)
        
        pre = output.view(-1).to(torch.float32)
        tru = labels.view(-1).to(torch.float32)
        loss_fct = nn.BCELoss()
        loss = loss_fct(pre, tru)

        return loss, pre
    
    
class C_Bert_2_Conv_2FC_5_6(BertPreTrainedModel):
    def __init__(self, config):
        super(C_Bert_2_Conv_2FC_5_6, self).__init__(config)
        
        self.bert = BertModel(config)
        
        self.conv1d_1 = nn.Conv1d(in_channels=768, out_channels=16, kernel_size=3, padding=0)
        self.activePReLU_1 = nn.PReLU()

        self.conv1d_2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=0)
        self.activePReLU_2 = nn.PReLU()

        self.maxpooling = nn.MaxPool1d(kernel_size=3, stride=2)
        
        self.flatten = nn.Flatten()
        
        self.dropout = nn.Dropout(0.50)
        
        self.classifier1 = nn.Linear(3040, 256)
        self.actFC1 = nn.Tanh()
        self.classifier2 = nn.Linear(256, 1)
        self.actSigm = nn.Sigmoid()
 

    def forward(self, input_ids, attention_mask, labels):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        
        embeddings = outputs[0]
        embedding  = outputs[1]
        
        output = embeddings[:, 1:-1]

        output = output.permute(0, 2, 1) # [batch_size, 199, 768]-->[batch_size, 768, 199]
        
        output = self.conv1d_1(output)
        output = self.activePReLU_1(output)
        output = self.conv1d_2(output)
        output = self.activePReLU_2(output)
  
        output = self.maxpooling(output)
        
        output = self.flatten(output)
        output = self.dropout(output)
        
        output = self.classifier1(output)
        output = self.actFC1(output)
        
        output = self.classifier2(output)
        output = self.actSigm(output)
        
        pre = output.view(-1).to(torch.float32)
        tru = labels.view(-1).to(torch.float32)
        loss_fct = nn.BCELoss()
        loss = loss_fct(pre, tru)

        return loss, pre
    