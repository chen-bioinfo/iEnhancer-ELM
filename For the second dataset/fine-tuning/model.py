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


class NewsDataset_2(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels
    
    # 读取单个样本
    def __getitem__(self, idx):
        item = dict()
        item["sequences"] = self.sequences[idx]

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


class C_Bert_2FC_concatenate(BertPreTrainedModel):
    def __init__(self, config):
        super(C_Bert_2FC_concatenate, self).__init__(config)
        
        self.bert = BertModel(config)
        
        self.dropout = nn.Dropout(0.30)
        
        self.classifier1 = nn.Linear(3072, 128)
        self.actTanh1 = nn.Tanh()

        self.classifier2 = nn.Linear(128, 1)
        self.actSigm = nn.Sigmoid()
 
    def forward(self, input_ids, token_numbers, attention_mask, labels, kmer=3):
        
        output_1 = self.bert(input_ids[:,0], attention_mask=attention_mask[:,0])
        output_2 = self.bert(input_ids[:,1], attention_mask=attention_mask[:,1])
        output_3 = self.bert(input_ids[:,2], attention_mask=attention_mask[:,2])
        output_4 = self.bert(input_ids[:,3], attention_mask=attention_mask[:,3])

        # only CLS
        # output_1: 
        # output = torch.cat((output_1[1], output_2[1], output_3[1], output_4[1]), dim=1) # only CLS

        # average kmer-token embeddings
        outputs = []
        for index in range(input_ids.shape[0]):
            if token_numbers[index][0] == 0:
                token_average_1 = torch.Tensor([0]*768)
            else:      
                token_average_1 = output_1[0][index][1:token_numbers[index][0]+1].mean(dim=-2)
            
            if token_numbers[index][1] == 0:
                token_average_2 = torch.Tensor([0]*768)
            else:      
                token_average_2 = output_2[0][index][1:token_numbers[index][1]+1].mean(dim=-2)

            if token_numbers[index][2] == 0:
                token_average_3 = torch.Tensor([0]*768)
            else:      
                token_average_3 = output_3[0][index][1:token_numbers[index][2]+1].mean(dim=-2)

            if token_numbers[index][3] == 0:
                token_average_4 = torch.Tensor([0]*768)
            else:      
                token_average_4 = output_4[0][index][1:token_numbers[index][3]+1].mean(dim=-2)

            # print(token_average_1.shape, token_average_2.shape, token_average_3.shape, token_average_4.shape)
            # print(type(token_average_1), type(token_average_2), type(token_average_3), type(token_average_4))

            if index == 0:
                outputs = torch.cat((token_average_1, token_average_2, token_average_3, token_average_4), dim=0)
            else:
                average_embedding = torch.cat((token_average_1, token_average_2, token_average_3, token_average_4), dim=0)
                outputs = torch.cat((outputs, average_embedding), dim=0)
        
        outputs = torch.reshape(outputs, (input_ids.shape[0], 3072))

        output = self.dropout(outputs)

        output = self.classifier1(output)
        output = self.actTanh1(output)
        
        output = self.classifier2(output)
        output = self.actSigm(output)
        
        pre = output.view(-1).to(torch.float32)
        tru = labels.view(-1).to(torch.float32)
        loss_fct = nn.BCELoss()
        loss = loss_fct(pre, tru)
        
        return loss, pre


class C_Bert_2FC_concatenate_BN_128(BertPreTrainedModel):
    def __init__(self, config):
        super(C_Bert_2FC_concatenate_BN_128, self).__init__(config)
        
        self.bert = BertModel(config)
        
        self.dropout = nn.Dropout(0.30)
        
        self.classifier1 = nn.Linear(3072, 128)
        self.BN = nn.BatchNorm1d(num_features=128)
        self.actTanh1 = nn.Tanh()

        self.classifier2 = nn.Linear(128, 1)
        self.actSigm = nn.Sigmoid()
 
    def forward(self, input_ids, token_numbers, attention_mask, labels, kmer=3):
        
        output_1 = self.bert(input_ids[:,0], attention_mask=attention_mask[:,0])    # output_1[0]: batch_size * sequence_length * 768
        output_2 = self.bert(input_ids[:,1], attention_mask=attention_mask[:,1])
        output_3 = self.bert(input_ids[:,2], attention_mask=attention_mask[:,2])
        output_4 = self.bert(input_ids[:,3], attention_mask=attention_mask[:,3])

        # only CLS
        # output_1: 
        # output = torch.cat((output_1[1], output_2[1], output_3[1], output_4[1]), dim=1) # only CLS
        
        # average kmer-token embeddings
        outputs = []
        for index in range(input_ids.shape[0]):     #
            if token_numbers[index][0] == 0:
                token_average_1 = torch.Tensor([0]*768)
            else:      
                token_average_1 = output_1[0][index][1:token_numbers[index][0]+1].mean(dim=-2)
            
            if token_numbers[index][1] == 0:
                token_average_2 = torch.Tensor([0]*768)
            else:      
                token_average_2 = output_2[0][index][1:token_numbers[index][1]+1].mean(dim=-2)

            if token_numbers[index][2] == 0:
                token_average_3 = torch.Tensor([0]*768)
            else:      
                token_average_3 = output_3[0][index][1:token_numbers[index][2]+1].mean(dim=-2)

            if token_numbers[index][3] == 0:
                token_average_4 = torch.Tensor([0]*768)
            else:      
                token_average_4 = output_4[0][index][1:token_numbers[index][3]+1].mean(dim=-2)

            # print(token_average_1.shape, token_average_2.shape, token_average_3.shape, token_average_4.shape)
            # print(type(token_average_1), type(token_average_2), type(token_average_3), type(token_average_4))

            if index == 0:
                outputs = torch.cat((token_average_1, token_average_2, token_average_3, token_average_4), dim=0)
            else:
                average_embedding = torch.cat((token_average_1, token_average_2, token_average_3, token_average_4), dim=0)
                outputs = torch.cat((outputs, average_embedding), dim=0)
        
        outputs = torch.reshape(outputs, (input_ids.shape[0], 3072))

        output = self.dropout(outputs)

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


class C_Bert_2FC_concatenate_BN_256(BertPreTrainedModel):
    def __init__(self, config):
        super(C_Bert_2FC_concatenate_BN_256, self).__init__(config)
        
        self.bert = BertModel(config)
        
        self.dropout = nn.Dropout(0.30)
        
        self.classifier1 = nn.Linear(3072, 256)
        self.BN = nn.BatchNorm1d(num_features=256)
        self.actTanh1 = nn.Tanh()

        self.classifier2 = nn.Linear(256, 1)
        self.actSigm = nn.Sigmoid()
 
    def forward(self, input_ids, token_numbers, attention_mask, labels, kmer=3):
        
        output_1 = self.bert(input_ids[:,0], attention_mask=attention_mask[:,0])
        output_2 = self.bert(input_ids[:,1], attention_mask=attention_mask[:,1])
        output_3 = self.bert(input_ids[:,2], attention_mask=attention_mask[:,2])
        output_4 = self.bert(input_ids[:,3], attention_mask=attention_mask[:,3])

        # only CLS
        # output_1: 
        # output = torch.cat((output_1[1], output_2[1], output_3[1], output_4[1]), dim=1) # only CLS

        # average kmer-token embeddings
        outputs = []
        for index in range(input_ids.shape[0]):
            if token_numbers[index][0] == 0:
                token_average_1 = torch.Tensor([0]*768)
            else:      
                token_average_1 = output_1[0][index][1:token_numbers[index][0]+1].mean(dim=-2)
            
            if token_numbers[index][1] == 0:
                token_average_2 = torch.Tensor([0]*768)
            else:      
                token_average_2 = output_2[0][index][1:token_numbers[index][1]+1].mean(dim=-2)

            if token_numbers[index][2] == 0:
                token_average_3 = torch.Tensor([0]*768)
            else:      
                token_average_3 = output_3[0][index][1:token_numbers[index][2]+1].mean(dim=-2)

            if token_numbers[index][3] == 0:
                token_average_4 = torch.Tensor([0]*768)
            else:      
                token_average_4 = output_4[0][index][1:token_numbers[index][3]+1].mean(dim=-2)

            # print(token_average_1.shape, token_average_2.shape, token_average_3.shape, token_average_4.shape)
            # print(type(token_average_1), type(token_average_2), type(token_average_3), type(token_average_4))

            if index == 0:
                outputs = torch.cat((token_average_1, token_average_2, token_average_3, token_average_4), dim=0)
            else:
                average_embedding = torch.cat((token_average_1, token_average_2, token_average_3, token_average_4), dim=0)
                outputs = torch.cat((outputs, average_embedding), dim=0)
        
        outputs = torch.reshape(outputs, (input_ids.shape[0], 3072))

        output = self.dropout(outputs)

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



class C_Bert_2FC_concatenate_BN_512(BertPreTrainedModel):
    def __init__(self, config):
        super(C_Bert_2FC_concatenate_BN_512, self).__init__(config)
        
        self.bert = BertModel(config)
        
        self.dropout = nn.Dropout(0.30)
        
        self.classifier1 = nn.Linear(3072, 512)
        self.BN = nn.BatchNorm1d(num_features=512)
        self.actTanh1 = nn.Tanh()

        self.classifier2 = nn.Linear(512, 1)
        self.actSigm = nn.Sigmoid()
 
    def forward(self, input_ids, token_numbers, attention_mask, labels, kmer=3):
        
        output_1 = self.bert(input_ids[:,0], attention_mask=attention_mask[:,0])
        output_2 = self.bert(input_ids[:,1], attention_mask=attention_mask[:,1])
        output_3 = self.bert(input_ids[:,2], attention_mask=attention_mask[:,2])
        output_4 = self.bert(input_ids[:,3], attention_mask=attention_mask[:,3])

        # only CLS
        # output_1: 
        # output = torch.cat((output_1[1], output_2[1], output_3[1], output_4[1]), dim=1) # only CLS

        # average kmer-token embeddings
        outputs = []
        for index in range(input_ids.shape[0]):
            if token_numbers[index][0] == 0:
                token_average_1 = torch.Tensor([0]*768)
            else:      
                token_average_1 = output_1[0][index][1:token_numbers[index][0]+1].mean(dim=-2)
            
            if token_numbers[index][1] == 0:
                token_average_2 = torch.Tensor([0]*768)
            else:      
                token_average_2 = output_2[0][index][1:token_numbers[index][1]+1].mean(dim=-2)

            if token_numbers[index][2] == 0:
                token_average_3 = torch.Tensor([0]*768)
            else:      
                token_average_3 = output_3[0][index][1:token_numbers[index][2]+1].mean(dim=-2)

            if token_numbers[index][3] == 0:
                token_average_4 = torch.Tensor([0]*768)
            else:      
                token_average_4 = output_4[0][index][1:token_numbers[index][3]+1].mean(dim=-2)

            # print(token_average_1.shape, token_average_2.shape, token_average_3.shape, token_average_4.shape)
            # print(type(token_average_1), type(token_average_2), type(token_average_3), type(token_average_4))

            if index == 0:
                outputs = torch.cat((token_average_1, token_average_2, token_average_3, token_average_4), dim=0)
            else:
                average_embedding = torch.cat((token_average_1, token_average_2, token_average_3, token_average_4), dim=0)
                outputs = torch.cat((outputs, average_embedding), dim=0)
        
        outputs = torch.reshape(outputs, (input_ids.shape[0], 3072))

        output = self.dropout(outputs)

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

class C_Bert_2FC_average_lager(BertPreTrainedModel):
    def __init__(self, config):
        super(C_Bert_2FC_average_lager, self).__init__(config)
        
        self.bert = BertModel(config)
        
        self.dropout = nn.Dropout(0.30)
        
        self.classifier1 = nn.Linear(768, 512)
        self.BN1 = nn.BatchNorm1d(num_features=512)
        self.actTanh1 = nn.Tanh()

        self.classifier2 = nn.Linear(512, 128)
        self.BN2 = nn.BatchNorm1d(num_features=128)
        self.actTanh2 = nn.Tanh()
        
        self.classifier3 = nn.Linear(128, 1)
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
        output = self.BN1(output)
        output = self.actTanh1(output)

        output = self.classifier2(output)
        output = self.BN2(output)
        output = self.actTanh2(output)

        output = self.classifier3(output)
        output = self.actSigm(output)
        
        pre = output.view(-1).to(torch.float32)
        tru = labels.view(-1).to(torch.float32)
        loss_fct = nn.BCELoss()
        loss = loss_fct(pre, tru)

        return loss, pre  


class FGM():
    def __init__(self, model, emb_name1='bert.embeddings', emb_name2='bert.embeddings'):
        self.model = model
        self.backup = {}
        self.emb_name1 = emb_name1
        self.emb_name2 = emb_name2

    def attack(self, epsilon=1.0):
        # emb_name这个参数要换成你模型中embedding的参数名
        # 例如，self.emb = nn.Embedding(5000, 100)
        for name, param in self.model.named_parameters():
            if param.requires_grad and (self.emb_name1 in name or self.emb_name2 in name):
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)  # 默认为2范数
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, ):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and (self.emb_name1 in name or self.emb_name2 in name):
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

    
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
    