
import os
import torch
import argparse
import numpy as np
import pandas as pd
from util import * 
from model import *
from pandas import DataFrame
import sklearn
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from transformers import BertTokenizer

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_embedding_for_scatter_token(args, dataloader, mer):

    all_token_embeddings_last_layers = []
    all_labels = []

    print("load model")
    model_after = C_Bert_2FC_average_embedding.from_pretrained(args.model_path, num_labels=1, output_hidden_states=True).to(device)
    model_dict = torch.load("model/C_Bert_2FC_average_{}-mer.pt".format(mer))
    model_after.load_state_dict(model_dict)
    
    with torch.no_grad():
        for batch_data in dataloader:
            input_ids = batch_data["input_ids"].to(device)
            attention_mask = batch_data["attention_mask"].to(device)
            labels = batch_data["labels"]

            outputs = model_after(input_ids, attention_mask, labels)

            token_embeddings_last_layers = outputs[4].to("cpu").tolist()
            all_token_embeddings_last_layers += token_embeddings_last_layers
            all_labels += batch_data["labels"]

    all_token_embeddings_last_layers = np.array(all_token_embeddings_last_layers)[:, 1]
    return all_token_embeddings_last_layers, all_labels


def get_attention_for_sequence_token(args, sequence_filename, mer):
   
    args.batch_size = 1
    token_attention_sequence = []
    token_keys = []

    df_raw = pd.read_csv(sequence_filename, sep="\t",header=None,names=["text","label"]) 
    dataloader = getDataloader(df_raw, args, training=True)

    model_after = C_Bert_2FC_average_atten.from_pretrained(args.model_path, num_labels=1, output_attentions=True).to(device)
    model_dict = torch.load("model/C_Bert_2FC_average_{}-mer.pt".format(mer))
    model_after.load_state_dict(model_dict)

    for index, dataitem in enumerate(dataloader):

        tokens = df_raw["text"][index].split(" ")
        token_keys.append(tokens)
    
        with torch.no_grad():
            input_ids = torch.tensor(dataitem["input_ids"]).to(device)
            attention_mask = torch.tensor(dataitem["attention_mask"]).to(device)

            outputs = model_after(input_ids, attention_mask, 1)

            attention_score = outputs[2][-1].tolist() # return the attention mechanism(12 head) in the last layer 
            attention_score = np.array(attention_score)
            if len(token_attention_sequence) == 0 :
                token_attention_sequence = attention_score 
            else:
                token_attention_sequence = np.concatenate((token_attention_sequence, attention_score), axis=0)
            print(token_attention_sequence.shape)

    token_keys = np.array(token_keys)
    return token_attention_sequence, token_keys


def tsne(token_embeddings):
    RS = 42
    token_embeddings_proj = TSNE(n_components=2, init='pca', random_state=RS).fit_transform(token_embeddings)
    return token_embeddings_proj


def plt_scatter(scatter_token_tsne, scatter_name):

    plt.figure(figsize=(6, 6))
    ax = plt.subplot(1,1,1)
    ax.scatter(scatter_token_tsne[0][0], scatter_token_tsne[0][1], marker="*", c="red", s=250)

    for i in range(1, len(scatter_token_tsne)):
        color = "deepskyblue"
        s=60
        if i == 1:
            color = "mediumblue"
            s=100
        ax.scatter(scatter_token_tsne[i][0], scatter_token_tsne[i][1], marker="o", c=color, s=s)

        if i != 1:
            ax.plot([scatter_token_tsne[i][0], scatter_token_tsne[i-1][0]], [scatter_token_tsne[i][1], scatter_token_tsne[i-1][1]], color="gray", linewidth=1)
            

    plt.xticks([])
    plt.yticks([])
    plt.title("{}".format(scatter_name))
    plt.savefig("image/{}.svg".format(scatter_name), bbox_inches='tight')
    plt.show()