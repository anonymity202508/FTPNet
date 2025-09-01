import numpy as np
import torch
from transformers import BertConfig
from models.FTPNet import FTPNet
import torch.nn as nn
from torch.utils.data import DataLoader
import json
import gc
import random
import time
import pandas as pd

np.random.seed(1234)
random.seed(1234)
torch.cuda.manual_seed(1234)
torch.manual_seed(1234)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# print(device)

train_dataset = torch.load("data/WU3D/IR4/c10p5t7_userWord_train_dataset.data")
print(train_dataset.__len__())
# valid_dataset = torch.load("data/WU3D/c20p10t7segib15k_valid_dataset.data")

Symptom_train_dataset = torch.load("data/" + "Symptom_train_dataset.data")

# print(device)

hidden_size = 128
num_layers = 2
dropout = 0.3
num_prototype = 30
num_labels = 2
filter_sizes = [1, 2, 3, 10]
dMin = 1
pretrain_path = 'pretrain/bert-base-chinese'
bert_config = BertConfig.from_pretrained(pretrain_path)

# model = TCRnnPSeNet(bert_config, hidden_size, num_layers, dropout, num_prototype, num_labels, filter_size)
model = FTPNet(bert_config, num_labels, num_prototype, filter_sizes, dMin, Symptom_train_dataset, max_clusters=10, max_posts=5)
model = model.to(device)

dict_state = torch.load("results/model_FTPNet.pkl")
model.load_state_dict(dict_state)

print("Hello")

prototype_embed = model.prototype

train_x_path = "data/WU3D/IR4/train_x.json"
train_y_path = "data/WU3D/IR4/train_y.json"
max_clusters = 10
max_posts = 5
# train_dataset = HieraDataset(train_x_path, train_y_path, pretrain_path, max_clusters, max_posts)

with open(train_x_path, "r", encoding='utf8') as f:
    train_x = json.load(f)

print(len(train_x))

train_x_new = []
for user in train_x:
    train_user_new = []
    for cluster in user:
        if len(cluster) < max_posts:
            extended_posts = ['' for _ in range(max_posts - len(cluster))]
            cluster.extend(extended_posts)
        cluster = cluster[len(cluster) - max_posts:len(cluster)]
        train_user_new.append(cluster)
    if len(user) < max_clusters:
        extended_clusters = [['' for _ in range(max_posts)] for _ in range(max_clusters - len(user))]
        train_user_new.extend(extended_clusters)
    train_user_new = train_user_new[len(train_user_new) - max_clusters:len(train_user_new)]
    train_x_new.append(train_user_new)


post_exist_tensor = torch.stack([train_dataset.__getitem__(i)[2] for i in range(len(train_dataset))])
post_encode_tensor = torch.stack([train_dataset.__getitem__(i)[0] for i in range(len(train_dataset))])
post_exist_tensor = post_exist_tensor.to(device)
prototype_embed = prototype_embed.to(device)
proto_posts = []
proto_min = []

print(len(post_encode_tensor))

column_list = ['no-depression 1', 'no-depression 2', 'no-depression 3', 'no-depression 5', 'no-depression 8',
               'no-depression 12', 'no-depression 16', 'no-depression 20',
               'depression 1', 'depression 2', 'depression 3', 'depression 5', 'depression 8', 'depression 12',
               'depression 16', 'depression 20',
               'prototype nearest post']
proto_results = pd.DataFrame(columns=column_list)

prototype_embed = prototype_embed.to(device)
post_encode_tensor = torch.reshape(post_encode_tensor, (len(post_exist_tensor)*10*5, 768))
for i in range(0, num_prototype):   # range(len(prototype_embed)):

    proto_dist1 = torch.sum((torch.unsqueeze(prototype_embed[i], 0) - post_encode_tensor) ** 2, 1)
    proto_dist1 = torch.reshape(proto_dist1, (len(post_exist_tensor), 10, 5))
    proto_dist1[post_exist_tensor == 0] = 500
    min1 = torch.min(proto_dist1)

    all_min = min1
    index = torch.argmin(all_min)


    user_index = int(index / (10 * 5))
    cluster_index = int((index % (10 * 5)) / 5)
    post_index = int((index % (10 * 5)) % 5)
    proto_post = train_x_new[user_index][cluster_index][post_index]

    print(i, proto_post)

    time.sleep(2)


