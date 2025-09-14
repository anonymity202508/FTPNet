import time
import numpy as np
import math
import sys
import datetime
from transformers import BertTokenizer
from tqdm import tqdm
import json
from transformers import BertModel
from transformers import BertConfig
import torch
import random
import os
import argparse
from bertopic import BERTopic
from utils.common import load_variable


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='WU3D')  # WU3D, eRisk
    parser.add_argument('--embedding', type=str, default='BERT')  # BERT, BERTopic
    parser.add_argument("--cuda", type=str, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    return args

def create(opt):
    current_directory = os.getcwd()
    print("Current directory：", current_directory)

    np.random.seed(opt.seed)
    random.seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.manual_seed(opt.seed)

    dataset = opt.dataset_name
    device = 'cuda:' + str(opt.cuda) if torch.cuda.is_available() else 'cpu'

    # data_path
    depression_data_path = 'data/' + dataset + '/depressed.json'
    normal_data_path = 'data/' + dataset + '/normal.json'
    data_paths = [depression_data_path, normal_data_path]

    if opt.embedding == 'BERT':
        # load bert
        if dataset == 'WU3D':
            pretrain_path = 'pretrain/chinese_mental_bert'
            # pretrain_path = 'pretrain/bert-base-chinese'
            tokenizer = BertTokenizer.from_pretrained(pretrain_path)
            bert_config = BertConfig.from_pretrained(pretrain_path)
            bert_config.max_position_embeddings = 1024
            bert = BertModel(config=bert_config).to(device)
        elif dataset == 'eRisk':
            pretrain_path = 'pretrain/mental_bert_base_uncased'
            # pretrain_path = 'pretrain/bert-base-chinese'
            tokenizer = BertTokenizer.from_pretrained(pretrain_path)
            bert_config = BertConfig.from_pretrained(pretrain_path)
            bert_config.max_position_embeddings = 1024
            bert = BertModel(config=bert_config).to(device)

        for param in bert.parameters():
            param.requires_grad = False
        bert.eval()
    elif opt.embedding == 'BERTopic':
        topic_model = load_variable('middle/' + dataset + '_topic_model.pkl')

    for data_path in data_paths:
        users = []
        with open(data_path, "r", encoding='utf8') as f:
            reader = json.load(f)
            for user in tqdm(reader):
                posts = []
                for post in user['tweets']:
                    if post['tweet_is_original'] == 'True' and len(post['tweet_content']) >= 2:
                        # isinstance(post['tweet_content'], 'NoneType')
                        if opt.embedding == 'BERT':
                            token = tokenizer(post['tweet_content'], add_special_tokens=True, padding='max_length',
                                              truncation=True, max_length=1024)
                            input_id = torch.LongTensor(token['input_ids']).unsqueeze(0).to(device)
                            token_type_id = torch.LongTensor(token['token_type_ids']).unsqueeze(0).to(device)
                            attention_mask = torch.LongTensor(token['attention_mask']).unsqueeze(0).to(device)
                            post_embedding = bert(input_id, token_type_id, attention_mask)[1]
                        elif opt.embedding == 'BERTopic':
                            post_embedding, _ = topic_model.approximate_distribution(post['tweet_content'], window=10, stride=2)
                            post_embedding = torch.tensor(post_embedding)
                        posts.append({'post': post['tweet_content'], 'post_embedding': post_embedding, 'time':
                            datetime.datetime.strptime(post['posting_time'][0:16], '%Y-%m-%d %H:%M')})
                if len(posts) > 10:
                    users.append(posts)
        if 'normal' in data_path:
            if opt.embedding == 'BERT':
                torch.save(users, 'data/' + dataset + '/exp_normal.pkl')
            elif opt.embedding == 'BERTopic':
                torch.save(users, 'data/' + dataset + '/exp_topic_normal.pkl')
        elif 'depressed' in data_path:
            if opt.embedding == 'BERT':
                torch.save(users, 'data/' + dataset + '/exp_depressed.pkl')
            elif opt.embedding == 'BERTopic':
                torch.save(users, 'data/' + dataset + '/exp_topic_depressed.pkl')

if __name__ == '__main__':
    opt = get_args()

    create(opt)
