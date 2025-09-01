from torch.utils.data import Dataset
from transformers import BertTokenizer
from tqdm import tqdm
import json
import numpy as np
from transformers import BertModel
from transformers import BertConfig
import torch
import random


class HieraWordDataset(Dataset):
    def __init__(self, x_path, y_path, pretrain_path='pretrain/bert-base-chinese', max_clusters=20, max_posts=10):
        super(HieraWordDataset, self).__init__()
        with open(x_path, "r", encoding='utf8') as f:
            x = json.load(f)
        with open(y_path, "r", encoding='utf8') as f:
            self.y = json.load(f)

        np.random.seed(1234)
        random.seed(1234)
        torch.cuda.manual_seed(1234)
        torch.manual_seed(1234)

        tokenizer = BertTokenizer.from_pretrained(pretrain_path)
        input_ids = []
        token_type_ids = []
        attention_masks = []
        bert_config = BertConfig.from_pretrained(pretrain_path)
        bert = BertModel(config=bert_config).to("cuda:0")
        for param in bert.parameters():
            param.requires_grad = False

        # 加载数据
        print('loading data from:', x_path, ',', y_path)

        for user in tqdm(x):
            user_input = []
            user_token = []
            user_mask = []
            for cluster in user:
                if len(cluster) < max_posts:
                    extended_posts = ['' for _ in range(max_posts - len(cluster))]
                    cluster.extend(extended_posts)
                cluster = cluster[len(cluster)-max_posts:len(cluster)]
                token = tokenizer(cluster, add_special_tokens=True, padding='max_length', truncation=True,
                                       max_length=128)
                user_input.append(token['input_ids'])
                user_token.append(token['token_type_ids'])
                user_mask.append(token['attention_mask'])
            if len(user) < max_clusters:
                extended_clusters = [['' for _ in range(max_posts)] for _ in range(max_clusters - len(user))]
                for cluster in extended_clusters:
                    token = tokenizer(cluster, add_special_tokens=True, padding='max_length', truncation=True,
                                           max_length=128)
                    user_input.append(token['input_ids'])
                    user_token.append(token['token_type_ids'])
                    user_mask.append(token['attention_mask'])
            user_input = user_input[len(user_input)-max_clusters:len(user_input)]
            user_token = user_token[len(user_input)-max_clusters:len(user_input)]
            user_mask = user_mask[len(user_input)-max_clusters:len(user_input)]
            input_ids.append(user_input)
            token_type_ids.append(user_token)
            attention_masks.append(user_mask)

        input_ids = torch.LongTensor(input_ids).to("cuda:0")
        token_type_ids = torch.LongTensor(token_type_ids).to("cuda:0")
        attention_masks = torch.LongTensor(attention_masks).to("cuda:0")

        self.post_exist = (torch.sum(attention_masks, 3) != 2).to(torch.int32)

        self.input_ids = torch.reshape(
            input_ids, (input_ids.shape[0], input_ids.shape[1]*input_ids.shape[2], input_ids.shape[3]))
        self.token_type_ids = torch.reshape(
            token_type_ids, (token_type_ids.shape[0], token_type_ids.shape[1]*token_type_ids.shape[2], token_type_ids.shape[3]))
        self.attention_masks = torch.reshape(
            attention_masks, (attention_masks.shape[0], attention_masks.shape[1]*attention_masks.shape[2], attention_masks.shape[3]))

    def __getitem__(self, index):
        return self.input_ids[index], self.token_type_ids[index], self.attention_masks[index], self.y[index], self.post_exist[index]

    def __len__(self):
        return len(self.y)
