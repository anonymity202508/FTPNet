from transformers import BertTokenizer
from transformers import BertModel
from transformers import BertConfig
import torch


class Bert_encoding:
    def __init__(self, pretrain_path='pretrain/bert-base-chinese', max_words=128):
        super(Bert_encoding, self).__init__()

        self.max_words = max_words
        self.tokenizer = BertTokenizer.from_pretrained(pretrain_path)
        bert_config = BertConfig.from_pretrained(pretrain_path)
        bert_config.max_position_embeddings = 1024
        self.bert = BertModel(config=bert_config).cuda()

        for param in self.bert.parameters():
            param.requires_grad = False

        self.bert.eval()

    def get_embedding(self, post):
        token = self.tokenizer(post, add_special_tokens=True, padding='max_length',
                          truncation=True, max_length=self.max_words)
        input_id = torch.LongTensor(token['input_ids']).unsqueeze(0).cuda()
        token_type_id = torch.LongTensor(token['token_type_ids']).unsqueeze(0).cuda()
        attention_mask = torch.LongTensor(token['attention_mask']).unsqueeze(0).cuda()
        post_embedding = self.bert(input_id, token_type_id, attention_mask)[1][0]

        return post_embedding


