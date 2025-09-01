import torch
import torch.nn as nn
from transformers import BertModel
import torch.nn.functional as F
from utils.common import euclidean_distances


class FTPNet(nn.Module):
    def __init__(self, bert_config, num_depression_labels=2, num_symptom_labels=2,
                 num_prototype=10, filter_sizes=[2, 3, 4], d_min=0.5,
                 lambda_p=0.5, lambda_c=0.1, lambda_e=0.1, lambda_d=0.1, lambda_s=0.01):
        super(FTPNet, self).__init__()
        self.bert = BertModel(config=bert_config)
        self.num_prototype = num_prototype
        self.prototype = nn.Parameter(torch.randn(num_prototype, bert_config.hidden_size))
        self.d_min = d_min

        self.filter_sizes = filter_sizes

        self.fc_depression = nn.Linear(num_prototype * len(filter_sizes), num_depression_labels)
        self.fc_symptom = nn.Linear(num_prototype, num_symptom_labels)

        self.lambda_p = lambda_p
        self.lambda_c = lambda_c
        self.lambda_e = lambda_e
        self.lambda_d = lambda_d
        self.lambda_s = lambda_s
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, user_posts, post_times, symptom_posts, user_labels=None, symptom_labels=None):
        """
        Args:
            user_posts: (batch_size, max_periods, max_posts, seq_len) - user posts grouped by periods
            post_times: (batch_size, max_periods, max_posts) - timestamps for temporal segmentation
            symptom_posts: (batch_size_p, seq_len) - posts for auxiliary symptom classification
            user_labels: (batch_size,) - depression labels (for training)
            symptom_labels: (batch_size_p,) - symptom labels (for training)
        Returns:
            depression_pred: (batch_size, num_depression_labels)
            symptom_pred: (batch_size_p, num_symptom_labels)
            total_loss: computed loss (if labels provided)
        """
        batch_size, max_periods, max_posts, seq_len = user_posts.shape

        user_posts_flat = user_posts.reshape(-1, seq_len)
        user_emb_flat = self.bert(input_ids=user_posts_flat).last_hidden_state[:, 0, :]
        user_emb = user_emb_flat.reshape(batch_size, max_periods, max_posts, -1)

        symptom_emb = self.bert(input_ids=symptom_posts).last_hidden_state[:, 0, :]

        proto_expanded = self.prototype.unsqueeze(1).unsqueeze(1).unsqueeze(1)
        user_emb_expanded = user_emb.unsqueeze(0)
        dist_post_proto = torch.sum((user_emb_expanded - proto_expanded) ** 2, dim=-1)
        s_kml = torch.exp(-dist_post_proto)
        s_km = torch.amax(s_kml, dim=3)

        proto_expanded_p = self.prototype.unsqueeze(0)
        symptom_emb_expanded = symptom_emb.unsqueeze(1)
        dist_symptom_proto = torch.sum((symptom_emb_expanded - proto_expanded_p) ** 2, dim=-1)
        s_kp = torch.exp(-dist_symptom_proto)

        batch_score_proto = []
        for filter_size in self.filter_sizes:
            s_km_reshaped = s_km.reshape(-1, 1, max_periods)

            conv_out = F.conv1d(
                s_km_reshaped,
                torch.ones(1, 1, filter_size, device=s_km.device) / filter_size,
                padding=0
            )

            g_kj = torch.amax(conv_out, dim=2)
            g_kj = g_kj.reshape(self.num_prototype, batch_size)

            batch_score_proto.append(g_kj)

        batch_score_proto = torch.cat(batch_score_proto, dim=0)
        temporal_features = batch_score_proto.permute(1, 0).float()

        depression_pred = self.fc_depression(temporal_features)  # (batch, 2)
        symptom_pred = self.fc_symptom(s_kp)  # (batch_p, 2)

        total_loss = None
        if user_labels is not None and symptom_labels is not None:
            ce_u = self.criterion(depression_pred, user_labels)
            ce_p = self.criterion(symptom_pred, symptom_labels)

            proto_dist = euclidean_distances(self.prototype, self.prototype)
            r_d = torch.sum(torch.max(torch.zeros_like(proto_dist), self.d_min - proto_dist) ** 2)

            min_dist_post = torch.amin(dist_post_proto, dim=0)
            r_c = torch.sum(min_dist_post)

            min_dist_proto = torch.amin(dist_post_proto, dim=[1, 2, 3])
            r_e = torch.sum(min_dist_proto)

            r_s = torch.norm(self.fc_depression.weight, 1) + torch.norm(self.fc_symptom.weight, 1)

            total_loss = ce_u + self.lambda_p * ce_p + \
                         self.lambda_c * r_c + self.lambda_e * r_e + \
                         self.lambda_d * r_d + self.lambda_s * r_s

        return depression_pred, symptom_pred, total_loss
