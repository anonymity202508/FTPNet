import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader
from models.FTPNet import FTPNet
from utils.common import get_evaluation
from dataset import HieraDataset
import datetime
import torch.nn as nn
from transformers import BertConfig
import warnings
import os
import pandas as pd
import random


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument('--dataset_name', type=str, default='WU3D')
    parser.add_argument("--num_epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--es_min_delta", type=float, default=0.0)
    parser.add_argument("--es_patience", type=int, default=10)
    parser.add_argument("--imbalance_ratio", type=int, default=8)
    parser.add_argument("--num_prototype", type=int, default=60)
    parser.add_argument("--filter_sizes", type=list, default=[2, 3, 4])
    parser.add_argument("--d_min", type=float, default=0.5)
    parser.add_argument("--lambda_p", type=float, default=0.5)
    parser.add_argument("--lambda_c", type=float, default=0.1)
    parser.add_argument("--lambda_e", type=float, default=0.1)
    parser.add_argument("--lambda_d", type=float, default=0.1)
    parser.add_argument("--lambda_s", type=float, default=0.01)
    parser.add_argument("--max_clusters", type=int, default=20)
    parser.add_argument("--max_posts", type=int, default=10)
    parser.add_argument("--TC_threshold", type=int, default=13)
    parser.add_argument("--pretrain_path", type=str, default='pretrain/bert-base-chinese')
    parser.add_argument("--dataLoad_type", type=str, default='cluster')
    parser.add_argument("--manual_seed", type=int, default=1234)
    parser.add_argument("--test_interval", type=int, default=1)
    args = parser.parse_args()
    return args


def train(opt):
    print("FTPNet parameters: {}".format(vars(opt)))
    warnings.filterwarnings("ignore")

    if opt.manual_seed is not None:
        torch.cuda.manual_seed(opt.manual_seed)
        torch.cuda.manual_seed_all(opt.manual_seed)
        torch.manual_seed(opt.manual_seed)
        np.random.seed(opt.manual_seed)
        random.seed(opt.manual_seed)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    train_dataset = torch.load(
        f"data/{opt.dataset_name}/IR{opt.imbalance_ratio}/c{opt.max_clusters}p{opt.max_posts}t{opt.TC_threshold}_train_dataset.data")
    valid_dataset = torch.load(
        f"data/{opt.dataset_name}/IR{opt.imbalance_ratio}/c{opt.max_clusters}p{opt.max_posts}t{opt.TC_threshold}_valid_dataset.data")
    test_dataset = torch.load(
        f"data/{opt.dataset_name}/IR{opt.imbalance_ratio}/c{opt.max_clusters}p{opt.max_posts}t{opt.TC_threshold}_test_dataset.data")

    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, drop_last=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=opt.batch_size, shuffle=False, drop_last=False)
    test_dataloader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, drop_last=False)

    bert_config = BertConfig.from_pretrained(opt.pretrain_path)
    num_depression_labels = len(set(train_dataset.depression_labels))
    num_symptom_labels = len(set(train_dataset.symptom_labels))

    model = FTPNet(
        bert_config,
        num_depression_labels=num_depression_labels,
        num_symptom_labels=num_symptom_labels,
        num_prototype=opt.num_prototype,
        filter_sizes=opt.filter_sizes,
        d_min=opt.d_min,
        lambda_p=opt.lambda_p,
        lambda_c=opt.lambda_c,
        lambda_e=opt.lambda_e,
        lambda_d=opt.lambda_d,
        lambda_s=opt.lambda_s
    )
    model = nn.DataParallel(model)
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr)

    best_f1 = 0
    best_epoch = 0
    best_test_metrics = {
        "precision": 0, "recall": 0, "accuracy": 0, "f1": 0, "auc": 0
    }

    column_list = ['model', 'mode', 'epoch', 'loss', 'precision', 'recall', 'accuracy', 'f1', 'auc', 'dateTime']
    log = pd.DataFrame(columns=column_list)

    for epoch in range(opt.num_epochs):
        model.train()
        for iter, (user_posts, post_times, symptom_posts, user_labels, symptom_labels) in enumerate(train_dataloader):
            optimizer.zero_grad()
            user_posts = user_posts.to(device)
            post_times = post_times.to(device)
            symptom_posts = symptom_posts.to(device)
            user_labels = user_labels.to(device)
            symptom_labels = symptom_labels.to(device)

            depression_pred, symptom_pred, total_loss = model(
                user_posts=user_posts,
                post_times=post_times,
                symptom_posts=symptom_posts,
                user_labels=user_labels,
                symptom_labels=symptom_labels
            )

            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=15, norm_type=2)
            optimizer.step()

        if epoch % opt.test_interval == 0:
            model.eval()
            tr_metrics, tr_loss = evaluate(model, train_dataloader, device)
            print(f"Train Epoch: {epoch}/{opt.num_epochs}, Loss: {float(tr_loss.cpu()):.4f}, "
                  f"precision: {tr_metrics['precision']:.4f}, recall: {tr_metrics['recall']:.4f}, "
                  f"Accuracy: {tr_metrics['accuracy']:.4f}, f1: {tr_metrics['f1']:.4f}, "
                  f"auc: {tr_metrics['auc']:.4f}, Time: {datetime.datetime.now().strftime('%X')}")
            log = update_log(log, 'train', epoch, tr_loss, tr_metrics)

            val_metrics, val_loss = evaluate(model, valid_dataloader, device)
            print(f"Valid Epoch: {epoch}/{opt.num_epochs}, Loss: {float(val_loss.cpu()):.4f}, "
                  f"precision: {val_metrics['precision']:.4f}, recall: {val_metrics['recall']:.4f}, "
                  f"Accuracy: {val_metrics['accuracy']:.4f}, f1: {val_metrics['f1']:.4f}, "
                  f"auc: {val_metrics['auc']:.4f}, Time: {datetime.datetime.now().strftime('%X')}")
            log = update_log(log, 'valid', epoch, val_loss, val_metrics)

            test_metrics, te_loss = evaluate(model, test_dataloader, device)
            print(f"Test Epoch: {epoch}/{opt.num_epochs}, Loss: {float(te_loss.cpu()):.4f}, "
                  f"precision: {test_metrics['precision']:.4f}, recall: {test_metrics['recall']:.4f}, "
                  f"Accuracy: {test_metrics['accuracy']:.4f}, f1: {test_metrics['f1']:.4f}, "
                  f"auc: {test_metrics['auc']:.4f}, Time: {datetime.datetime.now().strftime('%X')}")
            log = update_log(log, 'test', epoch, te_loss, test_metrics)

            if val_metrics["f1"] > best_f1:
                best_f1 = val_metrics["f1"]
                best_epoch = epoch
                best_test_metrics = test_metrics
                save_model(model, opt)

            if epoch - best_epoch > opt.es_patience > 0:
                print(f"Stop training at epoch {best_epoch}.")
                print(
                    f"Best Test Metrics - precision: {best_test_metrics['precision']:.4f}, recall: {best_test_metrics['recall']:.4f}, "
                    f"Accuracy: {best_test_metrics['accuracy']:.4f}, f1: {best_test_metrics['f1']:.4f}, auc: {best_test_metrics['auc']:.4f}")
                log = update_best_log(log, best_test_metrics)
                save_log(log, opt)
                break

    print(f"Final training stopped at epoch {best_epoch}.")
    print(
        f"Best Test Metrics - precision: {best_test_metrics['precision']:.4f}, recall: {best_test_metrics['recall']:.4f}, "
        f"Accuracy: {best_test_metrics['accuracy']:.4f}, f1: {best_test_metrics['f1']:.4f}, auc: {best_test_metrics['auc']:.4f}")
    log = update_best_log(log, best_test_metrics)
    save_log(log, opt)


def evaluate(model, dataloader, device):
    num_samples = 0
    loss_ls = []
    label_ls = []
    pred_ls = []

    for user_posts, post_times, symptom_posts, user_labels, symptom_labels in dataloader:
        num_sample = len(user_labels)
        num_samples += num_sample
        user_posts = user_posts.to(device)
        post_times = post_times.to(device)
        symptom_posts = symptom_posts.to(device)
        user_labels = user_labels.to(device)
        symptom_labels = symptom_labels.to(device)

        with torch.no_grad():
            depression_pred, symptom_pred, total_loss = model(
                user_posts=user_posts,
                post_times=post_times,
                symptom_posts=symptom_posts,
                user_labels=user_labels,
                symptom_labels=symptom_labels
            )

        loss_ls.append(total_loss * num_sample)
        label_ls.extend(user_labels.clone().cpu())
        pred_ls.append(depression_pred.clone().cpu())

    total_loss = sum(loss_ls) / num_samples
    pred = torch.cat(pred_ls, 0)
    label = np.array(label_ls)
    metrics = get_evaluation(label, pred, list_metrics=["precision", "recall", "accuracy", "f1", "auc"])
    return metrics, total_loss


def update_log(log, mode, epoch, loss, metrics):
    new_row = pd.DataFrame([[
        'FTPNet', mode, epoch, float(loss.cpu()),
        round(metrics['precision'], 4), round(metrics['recall'], 4),
        round(metrics['accuracy'], 4), round(metrics['f1'], 4),
        round(metrics['auc'], 4), datetime.datetime.now().strftime("%X")
    ]], columns=log.columns)
    return pd.concat([log, new_row], ignore_index=True)


def update_best_log(log, best_metrics):
    new_row = pd.DataFrame([[
        'FTPNet', 'Best', "-", "-",
        round(best_metrics['precision'], 4), round(best_metrics['recall'], 4),
        round(best_metrics['accuracy'], 4), round(best_metrics['f1'], 4),
        round(best_metrics['auc'], 4), datetime.datetime.now().strftime("%X")
    ]], columns=log.columns)
    return pd.concat([log, new_row], ignore_index=True)


def save_model(model, opt):
    save_path = f"results/{opt.manual_seed}/model_FTPNet_ir{opt.imbalance_ratio}p{opt.num_prototype}c{opt.max_clusters}p{opt.max_posts}t{opt.TC_threshold}.pkl"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)


def save_log(log, opt):
    log_path = f"results/{opt.manual_seed}/log_FTPNet_ir{opt.imbalance_ratio}p{opt.num_prototype}c{opt.max_clusters}p{opt.max_posts}t{opt.TC_threshold}.csv"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log.to_csv(log_path, index=False)


if __name__ == "__main__":
    opt = get_args()
    train(opt)
