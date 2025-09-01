import numpy as np
from sklearn import metrics
import pickle
import torch
import torch.nn.functional as F


def euclidean_distances(a, b):
    sq_a = a**2
    sum_sq_a = torch.sum(sq_a, dim=1).unsqueeze(1)
    sq_b = b**2
    sum_sq_b = torch.sum(sq_b, dim=1).unsqueeze(0)
    bt = b.t()
    return sum_sq_a + sum_sq_b - 2*a.mm(bt)


def get_evaluation(y_true, y_prob, list_metrics):
    y_prob = F.softmax(y_prob, -1)
    y_prob = y_prob.cpu().detach().numpy()
    y_pred = np.argmax(y_prob, -1)
    output = {}
    if 'accuracy' in list_metrics:
        output['accuracy'] = metrics.accuracy_score(y_true, y_pred)
    if 'recall' in list_metrics:
        output['recall'] = metrics.recall_score(y_true, y_pred)
    if 'precision' in list_metrics:
        output['precision'] = metrics.precision_score(y_true, y_pred)
    if 'f1' in list_metrics:
        output['f1'] = metrics.f1_score(y_true, y_pred)
    if 'auc' in list_metrics:
        y_prob = np.array(y_prob, dtype=float)
        fpr, tpr, threshold = metrics.roc_curve(y_true, y_prob.swapaxes(1, 0)[1])
        output['auc'] = metrics.auc(fpr, tpr)

    # if 'loss' in list_metrics:
    #     try:
    #         output['loss'] = metrics.log_loss(y_true, y_prob)
    #     except ValueError:
    #         output['loss'] = -1
    if 'confusion_matrix' in list_metrics:
        output['confusion_matrix'] = str(metrics.confusion_matrix(y_true, y_pred))
    return output


def save_variable(v, filename):
    f = open(filename, 'wb')
    pickle.dump(v, f)
    f.close()
    return filename


def load_variable(filename):
    f = open(filename, 'rb')
    r = pickle.load(f)
    f.close()
    return r

