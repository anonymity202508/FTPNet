import argparse
import json
import datetime
import numpy as np
import random
from tqdm import tqdm
import math
import torch
import pickle


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_period", type=int, default=14)
    parser.add_argument("--decay", type=int, default=14)
    parser.add_argument("--alpha", type=float, default=1)
    args = parser.parse_args()
    return args


class BUClustering:
    def __init__(self, max_period, decay, alpha):
        self.max_period = max_period
        self.decay = decay
        self.alpha = alpha
        self.X = None

    def fit(self, inputs):
        n_samples = len(inputs)
        self.X = [list([inputs[i]]) for i in range(len(inputs))]
        max_timediff = 0
        for i in range(n_samples):
            distances = self._compute_distances()
            for j in range(int(len(self.X) * (len(self.X) - 1)/2)):
                nearest_indices = np.unravel_index(np.argmax(distances), distances.shape)
                max_timediff = self._compute_timediff(nearest_indices)
                if max_timediff < self.max_period:
                    self._merge_clusters(nearest_indices)
                    break
                else:
                    distances[nearest_indices] = 0
            if max_timediff > self.max_period:
                break

        return self.X

    def _compute_distances(self):
        n_samples = len(self.X)
        distances = np.zeros((n_samples, n_samples))

        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                ints_time_i = [(sing_x['posting_time'].year - 1949) * 365 + sing_x['posting_time'].month * 30 + sing_x[
                    'posting_time'].day for sing_x in self.X[i]]
                ints_time_j = [(sing_x['posting_time'].year - 1949) * 365 + sing_x['posting_time'].month * 30 + sing_x[
                    'posting_time'].day for sing_x in self.X[j]]
                emds_sem_i = [sing_x['post_embedding'] for sing_x in self.X[i]]
                emds_sem_j = [sing_x['post_embedding'] for sing_x in self.X[j]]

                max_sim = 0
                for m in range(len(ints_time_i)):
                    for n in range(len(ints_time_j)):
                        dist_time = abs(ints_time_i[m] - ints_time_j[n])
                        sim_time = math.exp(-dist_time/self.decay)
                        dist_sem = torch.sum((emds_sem_i[m] - emds_sem_j[n]) ** 2)
                        sim_sem = torch.exp(-torch.sqrt(dist_sem))
                        sim_ij = self.alpha * sim_time + (1 - self.alpha) * sim_sem
                        if sim_ij > max_sim:
                            max_sim = sim_ij
                distances[i, j] = max_sim

        return distances

    def _compute_timediff(self, indices):
        ints_time_i = [(sing_x['posting_time'].year - 1949) * 365 + sing_x['posting_time'].month * 30 + sing_x[
            'posting_time'].day for sing_x in self.X[indices[0]]]
        ints_time_j = [(sing_x['posting_time'].year - 1949) * 365 + sing_x['posting_time'].month * 30 + sing_x[
            'posting_time'].day for sing_x in self.X[indices[1]]]
        max_distance = 0
        for m in range(len(ints_time_i)):
            for n in range(len(ints_time_j)):
                dist_time = abs(ints_time_i[m] - ints_time_j[n])
                if dist_time > max_distance:
                    max_distance = dist_time

        return max_distance

    def _merge_clusters(self, indices):
        self.X[indices[0]] = self.X[indices[0]] + self.X[indices[1]]
        del self.X[indices[1]]


def create(opt):
    depression_pretrain = pickle.load(open('depression_pretrain', 'rb'))
    normal_pretrain = pickle.load(open('normal_pretrain', 'rb'))

    clustering = BUClustering(max_period=opt.max_period, decay=opt.decay, alpha=opt.alpha)

    for i in tqdm(range(len(depression_pretrain))):
        print(i)
        depression_pretrain[i] = clustering.fit(depression_pretrain[i])

    for i in tqdm(range(len(normal_pretrain))):
        normal_pretrain[i] = clustering.fit(normal_pretrain[i])

    pickle.dump(depression_pretrain, open("data/WU3D/" + "depression_clusters_a" + str(opt.alpha * 10), 'wb'))
    pickle.dump(normal_pretrain, open("data/WU3D/" + "normal_clusters_a" + str(opt.alpha * 10), 'wb'))


if __name__ == '__main__':
    opt = get_args()
    create(opt)

