import copy
import os
import numpy as np
import pandas as pd
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
from data_provider.uea import (
    subsample,
    interpolate_missing,
    Normalizer,
    normalize_batch_ts,
    bandpass_filter_func,
)
import warnings
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


class APAVALoader(Dataset):
    def __init__(self, root_path, flag=None):
        self.root_path = root_path
        self.data_path = os.path.join(root_path, "Feature/")
        self.label_path = os.path.join(root_path, "Label/label.npy")

        data_list = np.load(self.label_path)

        all_ids = list(data_list[:, 1])
        val_ids = [15, 16, 19, 20]
        test_ids = [1, 2, 17, 18]
        train_ids = [int(i) for i in all_ids if i not in val_ids + test_ids]

        self.train_ids, self.val_ids, self.test_ids = train_ids, val_ids, test_ids

        self.X, self.y = self.load_apava(self.data_path, self.label_path, flag=flag)

        self.X = normalize_batch_ts(self.X)

        self.max_seq_len = self.X.shape[1]

    def load_apava(self, data_path, label_path, flag=None):
        feature_list = []
        label_list = []
        filenames = []

        subject_label = np.load(label_path)
        for filename in os.listdir(data_path):
            filenames.append(filename)
        filenames.sort()

        if flag == "TRAIN":
            ids = self.train_ids
            print("train ids:", ids)
        elif flag == "VAL":
            ids = self.val_ids
            print("val ids:", ids)
        elif flag == "TEST":
            ids = self.test_ids
            print("test ids:", ids)
        else:
            ids = subject_label[:, 1]
            print("all ids:", ids)

        for j in range(len(filenames)):
            trial_label = subject_label[j]
            path = data_path + filenames[j]
            subject_feature = np.load(path)
            for trial_feature in subject_feature:
                if j + 1 in ids:
                    feature_list.append(trial_feature)
                    label_list.append(trial_label)

        X = np.array(feature_list)
        y = np.array(label_list)
        X, y = shuffle(X, y, random_state=42)

        return X, y[:, 0]

    def __getitem__(self, index):
        return torch.from_numpy(self.X[index]), torch.from_numpy(
            np.asarray(self.y[index])
        )

    def __len__(self):
        return len(self.y)


class TDBRAINLoader(Dataset):
    def __init__(self, root_path, flag=None):
        self.root_path = root_path
        self.data_path = os.path.join(root_path, "Feature/")
        self.label_path = os.path.join(root_path, "Label/label.npy")

        train_ids = list(range(1, 18)) + list(range(29, 46))
        val_ids = [18, 19, 20, 21] + [46, 47, 48, 49]
        test_ids = [22, 23, 24, 25] + [50, 51, 52, 53]

        self.train_ids, self.val_ids, self.test_ids = train_ids, val_ids, test_ids

        self.X, self.y = self.load_tdbrain(self.data_path, self.label_path, flag=flag)
        self.X = normalize_batch_ts(self.X)
        self.max_seq_len = self.X.shape[1]

    def load_tdbrain(self, data_path, label_path, flag=None):
        feature_list = []
        label_list = []
        filenames = []
        subject_label = np.load(label_path)
        for filename in os.listdir(data_path):
            filenames.append(filename)
        filenames.sort()
        if flag == "TRAIN":
            ids = self.train_ids
            print("train ids:", ids)
        elif flag == "VAL":
            ids = self.val_ids
            print("val ids:", ids)
        elif flag == "TEST":
            ids = self.test_ids
            print("test ids:", ids)
        else:
            ids = subject_label[:, 1]
            print("all ids:", ids)

        for j in range(len(filenames)):
            trial_label = subject_label[j]
            path = data_path + filenames[j]
            subject_feature = np.load(path)
            for trial_feature in subject_feature:
                if j + 1 in ids:
                    feature_list.append(trial_feature)
                    label_list.append(trial_label)
        X = np.array(feature_list)
        y = np.array(label_list)
        X, y = shuffle(X, y, random_state=42)

        return X, y[:, 0]

    def __getitem__(self, index):
        return torch.from_numpy(self.X[index]), torch.from_numpy(
            np.asarray(self.y[index])
        )

    def __len__(self):
        return len(self.y)


class ADFDLoader(Dataset):
    def __init__(self, root_path, flag=None):
        self.root_path = root_path
        self.data_path = os.path.join(root_path, "Feature/")
        self.label_path = os.path.join(root_path, "Label/label.npy")

        a, b = 0.6, 0.8

        self.train_ids, self.val_ids, self.test_ids = self.load_train_val_test_list(
            self.label_path, a, b
        )
        self.X, self.y = self.load_adfd(self.data_path, self.label_path, flag=flag)

        self.X = normalize_batch_ts(self.X)
        self.max_seq_len = self.X.shape[1]

    def load_train_val_test_list(self, label_path, a=0.6, b=0.8):
        data_list = np.load(label_path)
        cn_list = list(data_list[np.where(data_list[:, 0] == 0)][:, 1])
        ftd_list = list(data_list[np.where(data_list[:, 0] == 1)][:, 1])
        ad_list = list(data_list[np.where(data_list[:, 0] == 2)][:, 1])

        train_ids = (
            cn_list[: int(a * len(cn_list))]
            + ftd_list[: int(a * len(ftd_list))]
            + ad_list[: int(a * len(ad_list))]
        )
        val_ids = (
            cn_list[int(a * len(cn_list)) : int(b * len(cn_list))]
            + ftd_list[int(a * len(ftd_list)) : int(b * len(ftd_list))]
            + ad_list[int(a * len(ad_list)) : int(b * len(ad_list))]
        )
        test_ids = (
            cn_list[int(b * len(cn_list)) :]
            + ftd_list[int(b * len(ftd_list)) :]
            + ad_list[int(b * len(ad_list)) :]
        )

        return train_ids, val_ids, test_ids

    def load_adfd(self, data_path, label_path, flag=None):
        feature_list = []
        label_list = []
        filenames = []
        subject_label = np.load(label_path)
        for filename in os.listdir(data_path):
            filenames.append(filename)
        filenames.sort()
        if flag == "TRAIN":
            ids = self.train_ids
            print("train ids:", ids)
        elif flag == "VAL":
            ids = self.val_ids
            print("val ids:", ids)
        elif flag == "TEST":
            ids = self.test_ids
            print("test ids:", ids)
        else:
            ids = subject_label[:, 1]
            print("all ids:", ids)

        for j in range(len(filenames)):
            trial_label = subject_label[j]
            path = data_path + filenames[j]
            subject_feature = np.load(path)
            for trial_feature in subject_feature:
                if j + 1 in ids:
                    feature_list.append(trial_feature)
                    label_list.append(trial_label)
        X = np.array(feature_list)
        y = np.array(label_list)
        X, y = shuffle(X, y, random_state=42)

        return X, y[:, 0]

    def __getitem__(self, index):
        return torch.from_numpy(self.X[index]), torch.from_numpy(
            np.asarray(self.y[index])
        )

    def __len__(self):
        return len(self.y)


class ADFDDependentLoader(Dataset):
    def __init__(self, root_path, flag=None):
        self.root_path = root_path
        self.data_path = os.path.join(root_path, 'Feature/')
        self.label_path = os.path.join(root_path, 'Label/label.npy')

        self.X, self.y = self.load_adfd_dependent(self.data_path, self.label_path, flag=flag)

        self.X = normalize_batch_ts(self.X)
        self.max_seq_len = self.X.shape[1]

    def load_adfd_dependent(self, data_path, label_path, flag=None):
        feature_list = []
        label_list = []
        filenames = []

        subject_label = np.load(label_path)
        for filename in os.listdir(data_path):
            filenames.append(filename)
        filenames.sort()

        for j in range(len(filenames)):
            trial_label = subject_label[j]
            path = data_path + filenames[j]
            subject_feature = np.load(path)
            for trial_feature in subject_feature:
                feature_list.append(trial_feature)
                label_list.append(trial_label)

        X_train, y_train = np.array(feature_list), np.array(label_list)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

        if flag == 'TRAIN':
            return X_train, y_train[:, 0]
        elif flag == 'VAL':
            return X_val, y_val[:, 0]
        elif flag == 'TEST':
            return X_test, y_test[:, 0]
        else:
            raise Exception('flag must be TRAIN, VAL, or TEST')

    def __getitem__(self, index):
        return torch.from_numpy(self.X[index]), \
               torch.from_numpy(np.asarray(self.y[index]))

    def __len__(self):
        return len(self.y)


class PTBLoader(Dataset):
    def __init__(self, root_path, flag=None):
        self.root_path = root_path
        self.data_path = os.path.join(root_path, "Feature/")
        self.label_path = os.path.join(root_path, "Label/label.npy")

        a, b = 0.55, 0.7

        self.train_ids, self.val_ids, self.test_ids = self.load_train_val_test_list(
            self.label_path, a, b
        )

        self.X, self.y = self.load_ptb(self.data_path, self.label_path, flag=flag)

        self.X = normalize_batch_ts(self.X)

        self.max_seq_len = self.X.shape[1]

    def load_train_val_test_list(self, label_path, a=0.6, b=0.8):
        data_list = np.load(label_path)
        hc_list = list(data_list[np.where(data_list[:, 0] == 0)][:, 1])
        my_list = list(data_list[np.where(data_list[:, 0] == 1)][:, 1])

        train_ids = hc_list[: int(a * len(hc_list))] + my_list[: int(a * len(my_list))]
        val_ids = (
            hc_list[int(a * len(hc_list)) : int(b * len(hc_list))]
            + my_list[int(a * len(my_list)) : int(b * len(my_list))]
        )
        test_ids = hc_list[int(b * len(hc_list)) :] + my_list[int(b * len(my_list)) :]

        return train_ids, val_ids, test_ids

    def load_ptb(self, data_path, label_path, flag=None):
        feature_list = []
        label_list = []
        filenames = []
        subject_label = np.load(label_path)
        for filename in os.listdir(data_path):
            filenames.append(filename)
        filenames.sort()
        if flag == "TRAIN":
            ids = self.train_ids
            print("train ids:", ids)
        elif flag == "VAL":
            ids = self.val_ids
            print("val ids:", ids)
        elif flag == "TEST":
            ids = self.test_ids
            print("test ids:", ids)
        else:
            ids = subject_label[:, 1]
            print("all ids:", ids)

        for j in range(len(filenames)):
            trial_label = subject_label[j]
            path = data_path + filenames[j]
            subject_feature = np.load(path)
            for trial_feature in subject_feature:
                if j + 1 in ids:
                    feature_list.append(trial_feature)
                    label_list.append(trial_label)
        X = np.array(feature_list)
        y = np.array(label_list)
        X, y = shuffle(X, y, random_state=42)

        return X, y[:, 0]

    def __getitem__(self, index):
        return torch.from_numpy(self.X[index]), torch.from_numpy(
            np.asarray(self.y[index])
        )

    def __len__(self):
        return len(self.y)


class PTBXLLoader(Dataset):
    _cache = {}

    def __init__(self, root_path, flag=None):
        self.root_path = root_path
        self.data_path = os.path.join(root_path, "Feature/")
        self.label_path = os.path.join(root_path, "Label/label.npy")

        a, b = 0.6, 0.8

        self.train_ids, self.val_ids, self.test_ids = self.load_train_val_test_list(
            self.label_path, a, b
        )

        cache_key = os.path.abspath(root_path)
        cache_entry = PTBXLLoader._cache.get(cache_key)
        if cache_entry is None:
            all_x, all_y, sample_subject_ids = self.load_ptbxl_all(
                self.data_path, self.label_path
            )
            all_x = normalize_batch_ts(all_x)
            cache_entry = {
                "X": all_x,
                "y": all_y,
                "sample_subject_ids": sample_subject_ids,
                "max_seq_len": all_x.shape[1],
            }
            PTBXLLoader._cache[cache_key] = cache_entry

        self.X, self.y = self.select_split(cache_entry, flag)
        self.max_seq_len = cache_entry["max_seq_len"]

    def load_train_val_test_list(self, label_path, a=0.6, b=0.8):
        data_list = np.load(label_path)
        super_labels = list(set(data_list[:, 0]))

        train_ids = []
        val_ids = []
        test_ids = []

        for label in super_labels:
            label_ids = list(data_list[np.where(data_list[:, 0] == label)][:, 1])
            train_ids += label_ids[: int(a * len(label_ids))]
            val_ids += label_ids[int(a * len(label_ids)) : int(b * len(label_ids))]
            test_ids += label_ids[int(b * len(label_ids)) :]

        return train_ids, val_ids, test_ids

    def load_ptbxl_all(self, data_path, label_path):
        feature_list = []
        label_list = []
        sample_subject_ids = []
        filenames = []
        subject_label = np.load(label_path)
        for filename in os.listdir(data_path):
            filenames.append(filename)
        filenames.sort()

        for j in range(len(filenames)):
            trial_label = subject_label[j]
            path = data_path + filenames[j]
            subject_feature = np.load(path)
            for trial_feature in subject_feature:
                feature_list.append(trial_feature)
                label_list.append(trial_label)
                sample_subject_ids.append(j + 1)
        X = np.array(feature_list)
        y = np.array(label_list)
        sample_subject_ids = np.array(sample_subject_ids)
        X, y, sample_subject_ids = shuffle(X, y, sample_subject_ids, random_state=42)

        return X, y[:, 0], sample_subject_ids

    def select_split(self, cache_entry, flag=None):
        if flag == "TRAIN":
            ids = self.train_ids
            print("train ids:", ids)
        elif flag == "VAL":
            ids = self.val_ids
            print("val ids:", ids)
        elif flag == "TEST":
            ids = self.test_ids
            print("test ids:", ids)
        else:
            ids = list(np.unique(cache_entry["sample_subject_ids"]))
            print("all ids:", ids)

        mask = np.isin(cache_entry["sample_subject_ids"], np.array(ids))
        return cache_entry["X"][mask], cache_entry["y"][mask]

    def __getitem__(self, index):
        return torch.from_numpy(self.X[index]), torch.from_numpy(
            np.asarray(self.y[index])
        )

    def __len__(self):
        return len(self.y)
