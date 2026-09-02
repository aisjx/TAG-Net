from copy import deepcopy
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, cal_accuracy
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import random
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

warnings.filterwarnings("ignore")


def dag_loss(A):
    """
    DAG constraint loss using NOTEARS algorithm.
    Constraint: tr(exp(A^T @ A)) - d = 0
    This penalizes cyclic structures in the adjacency matrix.

    Args:
        A: adjacency matrix of shape (d, d) or (batch, d, d)
           For (batch, d, d), we compute loss on the mean adjacency.
    """
    # Handle batch dimension if present
    if A.dim() == 3:
        A = A.mean(dim=0)  # Average over batch

    d = A.shape[0]
    # Zero out diagonal to prevent self-loops
    A_no_diag = A * (1 - torch.eye(d, device=A.device))
    expm_A = torch.matrix_exp(A_no_diag @ A_no_diag)
    h_A = torch.trace(expm_A) - d
    return h_A


def structure_loss(adj_list, lambda_dag=0.5, lambda_sparse=0.01):
    """
    Compute combined DAG + sparsity loss for a list of adjacency matrices.

    Args:
        adj_list: list of adjacency matrices, each of shape (d, d) or (batch, d, d)
        lambda_dag: weight for DAG constraint
        lambda_sparse: weight for sparsity (L1) penalty

    Returns:
        scalar loss value
    """
    loss = 0.0
    for A in adj_list:
        loss += lambda_dag * dag_loss(A)
        # For sparsity, also handle batch dimension
        if A.dim() == 3:
            A_flat = A.view(-1, A.shape[-1])
            loss += lambda_sparse * torch.norm(A_flat, p=1)
        else:
            loss += lambda_sparse * torch.norm(A, p=1)
    return loss


class Exp_Classification(Exp_Basic):
    def __init__(self, args):
        self._data_cache = {}
        self.epoch_durations = []
        self.epoch_memory_stats = []
        self.efficiency_log_path = None
        super().__init__(args)

        self.swa_model = optim.swa_utils.AveragedModel(self.model)
        self.swa = args.swa

    def _sync_cuda(self):
        if self.args.use_gpu and torch.cuda.is_available():
            torch.cuda.synchronize()

    def _profile_device_ids(self):
        if not (self.args.use_gpu and torch.cuda.is_available()):
            return []
        if self.args.use_multi_gpu:
            return list(self.args.device_ids)
        return [self.args.gpu]

    def _reset_peak_memory_stats(self):
        for device_id in self._profile_device_ids():
            torch.cuda.reset_peak_memory_stats(device_id)

    def _format_memory(self, num_bytes):
        unit = self.args.memory_unit.upper()
        if unit == "GB":
            divisor = 1024 ** 3
            suffix = "GB"
        else:
            divisor = 1024 ** 2
            suffix = "MB"
        return f"{num_bytes / divisor:.2f} {suffix}"

    def _collect_gpu_memory_stats(self):
        device_ids = self._profile_device_ids()
        if not device_ids:
            return None

        total_allocated = 0
        total_reserved = 0
        total_peak_allocated = 0
        total_peak_reserved = 0
        per_device_lines = []

        for device_id in device_ids:
            allocated = torch.cuda.memory_allocated(device_id)
            reserved = torch.cuda.memory_reserved(device_id)
            peak_allocated = torch.cuda.max_memory_allocated(device_id)
            peak_reserved = torch.cuda.max_memory_reserved(device_id)

            total_allocated += allocated
            total_reserved += reserved
            total_peak_allocated += peak_allocated
            total_peak_reserved += peak_reserved

            per_device_lines.append(
                "cuda:{device} alloc={alloc}, reserved={reserved}, "
                "peak_alloc={peak_alloc}, peak_reserved={peak_reserved}".format(
                    device=device_id,
                    alloc=self._format_memory(allocated),
                    reserved=self._format_memory(reserved),
                    peak_alloc=self._format_memory(peak_allocated),
                    peak_reserved=self._format_memory(peak_reserved),
                )
            )

        return {
            "allocated": total_allocated,
            "reserved": total_reserved,
            "peak_allocated": total_peak_allocated,
            "peak_reserved": total_peak_reserved,
            "summary_line": (
                "GPU Memory ({unit}, summed over tracked devices): alloc={alloc}, "
                "reserved={reserved}, peak_alloc={peak_alloc}, peak_reserved={peak_reserved}".format(
                    unit=self.args.memory_unit.upper(),
                    alloc=self._format_memory(total_allocated),
                    reserved=self._format_memory(total_reserved),
                    peak_alloc=self._format_memory(total_peak_allocated),
                    peak_reserved=self._format_memory(total_peak_reserved),
                )
            ),
            "per_device_lines": per_device_lines,
        }

    def _append_efficiency_log(self, setting, lines):
        if self.efficiency_log_path is None:
            folder_path = os.path.join(
                "./results", self.args.task_name, self.args.model_id, self.args.model
            )
            os.makedirs(folder_path, exist_ok=True)
            self.efficiency_log_path = os.path.join(
                folder_path, "efficiency_classification.txt"
            )

        with open(self.efficiency_log_path, "a") as f:
            f.write(setting + "\n")
            for line in lines:
                f.write(line + "\n")
            f.write("\n")

    def _build_model(self):
        # model input depends on data
        # Build a temporary test split first to infer shapes before caching loaders.
        test_data, _ = data_provider(self.args, "TEST")
        self.args.seq_len = test_data.max_seq_len  # redefine seq_len
        self.args.pred_len = 0
        # self.args.enc_in = train_data.feature_df.shape[1]
        # self.args.num_class = len(train_data.class_names)
        self.args.enc_in = test_data.X.shape[2]  # redefine enc_in
        self.args.num_class = len(np.unique(test_data.y))
        # Clear any stale cached loader built before seq_len was updated.
        self._data_cache.clear()
        # model init
        model = (
            self.model_dict[self.args.model].Model(self.args).float()
        )  # pass args to model
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        if flag in self._data_cache:
            return self._data_cache[flag]
        random.seed(self.args.seed)
        data_set, data_loader = data_provider(self.args, flag)
        self._data_cache[flag] = (data_set, data_loader)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        preds = []
        trues = []
        if self.swa:
            self.swa_model.eval()
        else:
            self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                if self.swa:
                    outputs = self.swa_model(batch_x, padding_mask, None, None)
                else:
                    outputs = self.model(batch_x, padding_mask, None, None)

                pred = outputs.detach().cpu()
                loss = criterion(pred, label.long().cpu())
                total_loss.append(loss.item())

                preds.append(outputs.detach())
                trues.append(label)

        total_loss = np.average(total_loss)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        probs = torch.nn.functional.softmax(
            preds
        )  # (total_samples, num_classes) est. prob. for each class and sample
        trues_onehot = (
            torch.nn.functional.one_hot(
                trues.reshape(
                    -1,
                ).to(torch.long),
                num_classes=self.args.num_class,
            )
            .float()
            .cpu()
            .numpy()
        )
        # print(trues_onehot.shape)
        predictions = (
            torch.argmax(probs, dim=1).cpu().numpy()
        )  # (total_samples,) int class index for each sample
        probs = probs.cpu().numpy()
        trues = trues.flatten().cpu().numpy()
        # accuracy = cal_accuracy(predictions, trues)
        metrics_dict = {
            "Accuracy": accuracy_score(trues, predictions),
            "Precision": precision_score(trues, predictions, average="macro"),
            "Recall": recall_score(trues, predictions, average="macro"),
            "F1": f1_score(trues, predictions, average="macro"),
            "AUROC": roc_auc_score(trues_onehot, probs, multi_class="ovr"),
            "AUPRC": average_precision_score(trues_onehot, probs, average="macro"),
        }

        if self.swa:
            self.swa_model.train()
        else:
            self.model.train()
        return total_loss, metrics_dict

    def train(self, setting):
        train_data, train_loader = self._get_data(flag="TRAIN")
        vali_data, vali_loader = self._get_data(flag="VAL")
        test_data, test_loader = self._get_data(flag="TEST")
        self.epoch_durations = []
        self.epoch_memory_stats = []
        print(train_data.X.shape)
        print(train_data.y.shape)
        print(vali_data.X.shape)
        print(vali_data.y.shape)
        print(test_data.X.shape)
        print(test_data.y.shape)

        path = (
            "./checkpoints/"
            + self.args.task_name
            + "/"
            + self.args.model_id
            + "/"
            + self.args.model
            + "/"
            + setting
            + "/"
        )
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(
            patience=self.args.patience, verbose=True, delta=1e-5
        )

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()


        total_params = 0
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad: continue
            param = parameter.numel()
            total_params += param
        print(f"Total Trainable Params: {total_params}")


        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            self._reset_peak_memory_stats()
            self._sync_cuda()
            train_epoch_start = time.time()

            for i, (batch_x, label, padding_mask) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                # Get adjacency matrices if structure loss is enabled
                use_structure_loss = getattr(self.args, 'use_structure_loss', False)
                if use_structure_loss and hasattr(self.model, 'forward_with_adjs'):
                    outputs, _, adj_list = self.model.forward_with_adjs(
                        batch_x, padding_mask, None, None
                    )
                    lambda_dag = getattr(self.args, 'lambda_dag', 0.5)
                    lambda_sparse = getattr(self.args, 'lambda_sparse', 0.01)
                    loss_struct = structure_loss(adj_list, lambda_dag, lambda_sparse)
                    loss = criterion(outputs, label.long()) + loss_struct
                else:
                    outputs = self.model(batch_x, padding_mask, None, None)
                    loss = criterion(outputs, label.long())
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print(
                        "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(
                            i + 1, epoch + 1, loss.item()
                        )
                    )
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * (
                        (self.args.train_epochs - epoch) * train_steps - i
                    )
                    print(
                        "\tspeed: {:.4f}s/iter; left time: {:.4f}s".format(
                            speed, left_time
                        )
                    )
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                model_optim.step()

            self._sync_cuda()
            train_epoch_seconds = time.time() - train_epoch_start
            self.epoch_durations.append(train_epoch_seconds)
            gpu_memory_stats = self._collect_gpu_memory_stats()
            if gpu_memory_stats is not None:
                self.epoch_memory_stats.append(gpu_memory_stats)

            self.swa_model.update_parameters(self.model)

            train_loss = np.average(train_loss)
            vali_loss, val_metrics_dict = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_metrics_dict = self.vali(test_data, test_loader, criterion)

            avg_epoch_seconds = float(np.mean(self.epoch_durations))
            epoch_summary_lines = [
                "Epoch: {} | train_time: {:.4f}s | avg_train_time: {:.4f}s/epoch".format(
                    epoch + 1, train_epoch_seconds, avg_epoch_seconds
                )
            ]
            if gpu_memory_stats is not None:
                epoch_summary_lines.append(gpu_memory_stats["summary_line"])
                epoch_summary_lines.extend(gpu_memory_stats["per_device_lines"])
            else:
                epoch_summary_lines.append("GPU Memory: CUDA not in use, skipped.")

            for line in epoch_summary_lines:
                print(line)

            print(
                f"Epoch: {epoch + 1}, Steps: {train_steps}, | Train Loss: {train_loss:.5f}\n"
                f"Validation results --- Loss: {vali_loss:.5f}, "
                f"Accuracy: {val_metrics_dict['Accuracy']:.5f}, "
                f"Precision: {val_metrics_dict['Precision']:.5f}, "
                f"Recall: {val_metrics_dict['Recall']:.5f}, "
                f"F1: {val_metrics_dict['F1']:.5f}, "
                f"AUROC: {val_metrics_dict['AUROC']:.5f}, "
                f"AUPRC: {val_metrics_dict['AUPRC']:.5f}\n"
                f"Test results --- Loss: {test_loss:.5f}, "
                f"Accuracy: {test_metrics_dict['Accuracy']:.5f}, "
                f"Precision: {test_metrics_dict['Precision']:.5f}, "
                f"Recall: {test_metrics_dict['Recall']:.5f} "
                f"F1: {test_metrics_dict['F1']:.5f}, "
                f"AUROC: {test_metrics_dict['AUROC']:.5f}, "
                f"AUPRC: {test_metrics_dict['AUPRC']:.5f}\n"
            )
            self._append_efficiency_log(
                setting,
                epoch_summary_lines
                + [
                    "Train Loss: {:.5f}".format(train_loss),
                    (
                        "Validation --- Loss: {:.5f}, Accuracy: {:.5f}, Precision: {:.5f}, "
                        "Recall: {:.5f}, F1: {:.5f}, AUROC: {:.5f}, AUPRC: {:.5f}"
                    ).format(
                        vali_loss,
                        val_metrics_dict["Accuracy"],
                        val_metrics_dict["Precision"],
                        val_metrics_dict["Recall"],
                        val_metrics_dict["F1"],
                        val_metrics_dict["AUROC"],
                        val_metrics_dict["AUPRC"],
                    ),
                    (
                        "Test --- Loss: {:.5f}, Accuracy: {:.5f}, Precision: {:.5f}, "
                        "Recall: {:.5f}, F1: {:.5f}, AUROC: {:.5f}, AUPRC: {:.5f}"
                    ).format(
                        test_loss,
                        test_metrics_dict["Accuracy"],
                        test_metrics_dict["Precision"],
                        test_metrics_dict["Recall"],
                        test_metrics_dict["F1"],
                        test_metrics_dict["AUROC"],
                        test_metrics_dict["AUPRC"],
                    ),
                ],
            )
            early_stopping(
                -val_metrics_dict["F1"],
                self.swa_model if self.swa else self.model,
                path,
            )
            if early_stopping.early_stop:
                print("Early stopping")
                break
            """if (epoch + 1) % 5 == 0:
                adjust_learning_rate(model_optim, epoch + 1, self.args)"""

        best_model_path = path + "checkpoint.pth"
        if self.swa:
            self.swa_model.load_state_dict(torch.load(best_model_path, map_location='cuda' if torch.cuda.is_available() else 'cpu'))
        else:
            self.model.load_state_dict(torch.load(best_model_path, map_location='cuda' if torch.cuda.is_available() else 'cpu'))

        if self.epoch_durations:
            overall_lines = [
                "Training Efficiency Summary",
                "Average train time per epoch: {:.4f}s".format(
                    float(np.mean(self.epoch_durations))
                ),
                "Fastest epoch: {:.4f}s".format(float(np.min(self.epoch_durations))),
                "Slowest epoch: {:.4f}s".format(float(np.max(self.epoch_durations))),
            ]
            if self.epoch_memory_stats:
                max_peak_allocated = max(
                    item["peak_allocated"] for item in self.epoch_memory_stats
                )
                max_peak_reserved = max(
                    item["peak_reserved"] for item in self.epoch_memory_stats
                )
                overall_lines.append(
                    "Max peak GPU memory across epochs: alloc={}, reserved={}".format(
                        self._format_memory(max_peak_allocated),
                        self._format_memory(max_peak_reserved),
                    )
                )
            for line in overall_lines:
                print(line)
            self._append_efficiency_log(setting, overall_lines)

        return self.model

    def test(self, setting, test=0):
        vali_data, vali_loader = self._get_data(flag="VAL")
        test_data, test_loader = self._get_data(flag="TEST")
        if test:
            print("loading model")
            path = (
                "./checkpoints/"
                + self.args.task_name
                + "/"
                + self.args.model_id
                + "/"
                + self.args.model
                + "/"
                + setting
                + "/"
            )
            model_path = path + "checkpoint.pth"
            if not os.path.exists(model_path):
                raise Exception("No model found at %s" % model_path)
            if self.swa:
                self.swa_model.load_state_dict(torch.load(model_path, map_location='cuda'))
            else:
                self.model.load_state_dict(torch.load(model_path, map_location='cuda'))

        criterion = self._select_criterion()
        vali_loss, val_metrics_dict = self.vali(vali_data, vali_loader, criterion)
        test_loss, test_metrics_dict = self.vali(test_data, test_loader, criterion)

        # result save
        folder_path = (
            "./results/"
            + self.args.task_name
            + "/"
            + self.args.model_id
            + "/"
            + self.args.model
            + "/"
        )
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # # save the adjacency matrix
        # adj_path = (
        #         "./adj_weight/"
        #         + self.args.task_name
        #         + "/"
        #         + self.args.model_id
        #         + "/"
        #         + self.args.model
        #         + "/"
        # )
        # if not os.path.exists(adj_path):
        #     os.makedirs(adj_path)

        # for l in range(len(adjacency_matrix_list)):
        #     matrix = adjacency_matrix_list[l].detach().cpu().numpy()
        #     np.save(adj_path + 'matrix_{}.npy'.format(l), matrix)


        print(
            f"Validation results --- Loss: {vali_loss:.5f}, "
            f"Accuracy: {val_metrics_dict['Accuracy']:.5f}, "
            f"Precision: {val_metrics_dict['Precision']:.5f}, "
            f"Recall: {val_metrics_dict['Recall']:.5f}, "
            f"F1: {val_metrics_dict['F1']:.5f}, "
            f"AUROC: {val_metrics_dict['AUROC']:.5f}, "
            f"AUPRC: {val_metrics_dict['AUPRC']:.5f}\n"
            f"Test results --- Loss: {test_loss:.5f}, "
            f"Accuracy: {test_metrics_dict['Accuracy']:.5f}, "
            f"Precision: {test_metrics_dict['Precision']:.5f}, "
            f"Recall: {test_metrics_dict['Recall']:.5f}, "
            f"F1: {test_metrics_dict['F1']:.5f}, "
            f"AUROC: {test_metrics_dict['AUROC']:.5f}, "
            f"AUPRC: {test_metrics_dict['AUPRC']:.5f}\n"
        )
        file_name = "result_classification.txt"
        f = open(os.path.join(folder_path, file_name), "a")
        f.write(setting + "  \n")
        f.write(
            f"Validation results --- Loss: {vali_loss:.5f}, "
            f"Accuracy: {val_metrics_dict['Accuracy']:.5f}, "
            f"Precision: {val_metrics_dict['Precision']:.5f}, "
            f"Recall: {val_metrics_dict['Recall']:.5f}, "
            f"F1: {val_metrics_dict['F1']:.5f}, "
            f"AUROC: {val_metrics_dict['AUROC']:.5f}, "
            f"AUPRC: {val_metrics_dict['AUPRC']:.5f}\n"
            f"Test results --- Loss: {test_loss:.5f}, "
            f"Accuracy: {test_metrics_dict['Accuracy']:.5f}, "
            f"Precision: {test_metrics_dict['Precision']:.5f}, "
            f"Recall: {test_metrics_dict['Recall']:.5f}, "
            f"F1: {test_metrics_dict['F1']:.5f}, "
            f"AUROC: {test_metrics_dict['AUROC']:.5f}, "
            f"AUPRC: {test_metrics_dict['AUPRC']:.5f}\n"
        )
        f.write("\n")
        f.write("\n")
        f.close()
        return test_metrics_dict
