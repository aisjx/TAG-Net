import argparse
import csv
import os
import torch
from exp.exp_classification import Exp_Classification
import random
import numpy as np


def summarize_metrics(metrics_list):
    metric_names = ("Accuracy", "Precision", "Recall", "F1", "AUROC", "AUPRC")
    means = {
        metric: float(np.mean([metrics[metric] for metrics in metrics_list]))
        for metric in metric_names
    }
    stds = {
        metric: float(np.std([metrics[metric] for metrics in metrics_list]))
        for metric in metric_names
    }
    return means, stds


def log_summary(args, means, stds):
    mean_line = (
        f"Mean accuracy: {means['Accuracy']:.4f}, precision: {means['Precision']:.4f},"
        f"recall: {means['Recall']:.4f}, f1: {means['F1']:.4f}, "
        f"AUROC: {means['AUROC']:.4f}, AUPRC: {means['AUPRC']:.4f}"
    )
    std_line = (
        f"Std accuracy: {stds['Accuracy']:.4f}, precision: {stds['Precision']:.4f},"
        f"recall: {stds['Recall']:.4f}, f1: {stds['F1']:.4f}, "
        f"AUROC: {stds['AUROC']:.4f}, AUPRC: {stds['AUPRC']:.4f}"
    )

    print(mean_line)
    print(std_line)

    folder_path = os.path.join(
        "./results", args.task_name, args.model_id, args.model
    )
    os.makedirs(folder_path, exist_ok=True)
    with open(os.path.join(folder_path, "result_classification.txt"), "a") as f:
        f.write(mean_line + "\n")
        f.write(std_line + "\n\n")


def append_seed_metrics_csv(args, setting, metrics):
    csv_path = args.metrics_csv
    if not csv_path:
        return

    csv_dir = os.path.dirname(csv_path)
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)

    fieldnames = [
        "task_name",
        "dataset",
        "method",
        "model",
        "model_id",
        "setting",
        "seed",
        "ablation_tag",
        "d_model",
        "d_ff",
        "n_heads",
        "e_layers",
        "resolution_list",
        "nodedim",
        "batch_size",
        "learning_rate",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "auroc",
        "auprc",
    ]
    row = {
        "task_name": args.task_name,
        "dataset": args.data,
        "method": args.metrics_method_name or args.model_id,
        "model": args.model,
        "model_id": args.model_id,
        "setting": setting,
        "seed": args.seed,
        "ablation_tag": build_ablation_tag(args),
        "d_model": args.d_model,
        "d_ff": args.d_ff,
        "n_heads": args.n_heads,
        "e_layers": args.e_layers,
        "resolution_list": args.resolution_list,
        "nodedim": args.nodedim,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "accuracy": metrics["Accuracy"],
        "precision": metrics["Precision"],
        "recall": metrics["Recall"],
        "f1": metrics["F1"],
        "auroc": metrics["AUROC"],
        "auprc": metrics["AUPRC"],
    }

    file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def build_ablation_tag(args):
    tags = []
    disable_post_channel_injection = (
        getattr(args, "disable_post_channel_injection", False)
        or args.disable_channel_context_injection
        or getattr(args, "disable_post_router_and_channel_injection", False)
    )
    disable_post_resolution_router = (
        args.disable_post_resolution_router
        or getattr(args, "disable_post_router_and_channel_injection", False)
    )
    if args.disable_frequency_branch:
        tags.append("noFreq")
    if args.disable_temporal_branch:
        tags.append("noTemp")
    if args.disable_channel_branch:
        tags.append("noChan")
    if args.disable_resolution_router:
        tags.append("noRouter")
    elif disable_post_resolution_router:
        tags.append("noPostRouter")
    if disable_post_channel_injection:
        tags.append("noChanInject")
    if args.disable_cross_graph_interaction:
        tags.append("noCrossGraph")
    if args.disable_channel_resolution_module:
        tags.append("noChanResEnc")
    if args.disable_structure_alignment:
        tags.append("noStructAlign")
    if args.single_gnn:
        tags.append("singleGNN")
        if args.single_gnn_fusion != "gated":
            tags.append(f"sgFusion{args.single_gnn_fusion}")
    if args.structure_delta_scale != 0.1:
        tags.append(f"sDelta{args.structure_delta_scale:g}")
    if getattr(args, 'use_structure_loss', False):
        tags.append(f"strLoss_dag{args.lambda_dag}_sp{args.lambda_sparse}")
    return "base" if not tags else "_".join(tags)


def build_setting(args):
    return "{}_{}_{}_dm{}_df{}_nh{}_el{}_res{}_node{}_{}_seed{}_bs{}_lr{}".format(
        args.model_id,
        args.model,
        args.data,
        args.d_model,
        args.d_ff,
        args.n_heads,
        args.e_layers,
        args.resolution_list,
        args.nodedim,
        build_ablation_tag(args),
        args.seed,
        args.batch_size,
        args.learning_rate,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TimesNet")

    # basic config
    parser.add_argument("--task_name", type=str, default="classification")
    parser.add_argument("--is_training", type=int, default=1, help="status")
    parser.add_argument("--model_id", type=str, default="APAVA-Subject", help="model id")
    parser.add_argument("--model", type=str, default="MedGNN", help="[MedGNN, Medformer, iTransformer]")
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    parser.add_argument("--data", type=str, default="APAVA", help="dataset type")
    parser.add_argument("--root_path", type=str, default="../dataset/APAVA", help="root path of the data file")
    parser.add_argument("--freq", type=str, default="h",
        help="freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h")

    parser.add_argument("--d_model", type=int, default=256, help="dimension of model")
    parser.add_argument("--d_ff", type=int, default=512, help="dimension of fcn")
    parser.add_argument("--n_heads", type=int, default=8, help="num of heads")
    parser.add_argument("--e_layers", type=int, default=4, help="num of encoder layers")
    parser.add_argument("--d_layers", type=int, default=1, help="num of decoder layers")
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout")
    parser.add_argument("--embed", type=str, default="timeF", help="time features encoding, options:[timeF, fixed, learned]")
    parser.add_argument("--activation", type=str, default="gelu", help="activation")
    parser.add_argument("--output_attention", action="store_true", help="whether to output attention in encoder")

    parser.add_argument("--patch_len_list", type=str, default="2,2,2,4,4,4,16,16,16,16,32,32,32,32,32", help="a list of patch len used in Medformer")
    parser.add_argument("--single_channel", action="store_true", default=False, help="whether to use single channel patching for Medformer")
    parser.add_argument("--augmentations", type=str, default="none,drop0.35",
                        help="a comma-seperated list of augmentation types (none, jitter or scale). Append numbers to specify the strength of the augmentation, e.g., jitter0.1")

    # MedGNN
    parser.add_argument('--resolution_list', type=str, default="2,4,6,8")
    parser.add_argument('--nodedim', type=int, default=10)
    parser.add_argument('--low_freq_ratio', type=float, default=0.5)
    parser.add_argument("--channel_layers", type=int, default=2)
    parser.add_argument("--disable_frequency_branch", action="store_true", default=False)
    parser.add_argument("--disable_temporal_branch", action="store_true", default=False)
    parser.add_argument("--disable_channel_branch", action="store_true", default=False)
    parser.add_argument("--disable_resolution_router", action="store_true", default=False)
    parser.add_argument("--disable_post_resolution_router", action="store_true", default=False)
    parser.add_argument("--disable_channel_context_injection", action="store_true", default=False)
    parser.add_argument(
        "--disable_post_channel_injection",
        action="store_true",
        default=False,
        help="Alias of disabling the post-channel injection stage in CAMRE.",
    )
    parser.add_argument(
        "--disable_post_router_and_channel_injection",
        action="store_true",
        default=False,
        help="Disable both post-resolution router and post-channel injection together.",
    )
    parser.add_argument("--disable_cross_graph_interaction", action="store_true", default=False)
    parser.add_argument("--disable_channel_resolution_module", action="store_true", default=False)
    parser.add_argument("--disable_structure_alignment", action="store_true", default=False)
    parser.add_argument("--single_gnn", action="store_true", default=False)
    parser.add_argument(
        "--single_gnn_fusion",
        type=str,
        default="gated",
        choices=["gated", "add", "concat", "hint"],
        help="Fusion strategy used before the shared single GNN when --single_gnn is enabled.",
    )
    parser.add_argument(
        "--structure_delta_scale",
        type=float,
        default=0.1,
        help="Strength of residual structure deviation around the shared adjacency.",
    )

    # Structure Loss (DAG + Sparsity) for MedGNN
    parser.add_argument('--use_structure_loss', action='store_true', default=False,
                        help='Enable DAG + Sparsity structural loss on adjacency matrices')
    parser.add_argument('--lambda_dag', type=float, default=0.5,
                        help='Weight for DAG constraint loss (NOTEARS cycle penalty)')
    parser.add_argument('--lambda_sparse', type=float, default=0.01,
                        help='Weight for sparsity loss (L1 norm of adjacency)')
    
    # optimization
    parser.add_argument("--num_workers", type=int, default=10, help="data loader num workers")
    parser.add_argument("--itr", type=int, default=1, help="experiments times")
    parser.add_argument("--train_epochs", type=int, default=10, help="train epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size of train input data")
    parser.add_argument("--patience", type=int, default=3, help="early stopping patience")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="optimizer learning rate")
    parser.add_argument(
        "--metrics_csv",
        type=str,
        default="./results/seed_metrics.csv",
        help="CSV file used to append per-seed test metrics. Set empty string to disable.",
    )
    parser.add_argument(
        "--metrics_method_name",
        type=str,
        default="",
        help="Optional method name written to metrics_csv; defaults to model_id.",
    )
    parser.add_argument("--des", type=str, default="test", help="exp description")
    parser.add_argument("--loss", type=str, default="MSE", help="loss function")
    parser.add_argument("--lradj", type=str, default="type1", help="adjust learning rate")
    parser.add_argument("--use_amp", action="store_true", default=False, help="use automatic mixed precision training")
    parser.add_argument("--swa", action="store_true", default=False, help="use stochastic weight averaging")
    parser.add_argument(
        "--memory_unit",
        type=str,
        default="MB",
        choices=["MB", "GB", "mb", "gb"],
        help="unit used when printing GPU memory statistics",
    )
    # GPU
    parser.add_argument("--use_gpu", type=bool, default=True, help="use gpu")
    parser.add_argument("--gpu", type=int, default=0, help="gpu")
    parser.add_argument("--use_multi_gpu", help="use multiple gpus", default=False)
    parser.add_argument("--devices", type=str, default="0, 1, 2, 3", help="device ids of multiple gpus")

    args = parser.parse_args()
    args.memory_unit = args.memory_unit.upper()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(" ", "")
        device_ids = args.devices.split(",")
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print("Args in experiment:")
    print(args)

    if args.task_name == "classification":
        Exp = Exp_Classification

    avg_metrics = []

    if args.is_training:
        for ii in range(args.itr):
            seed = 41 + ii
            random.seed(seed)
            os.environ["PYTHONHASHSEED"] = str(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # comment out the following lines if you are using dilated convolutions, e.g., TCN
            # otherwise it will slow down the training extremely
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True


            # setting record of experiments
            args.seed = seed
            setting = build_setting(args)

            exp = Exp(args)  # set experiments
            print(
                ">>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>".format(setting)
            )
            exp.train(setting)

            print(
                ">>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(setting)
            )
            test_metrics = exp.test(setting)
            avg_metrics.append(test_metrics)
            append_seed_metrics_csv(args, setting, test_metrics)
            torch.cuda.empty_cache()
    else:
        for ii in range(args.itr):
            seed = 41 + ii
            random.seed(seed)
            os.environ["PYTHONHASHSEED"] = str(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # comment out the following lines if you are using dilated convolutions
            # otherwise it will slow down the training extremely
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

            args.seed = seed
            setting = build_setting(args)

            exp = Exp(args)  # set experiments
            print(
                ">>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(setting)
            )
            test_metrics = exp.test(setting, test=1)
            avg_metrics.append(test_metrics)
            append_seed_metrics_csv(args, setting, test_metrics)
            torch.cuda.empty_cache()

    if avg_metrics:
        means, stds = summarize_metrics(avg_metrics)
        log_summary(args, means, stds)
