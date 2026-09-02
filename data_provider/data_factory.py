from data_provider.data_loader import (
    APAVALoader,
    ADFDLoader,
    ADFDDependentLoader,
    TDBRAINLoader,
    PTBLoader,
    PTBXLLoader,
)
from data_provider.uea import collate_fn
from torch.utils.data import DataLoader
from functools import partial

data_dict = {
    "APAVA": APAVALoader,
    "TDBRAIN": TDBRAINLoader,
    "ADFD": ADFDLoader,
    "ADFD-Sample": ADFDDependentLoader,
    "PTB": PTBLoader,
    "PTB-XL": PTBXLLoader,
}

def data_provider(args, flag):
    Data = data_dict[args.data]

    timeenc = 0 if args.embed != "timeF" else 1

    if flag == "test":
        shuffle_flag = False
        drop_last = True
        if args.task_name == "classification":
            batch_size = args.batch_size
        else:
            batch_size = 1
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    if args.task_name == "classification":
        drop_last = False
        data_set = Data(
            root_path=args.root_path,
            flag=flag,
        )

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            collate_fn=partial(collate_fn, max_len=getattr(args, "seq_len", 128)),
        )
        return data_set, data_loader
