import argparse
import torch
from torch.backends import cudnn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, BertTokenizer, AutoModel, utils
from datasets import load_dataset

from data.dataset import TrainDataset
from simcse import SimCSE

utils.logging.set_verbosity_error()  # Suppress standard warnings

parser = argparse.ArgumentParser(description="PyTorch SimCSE implementation")

parser.add_argument('--mode', choices=['sup', 'unsup'], default='unsup', help='Train with supervised or unsupervised manner.')
parser.add_argument('--arch', choices=['bert', 'roberta'], default='bert', help='Choose the model you want to train with.')
parser.add_argument('--disable_cuda', default=False, help='Whether you want to use cuda or not.')
parser.add_argument('--n_views', default=2)
# Parameters
parser.add_argument("--epochs", default=50, help="Set up the number of epochs you want to train.")
parser.add_argument("--batch_size", default=8, help="Set up the size of each batch you want to train.")
parser.add_argument("--lr", default=1e-5, help="Set up the learning rate.")
parser.add_argument("--max_len", default=32, help="Set up the maximum total input sequence length after tokenization.")
parser.add_argument("--pooling", choices=['cls', 'pooler', 'last-avg', 'first-last-avg'], default='cls', help='Choose the pooling method')
parser.add_argument("--temperature", default=0.05, help="Set uo the temperature parameter.")
# Additional HP
parser.add_argument("--weight_decay", default=5e-4, help="Set up the weight decay for optimizer.")
parser.add_argument("--log_every_n_steps", default=10, help="Frequency of keeping log")
parser.add_argument("--fp16_precision", action='store_true', help='Whether or not to use 16-bit precision GPU training.')
# Files
parser.add_argument("--train_data", default="./data/wiki1m_for_simcse.txt",
                    help="Choose the dataset you want to train with.")  # wiki1m_for_simcse.txt; nli_for_simcse.csv
parser.add_argument("--save_data", default=True)
# parser.add_argument("--log_dir", default="./log/")
# GPU
parser.add_argument('--gpu-index', default=0, type=int, help='Gpu index.')

def main():
    args = parser.parse_args()
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1

    # Dataset
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.
    data_files = {}
    if args.train_data is not None:
        data_files["train"] = args.train_data
    extension = args.train_data.split(".")[-1]
    if extension == "txt":
        extension = "text"
        datasets = load_dataset(extension, data_files=data_files)
    elif extension == "csv":
        datasets = load_dataset(extension, data_files=data_files,
                                delimiter="\t" if "tsv" in args.train_data else ",")
    else:
        raise ValueError("Error with the type of file.")

    # Tokenizer and Model
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    # tokenizer = BertTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained("bert-base-uncased", output_attentions=True)

    train_dataset = TrainDataset(data=datasets, args=args, tokenizer=tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    # optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
                                                           last_epoch=-1)

    #  It’s a no-op if the 'gpu_index' argument is a negative integer or None.
    with torch.cuda.device(args.gpu_index):
        simcse = SimCSE(args=args, model=model, tokenizer=tokenizer,
                        optimizer=optimizer, scheduler=scheduler)
        if args.mode == "sup":
            simcse.train_sup(train_loader=train_loader)
        elif args.mode == "unsup":
            simcse.train_unsup(train_loader=train_loader)
        else:
            raise ValueError("Unrecognised training mode")


if __name__ == '__main__':
    main()