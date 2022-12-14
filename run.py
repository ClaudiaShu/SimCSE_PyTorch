import argparse

import torch
from torch.backends import cudnn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer,  utils

from data.dataset import SimcseEvalDataset, SimcseTrainDataset
from function import seed_everything
from model.models import SimcseModelUnsup
from simcse import SimCSE

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

utils.logging.set_verbosity_error()  # Suppress standard warnings

parser = argparse.ArgumentParser(description="PyTorch SimCSE implementation")

parser.add_argument('--mode', choices=['sup', 'unsup'], type=str, default='sup',
                    help='Train with supervised or unsupervised manner.')
parser.add_argument('--disable_cuda', default=False, help='Whether you want to use cuda or not.')
parser.add_argument('--n_views', default=2, type=int)
parser.add_argument('--do_mlm', default=True, help='Choose to add mlm function or not.')
# Parameters
parser.add_argument("--epochs", default=5, type=int, help="Set up the number of epochs you want to train.")
parser.add_argument("--batch_size", default=64, type=int, help="Set up the size of each batch you want to train.")
parser.add_argument("--lr", default=3e-5, type=float, help="Set up the learning rate.")
parser.add_argument("--max_len", default=64, type=int, help="Set up the maximum total input sequence length after tokenization.")
parser.add_argument("--pooling", choices=['cls', 'pooler', 'last-avg', 'first-last-avg'], default='cls', type=str, help='Choose the pooling method')
parser.add_argument("--temperature", default=0.05, type=float, help="Set uo the temperature parameter.")
parser.add_argument("--dropout", default = 0.1, type=float, help="Set up the dropout ratio")
parser.add_argument("--pretrained_model", default="bert-base-uncased", type = str)
parser.add_argument("--num_workers",default=4, type=int)
# princeton-nlp/sup-simcse-bert-base-uncased
# princeton-nlp/unsup-simcse-roberta-base
# bert-base-uncased
# roberta-base

# Additional HP
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--weight_decay", default=5e-4, type=float, help="Set up the weight decay for optimizer.")
parser.add_argument("--log_every_n_steps", default=100, type=int, help="Frequency of keeping log")
parser.add_argument("--fp16_precision", action='store_true', help='Whether or not to use 16-bit precision GPU training.')
# Files
parser.add_argument("--train_data", type=str, default="./data/training/nli_for_simcse.csv",
                    help="Choose the dataset you want to train with.")  # wiki1m_for_simcse.txt; nli_for_simcse.csv
parser.add_argument("--dev_file", type=str, default="data/stsbenchmark/sts-dev.csv")
parser.add_argument("--test_file", type=str, default="data/stsbenchmark/sts-test.csv")
parser.add_argument("--save_data", default=True)
parser.add_argument("--output_path", default='test')
# GPU
parser.add_argument('--gpu-index', default=0, type=int, help='Gpu index.')

def main():
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1

    if args.output_path is not None:
        pass
    else:
        import socket
        from datetime import datetime
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        log_dir = os.path.join('results', args.mode, current_time)
        args.output_path = log_dir
    os.makedirs(args.output_path, exist_ok=True)

    # Bert or Roberta
    if 'roberta' in args.pretrained_model:
        args.arch = 'roberta'
    else:
        args.arch = 'bert'

    # Define tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
    model = SimcseModelUnsup(args=args)
    model = model.to(args.device)

    # Load dataset
    train_dataset = SimcseTrainDataset(data=args.train_data, args=args, tokenizer=tokenizer)
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              num_workers=args.num_workers,
                              shuffle=True,
                              pin_memory=False)

    eval_dataset = SimcseEvalDataset(eval_mode='eval', args=args, tokenizer=tokenizer)
    eval_loader = DataLoader(eval_dataset,
                             batch_size=args.batch_size,
                             num_workers=args.num_workers,
                             shuffle=True,
                             pin_memory=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    test_dataset = SimcseEvalDataset(eval_mode='test', args=args, tokenizer=tokenizer)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=args.batch_size,
                                 num_workers=args.num_workers,
                                 shuffle=True,
                                 pin_memory=False)

    with torch.cuda.device(args.gpu_index):
        simcse = SimCSE(args=args, model=model, optimizer=optimizer)
        simcse.train(train_loader=train_loader, eval_loader=eval_loader)

        model.load_state_dict(torch.load(os.path.join(args.output_path, 'simcse.pt')))
        model.eval()
        corrcoef = simcse.evaluate(model=model, dataloader=test_dataloader)
        print('corrcoef: {}'.format(corrcoef))

        # Todo: add test


if __name__ == '__main__':
    args = parser.parse_args()
    seed_everything(args.seed)
    main()