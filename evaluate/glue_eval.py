import os
import argparse
import time
import torch
import torch.nn as nn
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from scipy.stats import pearsonr, spearmanr
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('-seed', default=0, type=int)
parser.add_argument('-max_seq_length', default=128, type=int)
parser.add_argument('-batch_size', default=32, type=int)
parser.add_argument('-num_epochs', default=3, type=int)
parser.add_argument('-learning_rate', default=2e-5, type=float)
parser.add_argument('-max_grad_norm', default=1.0, type=float)
parser.add_argument('-warm_up_proportion', default=0.1, type=float)
parser.add_argument('-gradient_accumulation_step', default=1, type=int)
parser.add_argument('-bert_path', default='bert-base-uncased', type=str)
parser.add_argument('-dataset', default='SST-2', type=str, choices=["MRPC", "QQP", "SST-2", "QNLI", "RTE", "WNLI", "CoLA", "STS-B", "MNLI"])
parser.add_argument('-report_step', default=100, type=int)
args = parser.parse_args()

if args.dataset in ["MRPC", "RTE", "WNLI", "STS-B"]:
    args.report_step = 10


random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

tokenizer = BertTokenizer.from_pretrained(args.bert_path)
if args.dataset in ["MRPC", "QQP", "SST-2", "QNLI", "RTE", "WNLI", "CoLA"]:
    model = BertForSequenceClassification.from_pretrained(args.bert_path, num_labels=2)
elif args.dataset in ["STS-B"]:
    model = BertForSequenceClassification.from_pretrained(args.bert_path, num_labels=1)
elif args.dataset in ["MNLI"]:
    model = BertForSequenceClassification.from_pretrained(args.bert_path, num_labels=3)
else:
    assert False
model = torch.nn.DataParallel(model)
model.to(device)

def load_data(path):
    input_file = open(path, encoding='utf-8')
    if args.dataset in ["MRPC", "QQP", "SST-2", "QNLI", "RTE", "WNLI", "STS-B", "MNLI"]:
        lines = input_file.readlines()[1:]
    elif args.dataset in ["CoLA"]:
        lines = input_file.readlines()
    else:
        assert False
    input_file.close()
    input_ids, attention_mask, token_type_ids = [], [], []
    labels = []
    for line in tqdm(lines):
        line_split = line.strip().split("\t")
        if args.dataset == "MRPC":
            assert len(line_split) == 5
            ans = tokenizer.encode_plus(line_split[3], line_split[4], max_length=args.max_seq_length,
                                        padding="max_length", truncation="longest_first")
        elif args.dataset == "QQP":
            assert len(line_split) == 6
            ans = tokenizer.encode_plus(line_split[3], line_split[4], max_length=args.max_seq_length,
                                        padding="max_length", truncation="longest_first")
        elif args.dataset == "SST-2":
            assert len(line_split) == 2
            ans = tokenizer.encode_plus(line_split[0], max_length=args.max_seq_length,
                                        padding="max_length", truncation=True)
        elif args.dataset == "QNLI":
            assert len(line_split) == 4
            ans = tokenizer.encode_plus(line_split[1], line_split[2], max_length=args.max_seq_length,
                                        padding="max_length", truncation="longest_first")
        elif args.dataset == "RTE":
            assert len(line_split) == 4
            ans = tokenizer.encode_plus(line_split[1], line_split[2], max_length=args.max_seq_length,
                                        padding="max_length", truncation="longest_first")
        elif args.dataset == "WNLI":
            assert len(line_split) == 4
            ans = tokenizer.encode_plus(line_split[1], line_split[2], max_length=args.max_seq_length,
                                        padding="max_length", truncation="longest_first")
        elif args.dataset == "CoLA":
            assert len(line_split) == 4
            ans = tokenizer.encode_plus(line_split[3], max_length=args.max_seq_length,
                                        padding="max_length", truncation=True)
        elif args.dataset == "STS-B":
            assert len(line_split) == 10
            ans = tokenizer.encode_plus(line_split[7], line_split[8], max_length=args.max_seq_length,
                                        padding="max_length", truncation="longest_first")
        elif args.dataset == "MNLI":
            ans = tokenizer.encode_plus(line_split[8], line_split[9], max_length=args.max_seq_length,
                                        padding="max_length", truncation="longest_first")
        else:
            assert False
        input_ids.append(ans.input_ids)
        attention_mask.append(ans.attention_mask)
        token_type_ids.append(ans.token_type_ids)
        if args.dataset == "MRPC":
            labels.append(int(line_split[0]))
        elif args.dataset == "QQP":
            labels.append(int(line_split[5]))
        elif args.dataset == "SST-2":
            labels.append(int(line_split[1]))
        elif args.dataset == "QNLI":
            if line_split[3] == "not_entailment":
                labels.append(0)
            elif line_split[3] == "entailment":
                labels.append(1)
            else:
                assert False
        elif args.dataset == "RTE":
            if line_split[3] == "not_entailment":
                labels.append(0)
            elif line_split[3] == "entailment":
                labels.append(1)
            else:
                assert False
        elif args.dataset == "WNLI":
            labels.append(int(line_split[3]))
        elif args.dataset == "CoLA":
            labels.append(int(line_split[1]))
        elif args.dataset == "STS-B":
            labels.append(float(line_split[9]))
        elif args.dataset == "MNLI":
            if line_split[-1] == "contradiction":
                labels.append(0)
            elif line_split[-1] == "entailment":
                labels.append(1)
            elif line_split[-1] == "neutral":
                labels.append(2)
            else:
                assert False
        else:
            assert False
    return np.array(input_ids), np.array(attention_mask), np.array(token_type_ids), np.array(labels)

if args.dataset == "MRPC":
    train_input_ids, train_attention_mask, train_token_type_ids, y_train = load_data("../data/glue_data/MRPC/train.tsv")
    dev_input_ids, dev_attention_mask, dev_token_type_ids, y_dev = load_data("../data/glue_data/MRPC/dev.tsv")
elif args.dataset == "QQP":
    train_input_ids, train_attention_mask, train_token_type_ids, y_train = load_data("../data/glue_data/QQP/train.tsv")
    dev_input_ids, dev_attention_mask, dev_token_type_ids, y_dev = load_data("../data/glue_data/QQP/dev.tsv")
elif args.dataset == "SST-2":
    train_input_ids, train_attention_mask, train_token_type_ids, y_train = load_data("../data/glue_data/SST-2/train.tsv")
    dev_input_ids, dev_attention_mask, dev_token_type_ids, y_dev = load_data("../data/glue_data/SST-2/dev.tsv")
elif args.dataset == "QNLI":
    train_input_ids, train_attention_mask, train_token_type_ids, y_train = load_data("../data/glue_data/QNLI/train.tsv")
    dev_input_ids, dev_attention_mask, dev_token_type_ids, y_dev = load_data("../data/glue_data/QNLI/dev.tsv")
elif args.dataset == "RTE":
    train_input_ids, train_attention_mask, train_token_type_ids, y_train = load_data("../data/glue_data/RTE/train.tsv")
    dev_input_ids, dev_attention_mask, dev_token_type_ids, y_dev = load_data("../data/glue_data/RTE/dev.tsv")
elif args.dataset == "WNLI":
    train_input_ids, train_attention_mask, train_token_type_ids, y_train = load_data("../data/glue_data/WNLI/train.tsv")
    dev_input_ids, dev_attention_mask, dev_token_type_ids, y_dev = load_data("../data/glue_data/WNLI/dev.tsv")
elif args.dataset == "CoLA":
    train_input_ids, train_attention_mask, train_token_type_ids, y_train = load_data("../data/glue_data/CoLA/train.tsv")
    dev_input_ids, dev_attention_mask, dev_token_type_ids, y_dev = load_data("../data/glue_data/CoLA/dev.tsv")
elif args.dataset == "STS-B":
    train_input_ids, train_attention_mask, train_token_type_ids, y_train = load_data("../data/glue_data/STS-B/train.tsv")
    dev_input_ids, dev_attention_mask, dev_token_type_ids, y_dev = load_data("../data/glue_data/STS-B/dev.tsv")
elif args.dataset == "MNLI":
    train_input_ids, train_attention_mask, train_token_type_ids, y_train = load_data("../data/glue_data/MNLI/train.tsv")
    dev_input_ids, dev_attention_mask, dev_token_type_ids, y_dev = load_data("../data/glue_data/MNLI/dev_matched.tsv")
    extra_dev_input_ids, extra_dev_attention_mask, extra_dev_token_type_ids, extra_y_dev = load_data("../data/glue_data/MNLI/dev_mismatched.tsv")
else:
    assert False

train_input_ids = torch.tensor(train_input_ids, dtype=torch.long)
train_attention_mask = torch.tensor(train_attention_mask, dtype=torch.float)
train_token_type_ids = torch.tensor(train_token_type_ids, dtype=torch.long)
if args.dataset in ["STS-B"]:
    y_train = torch.tensor(y_train, dtype=torch.float)
else:
    y_train = torch.tensor(y_train, dtype=torch.long)
dev_input_ids = torch.tensor(dev_input_ids, dtype=torch.long)
dev_attention_mask = torch.tensor(dev_attention_mask, dtype=torch.float)
dev_token_type_ids = torch.tensor(dev_token_type_ids, dtype=torch.long)
if args.dataset in ["STS-B"]:
    y_dev = torch.tensor(y_dev, dtype=torch.float)
else:
    y_dev = torch.tensor(y_dev, dtype=torch.long)
train_data = TensorDataset(train_input_ids, train_attention_mask, train_token_type_ids, y_train)
train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
dev_data = TensorDataset(dev_input_ids, dev_attention_mask, dev_token_type_ids, y_dev)
dev_loader = DataLoader(dev_data, batch_size=args.batch_size, shuffle=False)
if args.dataset in ["MNLI"]:
    extra_dev_input_ids = torch.tensor(extra_dev_input_ids, dtype=torch.long)
    extra_dev_attention_mask = torch.tensor(extra_dev_attention_mask, dtype=torch.float)
    extra_dev_token_type_ids = torch.tensor(extra_dev_token_type_ids, dtype=torch.long)
    extra_y_dev = torch.tensor(extra_y_dev, dtype=torch.long)
    extra_dev_data = TensorDataset(extra_dev_input_ids, extra_dev_attention_mask, extra_dev_token_type_ids, extra_y_dev)
    extra_dev_loader = DataLoader(extra_dev_data, batch_size=args.batch_size, shuffle=False)

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, correct_bias=False)
scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=len(train_loader) * args.num_epochs * args.warm_up_proportion // args.gradient_accumulation_step,
                num_training_steps=len(train_loader) * args.num_epochs // args.gradient_accumulation_step)
total_step = len(train_loader)
start_time = time.time()
for epoch in range(args.num_epochs):
    model.train()
    model.zero_grad()
    pbar = tqdm(train_loader)
    for i, (cur_input_ids, cur_attention_mask, cur_token_type_ids, cur_y) in enumerate(pbar):
        cur_input_ids = cur_input_ids.to(device)
        cur_attention_mask = cur_attention_mask.to(device)
        cur_token_type_ids = cur_token_type_ids.to(device)
        cur_y = cur_y.to(device)
        outputs = model(cur_input_ids, cur_attention_mask, cur_token_type_ids)
        if args.dataset in ["STS-B"]:
            loss = nn.MSELoss()(outputs[0].view(-1), cur_y)
        else:
            loss = nn.CrossEntropyLoss()(outputs[0], cur_y)
        loss /= args.gradient_accumulation_step
        loss.backward()
        if (i + 1) % args.gradient_accumulation_step == 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            model.zero_grad()
        pbar.set_description(f"Epoch: {epoch + 1}/{args.num_epochs} | Step: {i + 1}/{total_step} | Loss: {loss.item()}")

    model.eval()
    with torch.no_grad():
        preds = []
        for cur_input_ids, cur_attention_mask, cur_token_type_ids, cur_y in tqdm(dev_loader):
            cur_input_ids = cur_input_ids.to(device)
            cur_attention_mask = cur_attention_mask.to(device)
            cur_token_type_ids = cur_token_type_ids.to(device)
            cur_y = cur_y.to(device)
            outputs = model(cur_input_ids, cur_attention_mask, cur_token_type_ids)
            if args.dataset in ["STS-B"]:
                preds.extend(list(outputs[0].view(-1).cpu().numpy()))
            else:
                preds.extend(list(torch.max(outputs[0], 1)[1].cpu().numpy()))
        if args.dataset in ["MRPC", "QQP", "SST-2", "QNLI", "RTE", "WNLI"]:
            cur_accuracy = accuracy_score(np.array(y_dev), np.array(preds))
            print("accuracy: {:.4f}".format(cur_accuracy))
        if args.dataset in ["MRPC", "QQP"]:
            cur_f1 = f1_score(np.array(y_dev), np.array(preds))
            print("f1: {:.4f}".format(cur_f1))
        if args.dataset in ["CoLA"]:
            cur_matthews = matthews_corrcoef(np.array(y_dev), np.array(preds))
            print("matthews corrcoef: {:.4f}".format(cur_matthews))
        if args.dataset in ["STS-B"]:
            preds = np.clip(np.array(preds), 0, 5)
            cur_pearsonr = pearsonr(np.array(y_dev), preds)[0]
            cur_spearmanr = spearmanr(np.array(y_dev), preds)[0]
            print("pearson corrcoef: {:.4f}".format(cur_pearsonr))
            print("spearman corrcoef: {:.4f}".format(cur_spearmanr))
        if args.dataset in ["MNLI"]:
            cur_accuracy = accuracy_score(np.array(y_dev), np.array(preds))
            print("matched accuracy: {:.4f}".format(cur_accuracy))
            preds = []
            for cur_input_ids, cur_attention_mask, cur_token_type_ids, cur_y in tqdm(extra_dev_loader):
                cur_input_ids = cur_input_ids.to(device)
                cur_attention_mask = cur_attention_mask.to(device)
                cur_token_type_ids = cur_token_type_ids.to(device)
                cur_y = cur_y.to(device)
                outputs = model(cur_input_ids, cur_attention_mask, cur_token_type_ids)
                preds.extend(list(torch.max(outputs[0], 1)[1].cpu().numpy()))
            cur_accuracy = accuracy_score(np.array(extra_y_dev), np.array(preds))
            print("mismatched accuracy: {:.4f}".format(cur_accuracy))
print("training time: {:.4f}".format(time.time() - start_time))