import os
import argparse
import time
import torch
import torch.nn as nn
from prettytable import PrettyTable
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
parser.add_argument('-batch_size_eval', default=128, type=int)
parser.add_argument('-num_epochs', default=5, type=int)
parser.add_argument('-learning_rate', default=2e-5, type=float)
parser.add_argument('-max_grad_norm', default=1.0, type=float)
parser.add_argument('-warm_up_proportion', default=0.1, type=float)
parser.add_argument('-gradient_accumulation_step', default=1, type=int)
parser.add_argument('-bert_model_path', default='bert-base-uncased', type=str)  # 'bert-base-uncased'
parser.add_argument('-bert_tokenizer_path', default='bert-base-uncased', type=str)  # 'bert-base-uncased'
parser.add_argument('-report_step', default=100, type=int)
args = parser.parse_args()

# Change the evaluation model here
# args.bert_model_path = "/media/storage/yuxuan/software/SimCSE/results/unsup/ri005/checkpoint_best/"

def print_table(task_names, scores):
    tb = PrettyTable()
    tb.field_names = task_names
    tb.add_row(scores)
    print(tb)

def load_data(path, dataset, tokenizer):
    input_file = open(path, encoding='utf-8')
    if dataset in ["MRPC", "QQP", "SST-2", "QNLI", "RTE", "WNLI", "STS-B", "MNLI"]:
        lines = input_file.readlines()[1:]
    elif dataset in ["CoLA"]:
        lines = input_file.readlines()
    else:
        assert False
    input_file.close()
    input_ids, attention_mask, token_type_ids = [], [], []
    labels = []

    pbar = tqdm(lines)
    for line in pbar:
        pbar.set_description(f"Getting {dataset}")
        line_split = line.strip().split("\t")
        if dataset == "MRPC":
            assert len(line_split) == 5
            ans = tokenizer.encode_plus(line_split[3], line_split[4], max_length=args.max_seq_length,
                                        padding="max_length", truncation="longest_first")
        elif dataset == "QQP":
            assert len(line_split) == 6
            ans = tokenizer.encode_plus(line_split[3], line_split[4], max_length=args.max_seq_length,
                                        padding="max_length", truncation="longest_first")
        elif dataset == "SST-2":
            assert len(line_split) == 2
            ans = tokenizer.encode_plus(line_split[0], max_length=args.max_seq_length,
                                        padding="max_length", truncation=True)
        elif dataset == "QNLI":
            assert len(line_split) == 4
            ans = tokenizer.encode_plus(line_split[1], line_split[2], max_length=args.max_seq_length,
                                        padding="max_length", truncation="longest_first")
        elif dataset == "RTE":
            assert len(line_split) == 4
            ans = tokenizer.encode_plus(line_split[1], line_split[2], max_length=args.max_seq_length,
                                        padding="max_length", truncation="longest_first")
        elif dataset == "WNLI":
            assert len(line_split) == 4
            ans = tokenizer.encode_plus(line_split[1], line_split[2], max_length=args.max_seq_length,
                                        padding="max_length", truncation="longest_first")
        elif dataset == "CoLA":
            assert len(line_split) == 4
            ans = tokenizer.encode_plus(line_split[3], max_length=args.max_seq_length,
                                        padding="max_length", truncation=True)
        elif dataset == "STS-B":
            assert len(line_split) == 10
            ans = tokenizer.encode_plus(line_split[7], line_split[8], max_length=args.max_seq_length,
                                        padding="max_length", truncation="longest_first")
        elif dataset == "MNLI":
            ans = tokenizer.encode_plus(line_split[8], line_split[9], max_length=args.max_seq_length,
                                        padding="max_length", truncation="longest_first")
        else:
            assert False
        input_ids.append(ans.input_ids)
        attention_mask.append(ans.attention_mask)
        token_type_ids.append(ans.token_type_ids)
        if dataset == "MRPC":
            labels.append(int(line_split[0]))
        elif dataset == "QQP":
            labels.append(int(line_split[5]))
        elif dataset == "SST-2":
            labels.append(int(line_split[1]))
        elif dataset == "QNLI":
            if line_split[3] == "not_entailment":
                labels.append(0)
            elif line_split[3] == "entailment":
                labels.append(1)
            else:
                assert False
        elif dataset == "RTE":
            if line_split[3] == "not_entailment":
                labels.append(0)
            elif line_split[3] == "entailment":
                labels.append(1)
            else:
                assert False
        elif dataset == "WNLI":
            labels.append(int(line_split[3]))
        elif dataset == "CoLA":
            labels.append(int(line_split[1]))
        elif dataset == "STS-B":
            labels.append(float(line_split[9]))
        elif dataset == "MNLI":
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

def train(dataset):
    if dataset in ["MRPC", "RTE", "WNLI", "STS-B"]:
        args.report_step = 10

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    tokenizer = BertTokenizer.from_pretrained(args.bert_tokenizer_path)
    if dataset in ["MRPC", "QQP", "SST-2", "QNLI", "RTE", "WNLI", "CoLA"]:
        model = BertForSequenceClassification.from_pretrained(args.bert_model_path, num_labels=2)
    elif dataset in ["STS-B"]:
        model = BertForSequenceClassification.from_pretrained(args.bert_model_path, num_labels=1)
    elif dataset in ["MNLI"]:
        model = BertForSequenceClassification.from_pretrained(args.bert_model_path, num_labels=3)
    else:
        assert False
    model = torch.nn.DataParallel(model)
    model.to(device)

    if dataset == "MRPC":
        train_input_ids, train_attention_mask, train_token_type_ids, y_train = load_data(
            path="../data/glue_data/MRPC/train.tsv", dataset=dataset, tokenizer=tokenizer)
        dev_input_ids, dev_attention_mask, dev_token_type_ids, y_dev = load_data(
            path="../data/glue_data/MRPC/dev.tsv", dataset=dataset, tokenizer=tokenizer)
    elif dataset == "QQP":
        train_input_ids, train_attention_mask, train_token_type_ids, y_train = load_data(
            path="../data/glue_data/QQP/train.tsv", dataset=dataset, tokenizer=tokenizer)
        dev_input_ids, dev_attention_mask, dev_token_type_ids, y_dev = load_data(
            path="../data/glue_data/QQP/dev.tsv", dataset=dataset, tokenizer=tokenizer)
    elif dataset == "SST-2":
        train_input_ids, train_attention_mask, train_token_type_ids, y_train = load_data(
            path="../data/glue_data/SST-2/train.tsv", dataset=dataset, tokenizer=tokenizer)
        dev_input_ids, dev_attention_mask, dev_token_type_ids, y_dev = load_data(
            path="../data/glue_data/SST-2/dev.tsv", dataset=dataset, tokenizer=tokenizer)
    elif dataset == "QNLI":
        train_input_ids, train_attention_mask, train_token_type_ids, y_train = load_data(
            path="../data/glue_data/QNLI/train.tsv", dataset=dataset, tokenizer=tokenizer)
        dev_input_ids, dev_attention_mask, dev_token_type_ids, y_dev = load_data(
            path="../data/glue_data/QNLI/dev.tsv", dataset=dataset, tokenizer=tokenizer)
    elif dataset == "RTE":
        train_input_ids, train_attention_mask, train_token_type_ids, y_train = load_data(
            path="../data/glue_data/RTE/train.tsv", dataset=dataset, tokenizer=tokenizer)
        dev_input_ids, dev_attention_mask, dev_token_type_ids, y_dev = load_data(
            path="../data/glue_data/RTE/dev.tsv", dataset=dataset, tokenizer=tokenizer)
    elif dataset == "WNLI":
        train_input_ids, train_attention_mask, train_token_type_ids, y_train = load_data(
            path="../data/glue_data/WNLI/train.tsv", dataset=dataset, tokenizer=tokenizer)
        dev_input_ids, dev_attention_mask, dev_token_type_ids, y_dev = load_data(
            path="../data/glue_data/WNLI/dev.tsv", dataset=dataset, tokenizer=tokenizer)
    elif dataset == "CoLA":
        train_input_ids, train_attention_mask, train_token_type_ids, y_train = load_data(
            path="../data/glue_data/CoLA/train.tsv", dataset=dataset, tokenizer=tokenizer)
        dev_input_ids, dev_attention_mask, dev_token_type_ids, y_dev = load_data(
            path="../data/glue_data/CoLA/dev.tsv", dataset=dataset, tokenizer=tokenizer)
    elif dataset == "STS-B":
        train_input_ids, train_attention_mask, train_token_type_ids, y_train = load_data(
            path="../data/glue_data/STS-B/train.tsv", dataset=dataset, tokenizer=tokenizer)
        dev_input_ids, dev_attention_mask, dev_token_type_ids, y_dev = load_data(
            path="../data/glue_data/STS-B/dev.tsv", dataset=dataset, tokenizer=tokenizer)
    elif dataset == "MNLI":
        train_input_ids, train_attention_mask, train_token_type_ids, y_train = load_data(
            path="../data/glue_data/MNLI/train.tsv", dataset=dataset, tokenizer=tokenizer)
        dev_input_ids, dev_attention_mask, dev_token_type_ids, y_dev = load_data(
            path="../data/glue_data/MNLI/dev_matched.tsv", dataset=dataset, tokenizer=tokenizer)
        extra_dev_input_ids, extra_dev_attention_mask, extra_dev_token_type_ids, extra_y_dev = load_data(
            path="../data/glue_data/MNLI/dev_mismatched.tsv", dataset=dataset, tokenizer=tokenizer)
    else:
        assert False

    train_input_ids = torch.tensor(train_input_ids, dtype=torch.long)
    train_attention_mask = torch.tensor(train_attention_mask, dtype=torch.float)
    train_token_type_ids = torch.tensor(train_token_type_ids, dtype=torch.long)
    if dataset in ["STS-B"]:
        y_train = torch.tensor(y_train, dtype=torch.float)
    else:
        y_train = torch.tensor(y_train, dtype=torch.long)
    dev_input_ids = torch.tensor(dev_input_ids, dtype=torch.long)
    dev_attention_mask = torch.tensor(dev_attention_mask, dtype=torch.float)
    dev_token_type_ids = torch.tensor(dev_token_type_ids, dtype=torch.long)
    if dataset in ["STS-B"]:
        y_dev = torch.tensor(y_dev, dtype=torch.float)
    else:
        y_dev = torch.tensor(y_dev, dtype=torch.long)
    train_data = TensorDataset(train_input_ids, train_attention_mask, train_token_type_ids, y_train)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    dev_data = TensorDataset(dev_input_ids, dev_attention_mask, dev_token_type_ids, y_dev)
    dev_loader = DataLoader(dev_data, batch_size=args.batch_size_eval, shuffle=False)
    if dataset in ["MNLI"]:
        extra_dev_input_ids = torch.tensor(extra_dev_input_ids, dtype=torch.long)
        extra_dev_attention_mask = torch.tensor(extra_dev_attention_mask, dtype=torch.float)
        extra_dev_token_type_ids = torch.tensor(extra_dev_token_type_ids, dtype=torch.long)
        extra_y_dev = torch.tensor(extra_y_dev, dtype=torch.long)
        extra_dev_data = TensorDataset(extra_dev_input_ids, extra_dev_attention_mask, extra_dev_token_type_ids, extra_y_dev)
        extra_dev_loader = DataLoader(extra_dev_data, batch_size=args.batch_size_eval, shuffle=False)

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
    best_accuracy, best_mism_accuracy, best_f1, best_matthews, best_pearsonr, best_spearmanr = 0, 0, 0, 0, 0, 0

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
            if dataset in ["STS-B"]:
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
                if dataset in ["STS-B"]:
                    preds.extend(list(outputs[0].view(-1).cpu().numpy()))
                else:
                    preds.extend(list(torch.max(outputs[0], 1)[1].cpu().numpy()))
            if dataset in ["MRPC", "QQP", "SST-2", "QNLI", "RTE", "WNLI"]:
                cur_accuracy = accuracy_score(np.array(y_dev), np.array(preds))
                print("accuracy: {:.4f}".format(cur_accuracy))
                if best_accuracy < cur_accuracy:
                    best_accuracy = cur_accuracy
            if dataset in ["MRPC", "QQP"]:
                cur_f1 = f1_score(np.array(y_dev), np.array(preds))
                print("f1: {:.4f}".format(cur_f1))
                if best_f1 < cur_f1:
                    best_f1 = cur_f1
            if dataset in ["CoLA"]:
                cur_matthews = matthews_corrcoef(np.array(y_dev), np.array(preds))
                print("matthews corrcoef: {:.4f}".format(cur_matthews))
                if best_matthews < cur_matthews:
                    best_matthews = cur_matthews
            if dataset in ["STS-B"]:
                preds = np.clip(np.array(preds), 0, 5)
                cur_pearsonr = pearsonr(np.array(y_dev), preds)[0]
                cur_spearmanr = spearmanr(np.array(y_dev), preds)[0]
                print("pearson corrcoef: {:.4f}".format(cur_pearsonr))
                print("spearman corrcoef: {:.4f}".format(cur_spearmanr))
                if best_pearsonr < cur_pearsonr:
                    best_pearsonr = cur_pearsonr
                if best_spearmanr < cur_spearmanr:
                    best_spearmanr = cur_spearmanr
            if dataset in ["MNLI"]:
                cur_accuracy = accuracy_score(np.array(y_dev), np.array(preds))
                print("matched accuracy: {:.4f}".format(cur_accuracy))
                if best_accuracy < cur_accuracy:
                    best_accuracy = cur_accuracy
                preds = []
                for cur_input_ids, cur_attention_mask, cur_token_type_ids, cur_y in tqdm(extra_dev_loader):
                    cur_input_ids = cur_input_ids.to(device)
                    cur_attention_mask = cur_attention_mask.to(device)
                    cur_token_type_ids = cur_token_type_ids.to(device)
                    cur_y = cur_y.to(device)
                    outputs = model(cur_input_ids, cur_attention_mask, cur_token_type_ids)
                    preds.extend(list(torch.max(outputs[0], 1)[1].cpu().numpy()))
                cur_mism_accuracy = accuracy_score(np.array(extra_y_dev), np.array(preds))
                print("mismatched accuracy: {:.4f}".format(cur_mism_accuracy))
                if best_mism_accuracy < cur_mism_accuracy:
                    best_mism_accuracy = cur_mism_accuracy

    if dataset in ["MRPC", "QQP"]:
        result = {
            "metric1": "accuracy",
            "score1": best_accuracy,
            "metric2": "f1 score",
            "score2": best_f1
        }
        return result
    elif dataset in ["SST-2", "QNLI", "RTE", "WNLI"]:
        result = {
            "metric": "accuracy",
            "score": best_accuracy,
        }
        return result
    elif dataset in ["CoLA"]:
        result = {
            "metric": "matthews corrcoef",
            "score": best_matthews,
        }
        return result
    elif dataset in ["STS-B"]:
        result = {
            "metric1": "pearson corrcoef",
            "score1": best_pearsonr,
            "metric2": "spearman corrcoef",
            "score2": best_spearmanr
        }
        return result
    elif dataset in ["MNLI"]:
        result = {
            "metric1": "matched accuracy",
            "score1": best_accuracy,
            "metric2": "mismatched accuracy",
            "score2": best_mism_accuracy
        }
        return result
    print("training time: {:.4f}".format(time.time() - start_time))


def main():
    tasks = ["WNLI", "MRPC", "CoLA", "RTE", "STS-B"]
    # tasks = ["WNLI", "MRPC", "CoLA", "RTE", "STS-B"]  # "MRPC", "QQP", "SST-2", "QNLI", "RTE", "WNLI", "CoLA", "STS-B", "MNLI"
    task_names = []
    scores = []
    for task in tasks:
        result = train(task)
        if task in ["SST-2", "QNLI", "RTE", "WNLI", "CoLA"]:
            task_names.append(task+" - "+result["metric"])
            try:
                scores.append(result["score"])
            except:
                pass
        elif task in ["MRPC", "QQP", "STS-B", "MNLI"]:
            task_names.append(task+" - "+result["metric1"])
            scores.append(result["score1"])

            task_names.append(task+" - "+result["metric2"])
            scores.append(result["score2"])
    print_table(task_names, scores)



if __name__ == '__main__':
    main()