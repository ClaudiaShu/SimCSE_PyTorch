import os
from typing import List

import pickle
from datasets import load_dataset
from torch.utils.data import Dataset
from tqdm import tqdm


class SimcseTrainDataset(Dataset):
    '''
    Training dataset
    '''
    def __init__(self, data, *args, **kwargs):
        self.args = kwargs['args']
        self.tokenizer = kwargs['tokenizer']

        # Dataset
        # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
        # https://huggingface.co/docs/datasets/loading_datasets.html.
        data_files = {}
        if data is not None:
            data_files["train"] = data
        extension = data.split(".")[-1]
        if extension == "txt":
            extension = "text"
            datasets = load_dataset(extension, data_files=data_files)
        elif extension == "csv":
            datasets = load_dataset(extension, data_files=data_files,
                                    delimiter="\t" if "tsv" in data else ",")
        else:
            raise ValueError("Error with the type of file.")

        self.data = datasets
        self.column_names = self.data["train"].column_names

    def __len__(self):
        return 10000
        # return len(self.data['train'])

    def text_2_id(self, text: str):
        if len(self.column_names) == 3:
            return self.tokenizer([text[self.column_names[0]], text[self.column_names[1]], text[self.column_names[2]]],
                                  max_length=self.args.max_len,
                                  truncation=True,
                                  padding="max_length",
                                  return_tensors='pt')
        elif len(self.column_names) == 2:
            return self.tokenizer([text[self.column_names[0]], text[self.column_names[1]]],
                                  max_length=self.args.max_len,
                                  truncation=True,
                                  padding="max_length",
                                  return_tensors='pt')
        elif len(self.column_names) == 1:
            return self.tokenizer([text[self.column_names[0]], text[self.column_names[0]]],
                                  max_length=self.args.max_len,
                                  truncation=True,
                                  padding="max_length",
                                  return_tensors='pt')
        else:
            raise ValueError("Mismatch in input dimension.")

    def __getitem__(self, index: int):

        return self.text_2_id(self.data['train'][index])

class SimcseEvalDataset(Dataset):
    def __init__(self, eval_mode, *args, **kwargs):
        self.args = kwargs['args']
        self.mode = eval_mode
        self.tokenizer = kwargs['tokenizer']
        self.features = self.load_eval_data()

    def load_eval_data(self):
        """
        加载验证集或者测试集
        """
        assert self.mode in ['eval', 'test'], 'mode should in ["eval", "test"]'
        if self.mode == 'eval':
            eval_file = self.args.dev_file
        else:
            eval_file = self.args.test_file
        feature_list = []
        with open(eval_file, 'r', encoding='utf8') as f:
            lines = f.readlines()
            for line in tqdm(lines):
                line = line.strip().split("\t")
                assert len(line) == 7 or len(line) == 9
                score = float(line[4])
                data1 = self.tokenizer(line[5].strip(), max_length=self.args.max_len, truncation=True, padding='max_length',
                                  return_tensors='pt')
                data2 = self.tokenizer(line[6].strip(), max_length=self.args.max_len, truncation=True, padding='max_length',
                                  return_tensors='pt')

                feature_list.append((data1, data2, score))
        return feature_list

    def __len__(self):
        return len(self.features)

    def __getitem__(self, item):
        return self.features[item]


