from typing import List
from torch.utils.data import Dataset

class TrainDataset(Dataset):
    '''
    Training dataset
    '''
    def __init__(self, data, *args, **kwargs):
        self.args = kwargs['args']
        self.tokenizer = kwargs['tokenizer']
        self.data = data
        self.column_names = None

    def __len__(self):
        return len(self.data['train'])

    def text_2_id(self, text: str):
        if len(self.column_names) == 3:
            return self.tokenizer([text[self.column_names[0]], text[self.column_names[1]], text[self.column_names[2]]],
                                  max_length=self.args.max_len,
                                  truncation=True,
                                  padding='max_length',
                                  return_tensors='pt')
        elif len(self.column_names) == 2:
            return self.tokenizer([text[self.column_names[0]], text[self.column_names[1]]],
                                  max_length=self.args.max_len,
                                  truncation=True,
                                  padding='max_length',
                                  return_tensors='pt')
        elif len(self.column_names) == 1:
            return self.tokenizer([text[self.column_names[0]], text[self.column_names[0]]],
                                  max_length=self.args.max_len,
                                  truncation=True,
                                  padding='max_length',
                                  return_tensors='pt')
        else:
            raise ValueError("Mismatch in input dimension.")
    def __getitem__(self, index: int):
        self.column_names = self.data["train"].column_names
        return self.text_2_id(self.data['train'][index])

