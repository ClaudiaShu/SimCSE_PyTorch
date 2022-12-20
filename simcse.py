import logging
from loguru import logger
import os

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from scipy.stats import spearmanr
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


from utils import save_config_file, accuracy, save_checkpoint

'''
refer: https://github.com/vdogmcgee/SimCSE-Chinese-Pytorch/blob/main/simcse_unsup.py
'''

class SimCSE(object):
    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']

        if self.args.do_train:
            os.makedirs(self.args.output_path, exist_ok=True)
            log_dir = self.args.output_path
            if not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=log_dir)

            logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)
        else:
            pass



    def simcse_unsup_loss(self, y_pred: 'tensor') -> 'tensor':
        """
        loss function for self-supervised training
        y_pred (tensor): bert output, [batch_size * 2, 768]

        """
        # Get the label for each prediction, [1, 0, 3, 2, ..., batch_size-1, batch_size-2]
        y_true = torch.arange(y_pred.shape[0], device=self.args.device)
        labels = (y_true - y_true % 2 * 2) + 1
        # Calculate the similarity matrix
        sim = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=-1)
        # Discard the diagnosis (we don't want the self-similarity)
        sim = sim - torch.eye(y_pred.shape[0], device=self.args.device) * 1e12
        # similarity divide by temperature
        logits = sim / self.args.temperature

        return logits, labels

    def simcse_sup_loss(self, y_pred: 'tensor') -> 'tensor':
        """
        loss function for supervised training
        y_pred (tensor): bert output, [batch_size * 3, 768]
        """
        # Get the label for each prediction
        # Note that there is no label for every third sentence, skip that, label= [1, 0, 4, 3, ...]
        y_true = torch.arange(y_pred.shape[0], device=self.args.device)
        use_row = torch.where((y_true + 1) % 3 != 0)[0]
        labels = (use_row - use_row % 3 * 2) + 1
        # Calculate the similarity matrix
        sim = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=-1)
        # Discard the diagnosis (we don't want the self-similarity)
        sim = sim - torch.eye(y_pred.shape[0], device=self.args.device) * 1e12
        # Choose the valid rows
        sim = torch.index_select(sim, 0, use_row)
        # similarity divide by temperature
        logits = sim / self.args.temperature

        return logits, labels

    def train(self, train_loader, eval_loader):
        logger.info("start training")
        self.model.train()
        device = self.args.device
        best = 0
        scaler = GradScaler(enabled=self.args.fp16_precision)
        save_config_file(self.writer.log_dir, self.args)

        logging.info(f"Start SimCSE training for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {self.args.disable_cuda}.")

        lr = []
        loss = 0

        for epoch in range(self.args.epochs):
            print(epoch)
            pbar = tqdm(train_loader)
            for batch_idx, data in enumerate(pbar):
                step = epoch * len(train_loader) + batch_idx
                pbar.set_description(f"step: {step} | Loss: {loss}")
                # [batch, n, seq_len] -> [batch * n, sql_len]
                if self.args.arch == 'roberta':
                    sql_len = data['input_ids'].shape[-1]
                    input_ids = data['input_ids'].view(-1, sql_len).to(device)
                    attention_mask = data['attention_mask'].view(-1, sql_len).to(device)
                    token_type_ids = None
                elif self.args.arch == 'bert':
                    sql_len = data['input_ids'].shape[-1]
                    input_ids = data['input_ids'].view(-1, sql_len).to(device)
                    attention_mask = data['attention_mask'].view(-1, sql_len).to(device)
                    token_type_ids = data['token_type_ids'].view(-1, sql_len).to(device)
                else:
                    raise ValueError("Unsupported pretrained model")

                out = self.model(input_ids, attention_mask, token_type_ids)
                if self.args.mode == 'unsup':
                    logits, labels = self.simcse_unsup_loss(out)
                else:
                    logits, labels = self.simcse_sup_loss(out)
                loss = F.cross_entropy(logits, labels)
                self.optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                step += 1

                if step % self.args.log_every_n_steps == 0:
                    top1, top5 = accuracy(logits, labels, topk=(1, 5))
                    corrcoef = self.evaluate(model=self.model, dataloader=eval_loader)
                    logger.info('loss:{}, corrcoef: {}'.format(loss, corrcoef))
                    logging.debug(f"Epoch: {epoch}\tLoss: {loss}\tcorrcoef: {corrcoef}\tTop1 accuracy: {top1[0]}")
                    if best < corrcoef:
                        best = corrcoef
                        if self.args.save_data:
                            torch.save(self.model.state_dict(), os.path.join(self.args.output_path, 'simcse.pt'))
                            checkpoint_name = 'checkpoint_best'
                            self.model.model.save_pretrained(os.path.join(self.writer.log_dir, checkpoint_name))
                        logger.info('higher corrcoef: {}'.format(best))

                if step >= 100:
                    self.scheduler.step(step)
                lr.append(self.scheduler.get_lr()[0])

            lr_x = np.array(lr)

            plt.plot(lr_x)
            plt.show()

    def evaluate(self, model, dataloader):
        model.eval()
        sim_tensor = torch.tensor([], device=self.args.device)
        label_array = np.array([])
        with torch.no_grad():
            for source, target, label in dataloader:
                # source        [batch, 1, seq_len] -> [batch, seq_len]
                source_input_ids = source.get('input_ids').squeeze(1).to(self.args.device)
                source_attention_mask = source.get('attention_mask').squeeze(1).to(self.args.device)
                if self.args.arch == 'roberta':
                    source_pred = model(source_input_ids, source_attention_mask)
                elif self.args.arch == 'bert':
                    source_token_type_ids = source.get('token_type_ids').squeeze(1).to(self.args.device)
                    source_pred = model(source_input_ids, source_attention_mask, source_token_type_ids)
                else:
                    raise ValueError
                # target        [batch, 1, seq_len] -> [batch, seq_len]
                target_input_ids = target.get('input_ids').squeeze(1).to(self.args.device)
                target_attention_mask = target.get('attention_mask').squeeze(1).to(self.args.device)
                if self.args.arch == 'roberta':
                    target_pred = model(source_input_ids, source_attention_mask)
                elif self.args.arch == 'bert':
                    target_token_type_ids = target.get('token_type_ids').squeeze(1).to(self.args.device)
                    target_pred = model(target_input_ids, target_attention_mask, target_token_type_ids)
                else:
                    raise ValueError

                # concat
                sim = F.cosine_similarity(source_pred, target_pred, dim=-1)
                sim_tensor = torch.cat((sim_tensor, sim), dim=0)
                label_array = np.append(label_array, np.array(label))
        # corrcoef
        return spearmanr(label_array, sim_tensor.cpu().numpy()).correlation