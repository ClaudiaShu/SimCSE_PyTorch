import logging
import os
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import save_config_file, accuracy, save_checkpoint

torch.manual_seed(0)

class SimCSE(object):
    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.tokenizer = kwargs['tokenizer']

        import socket
        from datetime import datetime
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        log_dir = os.path.join('runs', current_time)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)

        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)
        # self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)

    def info_nce_loss(self, features):
        labels = torch.cat([torch.arange(self.args.batch_size) for i in range(self.args.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.args.device)

        # features = F.normalize(features, dim=1)
        fea_anchor = features[1:-3:3]
        fea_positive = features[2:-2:3]
        fea_negative = features[3:-1:3]

        similarity_matrix = torch.matmul(fea_anchor, fea_positive.T)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)

        logits = logits / self.args.temperature
        return logits, labels

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

    def model_forward(self, input_ids, attention_mask, token_type_ids):

        # out = self.bert(input_ids, attention_mask, token_type_ids)
        out = self.model(input_ids, attention_mask, token_type_ids, output_hidden_states=True)

        if self.args.pooling == 'cls':
            return out.last_hidden_state[:, 0]  # [batch, 768]

        if self.args.pooling == 'pooler':
            return out.pooler_output  # [batch, 768]

        if self.args.pooling == 'last-avg':
            last = out.last_hidden_state.transpose(1, 2)  # [batch, 768, seqlen]
            return torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]

        if self.args.pooling == 'first-last-avg':
            first = out.hidden_states[1].transpose(1, 2)  # [batch, 768, seqlen]
            last = out.hidden_states[-1].transpose(1, 2)  # [batch, 768, seqlen]
            first_avg = torch.avg_pool1d(first, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
            last_avg = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
            avg = torch.cat((first_avg.unsqueeze(1), last_avg.unsqueeze(1)), dim=1)  # [batch, 2, 768]
            return torch.avg_pool1d(avg.transpose(1, 2), kernel_size=2).squeeze(-1)  # [batch, 768]

    def train_sup(self, train_loader):
        n_iter = 0
        self.model.train()

        scaler = GradScaler(enabled=self.args.fp16_precision)

        # save config file
        save_config_file(self.writer.log_dir, self.args)

        logging.info(f"Start SimCSE training for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {self.args.disable_cuda}.")

        for epoch in range(self.args.epochs):
            for batch_idx, source in enumerate(tqdm(train_loader), start=1):

                real_batch_num = source.get('input_ids').shape[0]
                input_ids = source.get('input_ids').view(real_batch_num * 3, -1).to(self.args.device)
                attention_mask = source.get('attention_mask').view(real_batch_num * 3, -1).to(self.args.device)
                token_type_ids = source.get('token_type_ids').view(real_batch_num * 3, -1).to(self.args.device)

                with autocast(enabled=self.args.fp16_precision):
                    out = self.model_forward(input_ids, attention_mask, token_type_ids)
                    logits, labels = self.simcse_sup_loss(out)
                    loss = F.cross_entropy(logits, labels)

                self.optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                if n_iter % self.args.log_every_n_steps == 0:
                    top1, top5 = accuracy(logits, labels, topk=(1, 5))
                    self.writer.add_scalar('loss', loss, global_step=n_iter)
                    self.writer.add_scalar('acc/top1', top1[0], global_step=n_iter)
                    self.writer.add_scalar('acc/top5', top5[0], global_step=n_iter)
                    self.writer.add_scalar('learning_rate', self.scheduler.get_lr()[0], global_step=n_iter)

                n_iter += 1

                # warmup for the first 10 epochs
                if epoch >= 10:
                    self.scheduler.step()
                logging.debug(f"Epoch: {epoch}\tLoss: {loss}\tTop1 accuracy: {top1[0]}")

        if self.args.save_data:
            logging.info("Training has finished.")
            # save model checkpoints
            checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(self.args.epochs)
            save_checkpoint({
                'epoch': self.args.epochs,
                'arch': self.args.arch,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))
            logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")
        else:
            logging.info(f"Model is trained but the checkpoint and metadata has not been save.")

    def train_unsup(self, train_loader):
        n_iter = 0
        self.model.train()

        scaler = GradScaler(enabled=self.args.fp16_precision)

        # save config file
        save_config_file(self.writer.log_dir, self.args)

        logging.info(f"Start SimCSE training for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {self.args.disable_cuda}.")

        for epoch in range(self.args.epochs):
            for batch_idx, source in enumerate(tqdm(train_loader), start=1):

                real_batch_num = source.get('input_ids').shape[0]
                input_ids = source.get('input_ids').view(real_batch_num * 2, -1).to(self.args.device)
                attention_mask = source.get('attention_mask').view(real_batch_num * 2, -1).to(self.args.device)
                token_type_ids = source.get('token_type_ids').view(real_batch_num * 2, -1).to(self.args.device)

                with autocast(enabled=self.args.fp16_precision):
                    out = self.model_forward(input_ids, attention_mask, token_type_ids)
                    logits, labels = self.simcse_unsup_loss(out)
                    loss = F.cross_entropy(logits, labels)

                self.optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                if n_iter % self.args.log_every_n_steps == 0:
                    top1, top5 = accuracy(logits, labels, topk=(1, 5))
                    self.writer.add_scalar('loss', loss, global_step=n_iter)
                    self.writer.add_scalar('acc/top1', top1[0], global_step=n_iter)
                    self.writer.add_scalar('acc/top5', top5[0], global_step=n_iter)
                    self.writer.add_scalar('learning_rate', self.scheduler.get_lr()[0], global_step=n_iter)

                n_iter += 1

                # warmup for the first 10 epochs
                if epoch >= 10:
                    self.scheduler.step()
                logging.debug(f"Epoch: {epoch}\tLoss: {loss}\tTop1 accuracy: {top1[0]}")

        if self.args.save_data:
            logging.info("Training has finished.")
            # save model checkpoints
            checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(self.args.epochs)
            save_checkpoint({
                'epoch': self.args.epochs,
                'arch': self.args.arch,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))
            logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")
        else:
            logging.info(f"Model is trained but the checkpoint and metadata has not been save.")



