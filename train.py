import os
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torchtext
from torchtext.legacy.datasets import Multi30k
from torchtext.legacy.data import Field, BucketIterator
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import spacy
import numpy as np
from tqdm import tqdm

import random
import math
import time

from model import Transformer
from utils import AverageMeter
from config import config


class Trainer:
    def __init__(self, config):
        # Configs & Parameters
        self.config = config
        self.src_vocab_size = config['src_vocab_size']
        self.trg_vocab_size = config['trg_vocab_size']
        self.mlp_expansion_dim = config['mlp_expansion_dim']
        self.embed_dim = config['embed_dim']
        self.n_blocks = config['n_blocks']
        self.n_heads = config['n_heads']
        self.max_length = config['max_length']
        self.dropout = config['dropout']
        self.device = config['device']
        self.src_pad_idx = config['src_pad_idx']
        self.trg_pad_idx = config['trg_pad_idx']
        self.lr = config['lr']
        self.grad_clip = config['grad_clip']
        # Model
        self.model = Transformer(self.src_vocab_size,
                                 self.trg_vocab_size,
                                 self.src_pad_idx,
                                 self.trg_pad_idx,
                                 self.embed_dim,
                                 self.n_blocks,
                                 self.n_heads,
                                 self.mlp_expansion_dim,
                                 self.max_length,
                                 self.dropout,
                                 self.device)
        self.model.to(self.device)

        # Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        # Loss Function
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.trg_pad_idx)
        self.criterion.to(self.device)

        # Metrics
        self.loss_tracker = AverageMeter('loss')

        # Tensorboard
        log_dir = os.path.join(self.config['log_dir'], self.config['name'])
        self.writer = SummaryWriter(log_dir=log_dir)

    def train(self, dataloader, epoch, total_epochs):
        self.model.train()
        self.loss_tracker.reset()
        with tqdm(dataloader, unit="batch", desc=f'Epoch: {epoch}/{total_epochs} ',
                  bar_format='{desc:<16}{percentage:3.0f}%|{bar:70}{r_bar}', ascii=" #") as iterator:
            for batch in iterator:
                src = batch.src
                trg = batch.trg

                self.optimizer.zero_grad()
                output = self.model(src, trg[:, :-1])
                output_dim = output[-1]
                output = output.contiguous().view(-1, output_dim)
                trg = trg[:, 1:].contiguous().view(-1)

                loss = self.criterion(output, trg)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

                self.loss_tracker.update(loss.item())
                avg_loss = self.loss_tracker.avg
                iterator.set_postfix(loss=avg_loss)
        return avg_loss

    def evaluate(self, dataloader):
        self.model.eval()
        self.loss_tracker.reset()
        with tqdm(dataloader, unit="batch", desc=f'Evaluating... ',
                  bar_format='{desc:<16}{percentage:3.0f}%|{bar:70}{r_bar}', ascii=" #") as iterator:
            for i, batch in enumerate(iterator):
                src = batch.src
                trg = batch.trg

                output = self.model(src, trg[:, :-1])

                output_dim = output.shape[-1]

                output = output.contiguous().view(-1, output_dim)
                trg = trg[:, 1:].contiguous().view(-1)

                loss = self.criterion(output, trg)
                self.loss_tracker.update(loss.item())
                avg_loss = self.loss_tracker.avg
                iterator.set_postfix(loss=avg_loss)
        return avg_loss

    def fit(self, train_loader, valid_loader, epochs):
        for epoch in range(1, epochs + 1):
            print()
            train_loss = self.train(train_loader, epoch, epochs)
            val_loss = self.evaluate(valid_loader)

            # tensorboard
            self.writer.add_scalar('train_loss', train_loss, epoch)
            self.writer.add_scalar('val_loss', val_loss, epoch)

            should_save_weights = lambda x: not bool(x % self.config['save_interval'])
            if should_save_weights(epoch):
                save_path = os.path.join(self.config['weights_dir'], f'{epoch}_{val_loss}.pt')
                torch.save(self.model.state_dict(), save_path)
                print(f'Saved Model at {save_path}')


if __name__ == '__main__':
    train_loader = ...
    valid_loader = ...
    trainer = Trainer(config)
    trainer.fit(train_loader, valid_loader, config['epochs'])
