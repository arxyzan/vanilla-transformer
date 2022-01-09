import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch
from tqdm import tqdm
from functools import partial

from model import Transformer
from utils import AverageMeter
from config import config
from dataset import TranslationDataset, collate_fn


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
        self._init_weights()
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

    def _init_weights(self):
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param.data, mean=0, std=0.01)
            else:
                nn.init.constant_(param.data, 0)

    def train(self, dataloader, epoch, total_epochs):
        self.model.train()
        self.loss_tracker.reset()
        with tqdm(dataloader, unit="batch", desc=f'Epoch: {epoch}/{total_epochs} ',
                  bar_format='{desc:<16}{percentage:3.0f}%|{bar:70}{r_bar}', ascii=" #") as iterator:
            for src, trg in iterator:
                src, trg = src.to(self.device), trg.to(self.device)
                output = self.model(src, trg[:, :-1])
                output_dim = output.shape[-1]
                output = output.contiguous().view(-1, output_dim)
                trg = trg[:, 1:].contiguous().view(-1)

                loss = self.criterion(output, trg)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                self.loss_tracker.update(loss.item())
                avg_loss = self.loss_tracker.avg
                iterator.set_postfix(loss=avg_loss)
        return avg_loss

    def evaluate(self, dataloader):
        self.model.eval()
        self.loss_tracker.reset()
        with tqdm(dataloader, unit="batch", desc=f'Evaluating... ',
                  bar_format='{desc:<16}{percentage:3.0f}%|{bar:70}{r_bar}', ascii=" #") as iterator:
            with torch.no_grad():
                for src, trg in iterator:
                    src, trg = src.to(self.device), trg.to(self.device)
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
    base_url = config['base_url']
    train_urls = config['train_urls']
    val_urls = config['val_urls']
    test_urls = config['test_urls']
    batch_size = config['train_batch_size']

    train_dataset = TranslationDataset(base_url, train_urls)
    val_dataset = TranslationDataset(base_url, val_urls)
    test_dataset = TranslationDataset(base_url, test_urls)

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              collate_fn=partial(collate_fn,
                                                 bos_idx=train_dataset.BOS_IDX,
                                                 eos_idx=train_dataset.EOS_IDX,
                                                 pad_idx=train_dataset.PAD_IDX))
    valid_loader = DataLoader(val_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              collate_fn=partial(collate_fn,
                                                 bos_idx=val_dataset.BOS_IDX,
                                                 eos_idx=val_dataset.EOS_IDX,
                                                 pad_idx=val_dataset.PAD_IDX))
    test_loader = DataLoader(val_dataset,
                             batch_size=batch_size,
                             collate_fn=partial(collate_fn,
                                                bos_idx=test_dataset.BOS_IDX,
                                                eos_idx=test_dataset.EOS_IDX,
                                                pad_idx=test_dataset.PAD_IDX))

    config['src_vocab_size'] = len(train_dataset.de_vocab)
    config['trg_vocab_size'] = len(train_dataset.en_vocab)
    config['src_pad_idx'] = train_dataset.en_vocab['<pad>']
    config['trg_pad_idx'] = train_dataset.de_vocab['<pad>']
    trainer = Trainer(config)
    trainer.fit(train_loader, valid_loader, config['epochs'])
    # trainer.evaluate(valid_loader)
