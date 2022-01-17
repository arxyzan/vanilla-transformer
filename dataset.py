import torch
from torch.utils.data import Dataset
from torchtext.vocab import build_vocab_from_iterator
from torchtext.utils import download_from_url, extract_archive
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.utils import get_tokenizer
import io


class Multi30kDe2En(Dataset):
    UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
    SPECIAL_SYMBOLS = ['<unk>', '<pad>', '<bos>', '<eos>']

    def __init__(self, base_url, urls):
        super(Multi30kDe2En, self).__init__()
        self.base_url = base_url
        self.urls = urls
        self.paths = [extract_archive(download_from_url(self.base_url + url))[0] for url in self.urls]
        self.de_tokenizer = get_tokenizer('spacy', language='de_core_news_sm')
        self.en_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
        self.de_texts = list(io.open(self.paths[0], encoding="utf8"))
        self.en_texts = list(io.open(self.paths[1], encoding="utf8"))
        self.de_tokens = [self.de_tokenizer(text) for text in self.de_texts]
        self.en_tokens = [self.en_tokenizer(text) for text in self.en_texts]
        self.de_vocab, self.en_vocab = self._load_vocabs()

    def __len__(self):
        return len(self.en_texts)

    def __getitem__(self, index):
        de_text = self.de_texts[index].rstrip("\n")
        en_text = self.en_texts[index].rstrip("\n")
        de_tensor = torch.tensor([self.de_vocab[token] for token in self.de_tokenizer(de_text)], dtype=torch.long)
        en_tensor = torch.tensor([self.en_vocab[token] for token in self.en_tokenizer(en_text)], dtype=torch.long)
        return de_tensor, en_tensor

    def _load_vocabs(self):
        de_vocab = build_vocab_from_iterator(iter(self.de_tokens), specials=self.SPECIAL_SYMBOLS)
        en_vocab = build_vocab_from_iterator(iter(self.en_tokens), specials=self.SPECIAL_SYMBOLS)
        de_vocab.set_default_index(self.UNK_IDX)
        en_vocab.set_default_index(self.UNK_IDX)

        return de_vocab, en_vocab

    @classmethod
    def collate_fn(cls, batch):
        de_batch, en_batch = [], []
        for de, en in batch:
            de_batch.append(torch.cat([torch.tensor([cls.BOS_IDX]), de, torch.tensor([cls.EOS_IDX])], dim=0))
            en_batch.append(torch.cat([torch.tensor([cls.BOS_IDX]), en, torch.tensor([cls.EOS_IDX])], dim=0))
        de_batch = pad_sequence(de_batch, padding_value=cls.PAD_IDX).permute(1, 0)
        en_batch = pad_sequence(en_batch, padding_value=cls.PAD_IDX).permute(1, 0)
        return de_batch, en_batch


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    base_url = 'https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/'
    urls = ('train.de.gz', 'train.en.gz')
    dataset = Multi30kDe2En(base_url, urls)
    dataloader = DataLoader(dataset, batch_size=16, collate_fn=Multi30kDe2En.collate_fn)
    de, en = next(iter(dataloader))
    print('done')
