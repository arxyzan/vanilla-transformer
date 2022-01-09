import torch
from torch.utils.data import Dataset
from torchtext.vocab import build_vocab_from_iterator
from torchtext.utils import download_from_url, extract_archive
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.utils import get_tokenizer
import io


class TranslationDataset(Dataset):
    def __init__(self, base_url, urls):
        super(TranslationDataset, self).__init__()
        self.base_url = base_url
        self.urls = urls
        self.paths = [extract_archive(download_from_url(self.base_url + url))[0] for url in self.urls]
        self.de_tokenizer = get_tokenizer('spacy', language='de_core_news_sm')
        self.en_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
        self.de_vocab, self.en_vocab = self._load_vocabs()
        self.PAD_IDX = self.en_vocab['<pad>']
        self.SOS_IDX = self.en_vocab['<sos>']
        self.BOS_IDX = self.en_vocab['<bos>']
        self.EOS_IDX = self.en_vocab['<eos>']
        self.de_texts = list((io.open(self.paths[0], encoding="utf8")))
        self.en_texts = list((io.open(self.paths[1], encoding="utf8")))

    def __len__(self):
        return len(self.en_texts)

    def __getitem__(self, index):
        en_text = self.en_texts[index]
        de_text = self.de_texts[index]
        en_tensor = torch.tensor([self.en_vocab[token] for token in self.de_tokenizer(en_text)], dtype=torch.long)
        de_tensor = torch.tensor([self.de_vocab[token] for token in self.de_tokenizer(de_text)], dtype=torch.long)
        return de_tensor, en_tensor

    def _load_vocabs(self):
        def yield_tokens(filepath, tokenizer):
            with io.open(filepath, encoding='utf8') as f:
                for text in f:
                    yield tokenizer(text)

        de_vocab = build_vocab_from_iterator(yield_tokens(self.paths[0], self.de_tokenizer),
                                             specials=['<unk>', '<pad>', '<bos>', '<eos>'])
        en_vocab = build_vocab_from_iterator(yield_tokens(self.paths[1], self.en_tokenizer),
                                             specials=['<unk>', '<pad>', '<bos>', '<eos>'])
        de_vocab.set_default_index(de_vocab["<unk>"])
        en_vocab.set_default_index(en_vocab["<unk>"])

        return de_vocab, en_vocab


def collate_fn(batch, bos_idx, eos_idx, pad_idx):
    de_batch, en_batch = [], []
    for de, en in batch:
        de_batch.append(torch.cat([torch.tensor([bos_idx]), de, torch.tensor([eos_idx])], dim=0))
        en_batch.append(torch.cat([torch.tensor([bos_idx]), en, torch.tensor([eos_idx])], dim=0))
    de_batch = pad_sequence(de_batch, padding_value=pad_idx).permute(1, 0)
    en_batch = pad_sequence(en_batch, padding_value=pad_idx).permute(1, 0)
    return de_batch, en_batch


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from functools import partial

    base_url = 'https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/'
    urls = ('val.de.gz', 'val.en.gz')
    dataset = TranslationDataset(base_url, urls)
    dataloader = DataLoader(dataset, batch_size=32, collate_fn=partial(collate_fn,
                                                                       bos_idx=dataset.BOS_IDX,
                                                                       eos_idx=dataset.EOS_IDX,
                                                                       pad_idx=dataset.PAD_IDX))
    print(next(iter(dataloader)))
    print('done')
