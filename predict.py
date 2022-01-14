import torch
import torch.nn as nn
from torchtext.vocab import Vocab
from torchtext.data.utils import get_tokenizer
from typing import Union
from model import Transformer


def translate_sentence(sentence: Union[list, str], model: Transformer, src_vocab: Vocab, trg_vocab: Vocab, max_len=50,
                       device='cpu'):
    model.eval()
    if isinstance(sentence, str):
        de_tokenizer = get_tokenizer('spacy', language='de_core_news_sm')
        tokens = de_tokenizer(sentence.lower())
    else:
        tokens = [token.lower() for token in sentence]

    tokens = ['<bos>'] + tokens + ['<eos>']  # add bos and eos tokens to the sides of the sentence
    src_indices = [src_vocab[token] for token in tokens]

    src_tensor = torch.LongTensor(src_indices).unsqueeze(0).to(device)
    src_mask = model.src_mask(src_tensor)

    with torch.no_grad():
        src_encoded = model.encoder(src_tensor, src_mask)

    trg_indices = ['<bos>']  # an empty target sentence to be filled in the following loop

    for i in range(max_len):
        trg_tensor = torch.LongTensor(trg_indices).unsqueeze(0).to(device)
        trg_mask = model.trg_mask(trg_tensor).to(device)

        with torch.no_grad():
            output = model.decoder(trg_tensor, src_encoded, trg_mask, src_mask)

        pred_token = output.argmax(2)[:, -1].item()
        trg_indices.append(pred_token)

        if pred_token == '<eos>':
            break

    output_tokens = [trg_vocab.itos(indice) for indice in trg_indices]

    return output_tokens[1:]
