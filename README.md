## Vanilla Transformer (PyTorch)
My PyTorch implementation of the original Transformer model from the paper [Attention Is All You Need](https://arxiv.org/abs/1706.03762) inspired by all the codes and blogs I've read on this topic. There's nothing really special going on here except the fact that I tried to make it as barebone as possible. There is also a training code prepared for a simple EN/DE translator written in pure PyTorch using Torchtext library.

### My Inspirations
- [The Illustrated Transformer by Jay Alammar](https://jalammar.github.io/illustrated-transformer/)
- [The Original Transformer (PyTorch) by Aleksa Gordic](https://github.com/gordicaleksa/pytorch-original-transformer)
- [Attention is all you need from scratch by Aladdin Persson](https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/more_advanced/Seq2Seq_attention/seq2seq_attention.py)
- [PyTorch Seq2Seq by Ben Trevett](https://github.com/bentrevett/pytorch-seq2seq)
- [Transformers: Attention in Disguise by Mihail Eric](https://www.mihaileric.com/posts/transformers-attention-in-disguise/)
- [The Annotated Transformer by Harvard NLP](http://nlp.seas.harvard.edu/2018/04/03/attention.html)

And probably a couple more which I don't remember ...

### Prerequisites
1. Install the required pip packages:
```bash
pip install -r requirements.txt
```
2. Install `spacy` models :
```bash
python -m spacy download de_core_news_sm
python -m spacy download en_core_web_sm
```

### Train
In `train.py` we train a simple German -> English translation model on Multi30k dataset using the Transformer model. Make sure you configure the necessary paths for weights, logs, etc in `config.py`. Then you can simply run the file as below:
```python
python train.py
```
**Note:** _This code uses Torchtext's new API (v0.11.1) and the `dataset.py` contains a custom text dataset class inherited from `torch.utils.data.Dataset` and is different from the classic methods using `Field` and `BucketIterator` (which are now moved to `torchtext.legacy`). Nevertheless `torchtext` library is still under heavy development so this code will probably break with the upcoming versions._

### TODO
- `predict.py` for inference
- visualize attentions
- add docstrings
- an in-depth notebook
