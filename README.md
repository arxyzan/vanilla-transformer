## Vanilla Transformer (PyTorch)
My PyTorch implementation of the original Transformer model from the paper [Attention Is All You Need](https://arxiv.org/abs/1706.03762) inspired by all the codes and blogs I've read on this topic. There's nothing really special going on here except the fact that I tried to make it as barebone as possible. There is also a training code prepared for a simple German -> English translator written in pure PyTorch using Torchtext library.

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
**Note:** _This code uses Torchtext's new API (v0.10.0+) and the `dataset.py` contains a custom text dataset class inherited from `torch.utils.data.Dataset` and is different from the classic methods using `Field` and `BucketIterator` (which are now moved to `torchtext.legacy`). Nevertheless `torchtext` library is still under heavy development so this code will probably break with the upcoming versions._

### Train
In `train.py` we train a simple German -> English translation model on Multi30k dataset using the Transformer model. Make sure you configure the necessary paths for weights, logs, etc in `config.py`. Then you can simply run the file as below:
```python
python train.py
```
```bash
Epoch: 1/10     100%|######################################################################| 227/227 [00:10<00:00, 21.61batch/s, loss=4.33]
Evaluating...   100%|######################################################################| 8/8 [00:00<00:00, 45.25batch/s, loss=3.13]
Saved Model at weights/1.pt

Epoch: 2/10     100%|######################################################################| 227/227 [00:10<00:00, 22.64batch/s, loss=2.82]
Evaluating...   100%|######################################################################| 8/8 [00:00<00:00, 51.68batch/s, loss=2.55]
Saved Model at weights/2.pt

Epoch: 3/10     100%|######################################################################| 227/227 [00:10<00:00, 22.56batch/s, loss=2.22]
Evaluating...   100%|######################################################################| 8/8 [00:00<00:00, 51.98batch/s, loss=2.22]
Saved Model at weights/3.pt

Epoch: 4/10     100%|######################################################################| 227/227 [00:10<00:00, 22.64batch/s, loss=1.83]
Evaluating...   100%|######################################################################| 8/8 [00:00<00:00, 52.20batch/s, loss=2.07]
Saved Model at weights/4.pt

Epoch: 5/10     100%|######################################################################| 227/227 [00:10<00:00, 22.64batch/s, loss=1.55]
Evaluating...   100%|######################################################################| 8/8 [00:00<00:00, 52.12batch/s, loss=2]   
Saved Model at weights/5.pt

Epoch: 6/10     100%|######################################################################| 227/227 [00:10<00:00, 22.25batch/s, loss=1.34]
Evaluating...   100%|######################################################################| 8/8 [00:00<00:00, 51.45batch/s, loss=1.95]
Saved Model at weights/6.pt

Epoch: 7/10     100%|######################################################################| 227/227 [00:10<00:00, 22.55batch/s, loss=1.17]
Evaluating...   100%|######################################################################| 8/8 [00:00<00:00, 51.34batch/s, loss=1.95]
Saved Model at weights/7.pt

Epoch: 8/10     100%|######################################################################| 227/227 [00:10<00:00, 22.46batch/s, loss=1.03]
Evaluating...   100%|######################################################################| 8/8 [00:00<00:00, 51.43batch/s, loss=1.96]
Saved Model at weights/8.pt

Epoch: 9/10     100%|######################################################################| 227/227 [00:10<00:00, 22.45batch/s, loss=0.91] 
Evaluating...   100%|######################################################################| 8/8 [00:00<00:00, 52.84batch/s, loss=1.99]
Saved Model at weights/9.pt

Epoch: 10/10    100%|######################################################################| 227/227 [00:10<00:00, 22.50batch/s, loss=0.808]
Evaluating...   100%|######################################################################| 8/8 [00:00<00:00, 51.74batch/s, loss=2.01]
Saved Model at weights/10.pt

```
### Inference
Given the sentence `Eine Gruppe von Menschen steht vor einem Iglu` as input in `predict.py` we get the following output which is pretty decent despite that our Transformer model is roughly complex and our dataset is fairly simple.
```bash
python predict.py
```
```bash
"Translation:  A group of people standing in front of a warehouse ."
```

### TODO
- [x] `predict.py` for inference
- [x] Add pretrained weights
- [ ] Visualize attentions
- [ ] Add docstrings
- [ ] An in-depth notebook
