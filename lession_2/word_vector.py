from __future__ import absolute_import, division, print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append(".")

from datasets import word_dataset
from collections import Counter
import numpy as np
import random
import math

import pandas as pd
import scipy
import sklearn
from sklearn.metrics.pairwise import cosine_similarity

USE_CUDA=torch.cuda.is_available()

#保证随机数一致，实验结果可复现
random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
if USE_CUDA:
    torch.cuda.manual_seed(1)

#设定Hyper parameters
NUM_EPOCHS=2
MAX_VOCAB_SIZE=30000
LEARNING_RATE=0.2
EMBEDDING_SIZE=100
BATCH_SIZE=2

def word_tokenize(text):
    return text.split()

with open("text8.train.txt","r") as fin:
    text=fin.read()

text=text.split()

vocab=dict(Counter(text).most_common(MAX_VOCAB_SIZE-1))
vocab["<unk>"]=len(text)-np.sum(list(vocab.values()))

idx_to_word=[word for word in vocab.keys()]
word_to_idx={word:i for i,word in enumerate(idx_to_word)}#Q：跟vocab有不是一样吗？？？A：值是0,1,2而不是出现的次数了

word_counts=np.array([count for count in vocab.values()],dtype=np.float32)
word_freqs=word_counts/np.sum(word_counts)
word_freqs=word_freqs**(3./4.)
word_freqs=word_freqs/np.sum(word_freqs)#normalize,不是应该除word_freqs的叠加???
VOCAB_SIZE=len(idx_to_word)

dataset=word_dataset.WordEmbeddingDataset(text,word_to_idx,idx_to_word,word_freqs,word_counts)
dataloader=word_dataset.tud.DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=4)



for i,(center_word,pos_words,neg_words) in enumerate(dataloader):
    print(i,center_word,pos_words,neg_words)
    if i>5:
        break

#print(list(vocab.items())[:100])
#print(vocab["<unk>"])