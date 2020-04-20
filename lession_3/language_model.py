import torchtext
from torchtext.vocab import Vectors
import torch
import torch.nn as nn
import numpy as np
import random

import sys
sys.path.append(".")
from networks import language_model_network 

USE_CUDA=torch.cuda.is_available()

random.seed(53113)
np.random.seed(53113)
torch.manual_seed(53113)
if USE_CUDA:
    torch.cuda.manual_seed(53113)

BATCH_SIZE=22
EMBEDDING_SIZE=100
HIDDEN_SIZE=100
MAX_VOCAB_SIZE=50000
NUM_EPOCHS=2
GRAD_CLIP=5.0

TEXT=torchtext.data.Field(lower=True)
train,val,test=torchtext.datasets.LanguageModelingDataset.splits(path="./text8",train="text8.train.txt",validation="text8.dev.txt",test="text8.test.txt",text_field=TEXT)
TEXT.build_vocab(train,max_size=MAX_VOCAB_SIZE)

device=torch.device("cuda" if USE_CUDA else "cpu")

train_iter,val_iter,text_iter=torchtext.data.BPTTIterator.splits(
    (train,val,test),batch_size=BATCH_SIZE,device=device,bptt_len=50,repeat=False,shuffle=True)

VOCAB_SIZE=len(TEXT.vocab)
model=language_model_network.RNNModel(vocab_size=VOCAB_SIZE,
                                        embed_size=EMBEDDING_SIZE,
                                        hidden_size=HIDDEN_SIZE)

if USE_CUDA:
    model=model.to(device)

loss_fn=nn.CrossEntropyLoss()
learning_rate=0.001
optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)

def repackage_hidden(h):
    if isinstance(h,torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

for epoch in range(NUM_EPOCHS):
    model.train()
    it=iter(train_iter)
    hidden=model.init_hidden(BATCH_SIZE)
    for i,batch in enumerate(it):
        data,target=batch.text,batch.target
        hidden=repackage_hidden(hidden)
        output,hidden=model(data,hidden)

        loss=loss_fn(output.view(-1,VOCAB_SIZE),target.view(-1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),GRAD_CLIP)
        optimizer.step()

        if i%100==0:
            print("loss",loss.item())
        #保存模型
        if i%1000==0:
            torch.save(model.state_dict(),"1m.pth")
        

    

#print(TEXT.vocab.itos[:10])
#print(batch)
