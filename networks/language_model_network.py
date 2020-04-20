import torch.nn as nn

class RNNModel(nn.Module):
    def __init__(self,vocab_size,embed_size,hidden_size):
        super(RNNModel,self).__init__()
        self.embed=nn.Embedding(vocab_size,embed_size)
        self.lstm=nn.LSTM(embed_size,hidden_size)
        self.decoder=nn.Linear(hidden_size,vocab_size)
        self.hidden_size=hidden_size


    def forward(self,text,hidden):
        #text:seq_length*batch_size
        emb=self.embed(text)#seq_length*batch_size*embed_size
        output,hidden=self.lstm(emb,hidden)
        #output:seq_length*batch_size*hidden_size
        #hidden:(1*batch_size*hidden_size,1*batch_size*hidden_size)
        out_vocab=self.decoder(output.view(-1,output.shape[2]))#(seq_length*batch_size)*vocab_size
        out_vocab=out_vocab.view(output.size(0),output.size(1),-1)

        return out_vocab,hidden

    def init_hidden(self,bsz,requires_grad=True):
        weight=next(self.parameters())
        return (weight.new_zeros((1,bsz,self.hidden_size),requires_grad=True),
        weight.new_zeros((1,bsz,self.hidden_size),requires_grad=True))

