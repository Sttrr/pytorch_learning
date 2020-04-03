import torch
import torch.utils.data as tud

#设定Hyper parameters
C=3
K=100

class WordEmbeddingDataset(tud.Dataset):
    def __init__(self,text,word_to_idx,idx_to_word,word_freqs,word_counts):
        super(WordEmbeddingDataset,self).__init__()
        self.text_encoded=[word_to_idx.get(word,word_to_idx["<unk>"]) for word in text]
        self.text_encoded=torch.LongTensor(self.text_encoded)
        self.word_to_idx=word_to_idx
        self.idx_to_word=idx_to_word
        self.word_freqs=torch.Tensor(word_freqs)
        self.word_counts=torch.Tensor(word_counts)

    def __len__(self):
        return len(self.text_encoded)

    def __getitem__(self,idx):
        center_word=self.text_encoded[idx]
        pos_indices=list(range(idx-C))+list(range(idx+1,idx+C+1))#window内单词index
        pos_indices=[i%len(self.text_encoded) for i in pos_indices]#取余，防止超出text长度
        pos_words=self.text_encoded[pos_indices]#周围单词
        neg_words=torch.multinomial(self.word_freqs,K*pos_words.shape[0],True)#负例采样

        return center_word,pos_words,neg_words