import torch
import torch.nn as nn



class Dictionary():
    def __init__(self, word2idx=None, idx2word=None):
        if word2idx is None:
            word2idx = {}
        if idx2word is None:
            idx2word = []
        self.word2idx = word2idx
        self.idx2word = idx2word


class Captioning(nn.Module):
    def __init__(self, vocab_size, embedding_dim, feature_dim, hidden, memdim):
        super(Captioning, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.feature_dim = feature_dim
        self.hidden = hidden
        self.memdim = memdim

        self.embed = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.topdown = nn.LSTM(self.embedding_dim + self.feature_dim, self.memdim)

        self.att1 = nn.Linear(self.feature_dim, self.hidden, bias=False)
        self.att2 = nn.Linear(self.memdim, self.hidden, bias=False)
        self.att3 = nn.Tanh()
        self.w_a = nn.Linear(self.hidden, 1, bias=False)
        self.att4 = nn.Softmax(dim=-1)

        self.langlstm = nn.LSTM(self.feature_dim + self.memdim, self.memdim)
        self.word = nn.Linear(self.memdim, self.vocab_size)
        self.act = nn.Softmax(dim=-1)

    """
    input : (image feature, sequence)
    image feature ...   mean-pooled convolutional features extracted from regions of interest
                        size should be (featurenum, featuredim)
    sequence      ...   sequence containing index integers. should be LongTensor
                        size should be (batchsize, seq_length)
    output : (seq_length, batchsize, vocabularysize)
    """
    def forward(self, feature, sequence):
        batchsize = sequence.size(0)
        # expfeature : (batchsize, featurenum, featuredim)
        expfeature = feature.unsqueeze(0).expand(batchsize, -1, -1)
        # avgfeature : (batchsize, featuredim)
        avgfeature = expfeature.mean(1)
        # wordft : (batchsize, seq_length, embedding_dim)
        wordft = self.embed(sequence)
        input = avgfeature.unsqueeze(1).expand(-1, wordft.size(1), -1)
        # input : (batchsize, seq_length, embedding_dim + featuredim)
        input = torch.cat((wordft, input), 2)
        # h_t1 : (seq_length, batchsize, memdim)
        h_t1, _ = self.topdown(input.transpose(0, 1))
        # out : (seq_length, batchsize, featurenum, hidden)
        out = self.att3(self.att1(expfeature) + self.att2(h_t1).unsqueeze(2))
        # a_t : (seq_length, batchsize, featurenum)
        a_t = self.w_a(out).squeeze(-1)
        alpha_t = self.att4(a_t)
        # v_hat : (seq_length, batchsize, featuredim)
        v_hat = torch.matmul(alpha_t, feature)

        # input2 : (seq_length, batchsize, featuredim + memdim)
        input2 = torch.cat((v_hat, h_t1), 2)
        # h_t2 : (seq_length, batchsize, memdim)
        h_t2, _ = self.langlstm(input2)
        # y_t : (seq_length, batchsize, vocab.size)
        y_t = self.act(self.word(h_t2))
        return y_t



