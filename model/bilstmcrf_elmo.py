import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from torchcrf import CRF
from elmoformanylangs import Embedder
import numpy as np
import jieba

def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()
    
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

# START_TAG = "<START>"
# STOP_TAG = "<STOP>"
START_IDX = 74
STOP_IDX = 75
class BiLSTM_CRF(nn.Module):

    def __init__(self, config):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = config.embed_size
        self.hidden_dim = config.hidden_size
        self.vocab_size = config.vocab_size
        self.tag_pad_idx = config.tag_pad_idx
        # self.tag_to_ix = tag_to_ix
        self.tagset_size = config.num_tags
        self.embedder = Embedder('./pretrain/zhs.model')
        print(self.vocab_size, self.embedding_dim)
        # self.word_embed = nn.Embedding(self.vocab_size, self.embedding_dim)
        
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim // 2,
                            num_layers=1, bidirectional=True, batch_first=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(self.hidden_dim, self.tagset_size)

        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        self.crf = CRF(self.tagset_size, batch_first=True)

        self.hidden = self.init_hidden()
    def init_hidden(self, batch_size=32):
        return (torch.randn(2, batch_size, self.hidden_dim // 2).to('cuda'),
                torch.randn(2, batch_size, self.hidden_dim // 2).to('cuda'))

    def decode(self, label_vocab, batch):
        tag_ids = batch.tag_ids
        tag_mask = batch.tag_mask
        sentence = batch.input_ids
        lengths = batch.lengths
        labels = batch.labels
        utt = batch.utt
        batch_size = batch.input_ids.shape[0]
        predictions = []
        lstm_feats = self._get_lstm_features(sentence, utt)
        # score, tag_seq = self._viterbi_decode(lstm_feats)
        res = self.crf.decode(lstm_feats, tag_mask.bool())
        for i in range(batch_size):
            pred = res[i]
            pred_tuple = []
            idx_buff, tag_buff, pred_tags = [], [], []
            # pred = pred[:len(batch.utt[i])]
            for idx, tid in enumerate(pred):

                tag = label_vocab.convert_idx_to_tag(tid)
                pred_tags.append(tag)
                if (tag == 'O' or tag.startswith('B')) and len(tag_buff) > 0:
                    slot = '-'.join(tag_buff[0].split('-')[1:])
                    value = ''.join([batch.utt[i][j] for j in idx_buff])
                    idx_buff, tag_buff = [], []
                    pred_tuple.append(f'{slot}-{value}')
                    if tag.startswith('B'):
                        idx_buff.append(idx)
                        tag_buff.append(tag)
                elif tag.startswith('I') or tag.startswith('B'):
                    idx_buff.append(idx)
                    tag_buff.append(tag)
            if len(tag_buff) > 0:
                slot = '-'.join(tag_buff[0].split('-')[1:])
                value = ''.join([batch.utt[i][j] for j in idx_buff])
                pred_tuple.append(f'{slot}-{value}')
            predictions.append(pred_tuple)
        # print(predictions)
        return predictions, labels

    def forward(self, batch):
        sentence = batch.input_ids
        tag_ids = batch.tag_ids
        tag_mask = batch.tag_mask
        utt = batch.utt
        feats = self._get_lstm_features(sentence, utt)
        fwd_loss = (-1)*self.crf(feats, tag_ids, tag_mask.bool(), reduction='mean')
        return fwd_loss

    def _get_lstm_features(self, sentence, utt):
        self.hidden = self.init_hidden(sentence.shape[0])
        # embeds = self.word_embed(sentence)
        length = len(utt[0])
        embeds = torch.Tensor(np.stack([np.pad(res, ((0, length - res.shape[0]),(0, 0))) for res in self.embedder.sents2elmo(utt)])).to('cuda')
        # max_len = 0
        # max_len = len(utt[0])
        # cut_utt = []
        # for item in utt:
        #     res = list(jieba.cut(item))
        #     cut_utt.append(res)
        #     # max_len = max(max_len, len(res))
        # embeds = torch.Tensor(np.stack([np.pad(res, ((0, max_len - res.shape[0]),(0, 0))) for res in self.embedder.sents2elmo(cut_utt)])).to('cuda')
        print(embeds.shape)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(sentence.shape[0], sentence.shape[1], self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats



