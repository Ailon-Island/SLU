import torch
import math
import torch.nn as nn
import numpy as np
import torch.nn.utils.rnn as rnn_utils
from model.slu_baseline_tagging import TaggingFNNDecoder
import torch.nn.functional as F
from model.attention import SelfAttention
from elmoformanylangs import Embedder
import numpy as np
from model.slstm import SLSTM

class sLSTM_attention(nn.Module):
    def __init__(self, d_hidden, dropout):
        super(sLSTM_attention, self).__init__()
        self.sw1 = nn.Sequential(nn.Conv1d(d_hidden, d_hidden, kernel_size=1, padding=0),
                                 nn.BatchNorm1d(d_hidden), nn.ReLU())
        self.sw3 = nn.Sequential(nn.Conv1d(d_hidden, d_hidden, kernel_size=1, padding=0),
                                 nn.ReLU(), nn.BatchNorm1d(d_hidden),
                                 nn.Conv1d(d_hidden, d_hidden, kernel_size=3, padding=1),
                                 nn.ReLU(), nn.BatchNorm1d(d_hidden))
        self.sw33 = nn.Sequential(nn.Conv1d(d_hidden, d_hidden, kernel_size=1, padding=0),
                                  nn.ReLU(), nn.BatchNorm1d(d_hidden),
                                  nn.Conv1d(d_hidden, d_hidden, kernel_size=5, padding=2),
                                  nn.ReLU(), nn.BatchNorm1d(d_hidden))
        self.linear = nn.Sequential(nn.Linear(2 * d_hidden, 2 * d_hidden), nn.GLU(),
                                    nn.Dropout(dropout))
        self.filter_linear = nn.Linear(3 * d_hidden, d_hidden)
        self.attention = SelfAttention(input_dim=d_hidden, hidden_dim=d_hidden, output_dim=d_hidden,
                                       dropout_rate=dropout)
        self.sigmoid = nn.Sigmoid()

    def _get_conv(self, src):
        old = src
        src = src.transpose(0, 1).transpose(1, 2)  # (l,b,d) ->(b,l,d) ->(b,d,l)
        conv1 = self.sw1(src)
        conv3 = self.sw3(src)
        conv33 = self.sw33(src)
        conv = torch.cat([conv1, conv3, conv33], dim=1)  # (b,3d,l)
        conv = self.filter_linear(conv.transpose(1, 2)).transpose(0, 1)  # (b,3d,l)->(b,l,3d)-> (b,l,d) -> (l,b,d)
        conv += old
        return conv

    def _get_self_attn(self, src, mask):
        attn = self.attention(src, mask)
        attn += src
        return attn

    def forward(self, seq, mask):
        conv = self._get_conv(seq)
        attn = self._get_self_attn(conv, mask)
        attn_gate = self.sigmoid(attn)
        return attn_gate


class sLSTM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.output_layer = TaggingFNNDecoder(config.hidden_size, config.num_tags, config.tag_pad_idx)
        self.word_embed = nn.Embedding(config.vocab_size, config.embed_size, padding_idx=0)
        self.Rnn = SLSTM(config.vocab_szie, config.embed_size, config.hidden_size, num_layer=1, return_all=False)
        self.dropout_layer = nn.Dropout(p=config.dropout)
        self.d_hidden = config.hidden_size
        self.attention = sLSTM_attention(self.d_hidden, config.dropout)
        self.steps = 5
        self.sentence_node = 1
        self.embedder= Embedder('./pretrain/zhs.model')
    def get_len(self, tensor):
        mask = tensor.ne(0)
        return mask.sum(dim=-1)

    def forward(self, batch):
        tag_ids = batch.tag_ids
        tag_mask = batch.tag_mask
        input_ids = batch.input_ids
        lengths = batch.lengths

        utt = batch.utt
        utt_length = len(utt[0])
        embed = torch.Tensor(np.stack([np.pad(res, ((0, utt_length - res.shape[0]),(0, 0))) for res in self.embedder.sents2elmo(utt)])).to('cuda')
        x = self.Rnn(embed, tag_mask)
        hidden = self.lstm_attention(x)
        tag_output = self.output_layer(hidden, tag_mask, tag_ids)
        return tag_output

    def decode(self, label_vocab, batch):
        batch_size = len(batch)
        labels = batch.labels
        prob, loss = self.forward(batch)
        predictions = []
        for i in range(batch_size):
            pred = torch.argmax(prob[i], dim=-1).cpu().tolist()
            pred_tuple = []
            idx_buff, tag_buff, pred_tags = [], [], []
            pred = pred[:len(batch.utt[i])]
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
        return predictions, labels, loss.cpu().item()
