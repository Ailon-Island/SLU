#coding=utf8
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import numpy as np
from torchcrf import CRF
from elmoformanylangs import Embedder

class SLUTagging(nn.Module):

    def __init__(self, config):
        super(SLUTagging, self).__init__()
        self.config = config
        self.cell = config.encoder_cell
        self.word_embed = nn.Embedding(config.vocab_size, config.embed_size, padding_idx=0)
        self.embedder = Embedder('./pretrain/zhs.model')
        self.rnn = getattr(nn, self.cell)(config.embed_size, config.hidden_size // 2, num_layers=config.num_layer, bidirectional=True, batch_first=True)
        self.dropout_layer = nn.Dropout(p=config.dropout)
        self.output_layer = TaggingFNNDecoder(config.hidden_size, config.num_tags, config.tag_pad_idx)

    def forward(self, batch):
        tag_ids = batch.tag_ids
        tag_mask = batch.tag_mask
        # print(tag_mask)
        input_ids = batch.input_ids
        lengths = batch.lengths
        # utt = batch.utt
        # utt_length = len(utt[0])
        # embed = torch.Tensor(np.stack([np.pad(res, ((0, utt_length - res.shape[0]),(0, 0))) for res in self.embedder.sents2elmo(utt)])).to('cuda')
        embed = self.word_embed(input_ids)
        packed_inputs = rnn_utils.pack_padded_sequence(embed, lengths, batch_first=True, enforce_sorted=True)
        packed_rnn_out, h_t_c_t = self.rnn(packed_inputs)  # bsize x seqlen x dim
        rnn_out, unpacked_len = rnn_utils.pad_packed_sequence(packed_rnn_out, batch_first=True)
        hiddens = self.dropout_layer(rnn_out)
        tag_output = self.output_layer(hiddens, tag_mask, tag_ids)

        return tag_output

    def decode(self, label_vocab, batch):
        labels = batch.labels
        prob, loss = self.forward(batch)
        predictions = label_vocab.decode(prob, batch.utt)

        return predictions, labels, loss.cpu().item()


class TaggingFNNDecoder(nn.Module):

    def __init__(self, input_size, num_tags, pad_id):
        super(TaggingFNNDecoder, self).__init__()
        self.num_tags = num_tags
        self.output_layer = nn.Linear(input_size, num_tags)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=pad_id)

    def forward(self, hiddens, mask, labels=None):
        logits = self.output_layer(hiddens)
        logits += (1 - mask).unsqueeze(-1).repeat(1, 1, self.num_tags) * -1e32
        # print(logits.shape) # [batch, seq_len, num_tags]
        # print(self.num_tags) # 74
        prob = torch.softmax(logits, dim=-1)
        # print(prob.shape) # [batch, seq_len, num_tags]
        # print(labels)
        if labels is not None:
            loss = self.loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))
            return prob, loss
        return prob

class CRFDecoder(nn.Module):
    def __init__(self, input_size, num_tags, pad_id=0):
        super().__init__()
        self.num_tags = num_tags
        self.output_layer = nn.Linear(input_size, num_tags)
        self.crf = CRF(num_tags, batch_first=True)

    def forward(self, hiddens, mask, labels=None):
        logits = self.output_layer(hiddens)
        res = self.crf.decode(logits, mask.bool())
        if labels is not None:
            fwd_loss = (-1)*self.crf(logits, labels, mask.bool(), reduction='mean')
            return res, fwd_loss
        return res