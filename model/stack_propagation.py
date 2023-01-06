import math

import ipdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from model.attention import SelfAttention


class Stack_propagation(nn.Module):
    def __init__(self, config):
        super().__init__()
        vocab_size = config.vocab_size
        word_emb_dim = config.embed_size
        hidden_dim = config.hidden_size
        num_classes = config.num_tags
        dropout = config.dropout
        self.word_embed = nn.Embedding(vocab_size, word_emb_dim, padding_idx=0)
        self.encoder = LSTMEncoder(word_emb_dim, hidden_dim, dropout)
        self.attention = SelfAttention(word_emb_dim, hidden_dim, hidden_dim, dropout)
        self.decoder = LSTMDecoder(2 * hidden_dim, hidden_dim, num_classes, dropout)

    def forward(self, batch):
        input_ids = batch.input_ids
        tag_mask = batch.tag_mask
        tag_ids = batch.tag_ids
        length = batch.lengths

        embed = self.word_embed(input_ids)
        lstm_hidden = self.encoder(embed, length)
        attention_hidden = self.attention(embed, length, is_flat=True)
        import ipdb
        ipdb.set_trace()
        hidden = torch.cat([attention_hidden, lstm_hidden], dim=1)
        slot = self.decoder(hidden, length)
        import ipdb
        ipdb.set_trace()
        return slot

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

class LSTMEncoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, dropout_rate):
        super(LSTMEncoder, self).__init__()
        self.__embedding_dim = embedding_dim
        self.__hidden_dim = hidden_dim // 2
        self.__dropout_rate = dropout_rate
        self.__dropout_layer = nn.Dropout(self.__dropout_rate)
        self.__lstm_layer = nn.LSTM(
            input_size=self.__embedding_dim,
            hidden_size=self.__hidden_dim,
            batch_first=True,
            bidirectional=True,
            dropout=self.__dropout_rate,
            num_layers=1
        )

    def forward(self, embedded_text, seq_lens):
        dropout_text = self.__dropout_layer(embedded_text)
        packed_text = pack_padded_sequence(dropout_text, seq_lens, batch_first=True)
        lstm_hiddens, (h_last, c_last) = self.__lstm_layer(packed_text)
        padded_hiddens, _ = pad_packed_sequence(lstm_hiddens, batch_first=True)
        return torch.cat([padded_hiddens[i][:seq_lens[i], :] for i in range(0, len(seq_lens))], dim=0)


class LSTMDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate):
        super(LSTMDecoder, self).__init__()
        self.__input_dim = input_dim
        self.__hidden_dim = hidden_dim
        self.__output_dim = output_dim
        self.__dropout_rate = dropout_rate
        lstm_input_dim = self.__input_dim

        self.__dropout_layer = nn.Dropout(self.__dropout_rate)
        self.__lstm_layer = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=self.__hidden_dim,
            batch_first=True,
            bidirectional=False,
            dropout=self.__dropout_rate,
            num_layers=1
        )
        self.__linear_layer = nn.Linear(
            self.__hidden_dim,
            self.__output_dim
        )

    def forward(self, encoded_hiddens, seq_lens):
        input_tensor = encoded_hiddens
        output_tensor_list, sent_start_pos = [], 0
        for sent_i in range(0, len(seq_lens)):
            sent_end_pos = sent_start_pos + seq_lens[sent_i]
            seg_hiddens = input_tensor[sent_start_pos: sent_end_pos, :]
            combined_input = seg_hiddens
            dropout_input = self.__dropout_layer(combined_input)
            lstm_out, _ = self.__lstm_layer(dropout_input.view(1, seq_lens[sent_i], -1))
            linear_out = self.__linear_layer(lstm_out.view(seq_lens[sent_i], -1))
            output_tensor_list.append(linear_out)
            sent_start_pos = sent_end_pos
        import ipdb
        ipdb.set_trace()
        return torch.cat(output_tensor_list, dim=0)
