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

class sLSTMCell(nn.Module):
    def __init__(self, d_word, d_hidden, n_windows,
                 n_sent_nodes, bias, batch_first,
                 init_method='normal'):
        super().__init__()
        self.d_input = d_word
        self.d_hidden = d_hidden
        self.n_windows = n_windows
        self.num_g = n_sent_nodes
        self.initial_method = init_method
        self.bias = bias
        self.batch_first = batch_first
        self.lens_dim = 1 if batch_first is True else 0
        self._all_gate_weights = []
        word_gate_dict = dict(
            [('input_gate', 'i'), ('left_forget_gate', 'l'),
             ('right_forget_gate', 'r'), ('forget_gate', 'f'),
             ('sentence_forget_gate', 's'), ('output_gate', 'o'),
             ('recurrent_input', 'u')])

        for (gate_name, gate_tag) in word_gate_dict.items():
            w_w = nn.Parameter(torch.Tensor(d_hidden,
                                            (n_windows * 2 + 1) * d_hidden))
            w_u = nn.Parameter(torch.Tensor(d_hidden, d_word))
            w_v = nn.Parameter(torch.Tensor(d_hidden, d_hidden))
            w_b = nn.Parameter(torch.Tensor(d_hidden))

            gate_params = (w_w, w_u, w_v, w_b)
            param_names = ['w_w{}', 'w_u{}', 'w_v{}', 'w_b{}']
            param_names = [x.format(gate_tag) for x in param_names]  # {
            for name, param in zip(param_names, gate_params):
                setattr(self, name, param)  # self.name = param
            self._all_gate_weights.append(param_names)

        sentence_gate_dict = dict(
            [('sentence_forget_gate', 'g'), ('word_forget_gate', 'f'),
             ('output_gate', 'o')])

        for (gate_name, gate_tag) in sentence_gate_dict.items():
            s_w = nn.Parameter(torch.Tensor(d_hidden, d_hidden))
            s_u = nn.Parameter(torch.Tensor(d_hidden, d_hidden))
            s_b = nn.Parameter(torch.Tensor(d_hidden))
            gate_params = (s_w, s_u, s_b)
            param_names = ['s_w{}', 's_u{}', 's_b{}']
            param_names = [x.format(gate_tag) for x in param_names]
            for name, param in zip(param_names, gate_params):
                setattr(self, name, param)
            self._all_gate_weights.append(param_names)
        self.reset_parameters(self.initial_method)

    def reset_parameters(self, init_method):
        if init_method is 'normal':
            std = 0.1
            for weight in self.parameters():
                weight.data.normal_(mean=0.0, std=std)
        else:  # uniform
            stdv = 1.0 / math.sqrt(self.d_hidden)
            for weight in self.parameters():
                weight.data.uniform_(-stdv, stdv)

    def sequence_mask(self, size, length):  # ???
        mask = torch.LongTensor(range(size[0])).view(size[0], 1).cuda()  # (l,1)
        length = length.squeeze(dim=1)  # (b)
        result = (mask >= length).unsqueeze(dim=2)  # (l,b,1)
        return result

    def in_window_context(self, hx, window_size=1, average=False):
        slices = torch.unbind(hx, dim=0)  # torch.size([18,32,256]) -> ([32,256]) * 18
        zeros = torch.unbind(torch.zeros_like(hx), dim=0)

        context_l = [torch.stack(zeros[:i] + slices[:len(slices) - i], dim=0)
                     for i in range(window_size, 0, -1)]
        context_l.append(hx)
        context_r = [torch.stack(slices[i + 1: len(slices)] + zeros[:i + 1], dim=0)
                     for i in range(0, window_size)]

        context = context_l + context_r
        return torch.stack(context).mean(dim=0) if average \
            else torch.cat(context, dim=2)

    def forward(self, src_seq, src_len, state=None):
        seq_mask = self.sequence_mask(src_seq.size(), src_len)
        h_gt_1 = state[0][-self.num_g:]
        h_wt_1 = state[0][:-self.num_g].masked_fill(seq_mask, 0)
        c_gt_1 = state[1][-self.num_g:]
        c_wt_1 = state[1][:-self.num_g].masked_fill(seq_mask, 0)
        h_hat = h_wt_1.mean(dim=0)
        fg = torch.sigmoid(F.linear(h_gt_1, self.s_wg) +
                           F.linear(h_hat, self.s_ug) +
                           self.s_bg)
        o = torch.sigmoid(F.linear(h_gt_1, self.s_wo) +
                          F.linear(h_hat, self.s_uo) + self.s_bo)
        fi = torch.sigmoid(F.linear(h_gt_1, self.s_wf) +
                           F.linear(h_wt_1, self.s_uf) +
                           self.s_bf).masked_fill(seq_mask, -1e25)
        fi_normalized = F.softmax(fi, dim=0)
        c_gt = fg.mul(c_gt_1).add(fi_normalized.mul(c_wt_1).sum(dim=0))
        h_gt = o.mul(torch.tanh(c_gt))
        epsilon = self.in_window_context(h_wt_1, window_size=self.n_windows)
        i = torch.sigmoid(F.linear(epsilon, self.w_wi) +
                          F.linear(src_seq, self.w_ui) +
                          F.linear(h_gt_1, self.w_vi) + self.w_bi)
        l = torch.sigmoid(F.linear(epsilon, self.w_wl) +
                          F.linear(src_seq, self.w_ul) +
                          F.linear(h_gt_1, self.w_vl) + self.w_bl)
        r = torch.sigmoid(F.linear(epsilon, self.w_wr) +
                          F.linear(src_seq, self.w_ur) +
                          F.linear(h_gt_1, self.w_vr) + self.w_br)
        f = torch.sigmoid(F.linear(epsilon, self.w_wf) +
                          F.linear(src_seq, self.w_uf) +
                          F.linear(h_gt_1, self.w_vf) + self.w_bf)
        s = torch.sigmoid(F.linear(epsilon, self.w_ws) +
                          F.linear(src_seq, self.w_us) +
                          F.linear(h_gt_1, self.w_vs) + self.w_bs)
        o = torch.sigmoid(F.linear(epsilon, self.w_wo) +
                          F.linear(src_seq, self.w_uo) +
                          F.linear(h_gt_1, self.w_vo) + self.w_bo)
        u = torch.tanh(F.linear(epsilon, self.w_wu) +
                       F.linear(src_seq, self.w_uu) +
                       F.linear(h_gt_1, self.w_vu) + self.w_bu)
        gates = torch.stack((l, f, r, s, i), dim=0)
        gates_normalized = F.softmax(gates.masked_fill(seq_mask, -1e25), dim=0)
        c_wt_l, c_wt_1, c_wt_r = \
            self.in_window_context(c_wt_1).chunk(3, dim=2)
        c_mergered = torch.stack((c_wt_l, c_wt_1, c_wt_r,
                                  c_gt_1.expand_as(c_wt_1.data), u), dim=0)

        c_wt = gates_normalized.mul(c_mergered).sum(dim=0)
        c_wt = c_wt.masked_fill(seq_mask, 0)
        h_wt = o.mul(torch.tanh(c_wt))

        h_t = torch.cat((h_wt, h_gt), dim=0)
        c_t = torch.cat((c_wt, c_gt), dim=0)
        return (h_t, c_t)


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
        self.cell = sLSTMCell(d_word=config.embed_size, d_hidden=config.hidden_size,
                              n_windows=3, n_sent_nodes=1, bias=True, batch_first=True)
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
        # embed = self.word_embed(input_ids)

        utt = batch.utt
        utt_length = len(utt[0])
        embed = torch.Tensor(np.stack([np.pad(res, ((0, utt_length - res.shape[0]),(0, 0))) for res in self.embedder.sents2elmo(utt)])).to('cuda')
        h_t = torch.zeros(embed.size(0) + self.sentence_node, embed.size(1), self.d_hidden).cuda()
        c_t = torch.zeros_like(h_t).cuda()
        src_len = torch.LongTensor(np.array(self.get_len(input_ids.cpu().transpose(0, 1)))).unsqueeze(1).cuda()
        for step in range(self.steps):
            h_t, c_t = self.cell(embed, src_len, (h_t, c_t))
        h_t = self.dropout_layer(h_t)
        hidden = h_t[:-self.sentence_node]
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
