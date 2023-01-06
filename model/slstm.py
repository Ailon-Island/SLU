import torch
import torch.nn as nn
import torch.nn.functional as F
from fastNLP.modules.torch import decoder
from model.slu_baseline_tagging import TaggingFNNDecoder
from model.attention import SelfAttention


def TZ(*args):
    return torch.zeros(*args).cuda()


class HyperLinear(nn.Module):
    def __init__(self, ninp, nout, nz=10):
        super(HyperLinear, self).__init__()
        # self.zW = nn.Linear(nz, ninp*nout)
        self.zb = nn.Linear(nz, nout)
        self.iz = nn.Linear(ninp, nz)
        self.zz = nn.Linear(nz, nz * ninp)
        self.zW = nn.Linear(nz, nout)
        self.ninp, self.nout, self.nz = ninp, nout, nz

    def forward(self, data):
        z = self.iz(data)
        W = self.zW(self.zz(z).view(-1, self.ninp, self.nz))
        b = self.zb(z)
        data_size = list(data.size()[:-1]) + [self.nout]
        return (torch.matmul(data.view(-1, 1, self.ninp), W) + b.view(-1, 1, self.nout)).view(*data_size)

    def fake_forward(self, data):
        z = self.iz(data)
        W = self.zW(z).view(-1, self.ninp, self.nout)
        b = self.zb(z)
        data_size = list(data.size()[:-1]) + [self.nout]
        return (torch.matmul(data.view(-1, 1, self.ninp), W) + b.view(-1, 1, self.nout)).view(*data_size)


class SLSTM(nn.Module):
    def __init__(self, nvocab, nemb, nhid, num_layer, hyper=False, dropout=0.5, return_all=True):
        super(SLSTM, self).__init__()
        if hyper:
            self.n_fc = nn.Linear(4 * nhid + nemb, 2 * nhid)
            self.n_h_fc = HyperLinear(4 * nhid + nemb, 5 * nhid)
            self.g_out_fc = nn.Linear(2 * nhid, nhid)
            self.g_att_fc = nn.Linear(2 * nhid, nhid)
        else:
            self.n_fc = nn.Linear(5 * nhid, 7 * nhid)
            self.g_out_fc = nn.Linear(2 * nhid, nhid)
            self.g_att_fc = nn.Linear(2 * nhid, nhid)

        # self.emb = nn.Embedding.from_pretrained(Tar_emb)
        self.input = nn.Linear(nemb, nhid)
        self.fc = nn.Linear(2 * nhid, 2)
        self.up_fc = nn.Linear(nhid, 2 * nhid)

        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

        self.MP_ITER = 9
        self.nemb, self.nhid, self.num_layer, self.hyper = nemb, nhid, num_layer, hyper

        self._return_all = return_all

    def forward(self, data, mask):
        B, L, H = data.size()
        H = self.nhid

        def update_nodes(embs, nhs, ncs, gh, gc):
            ihs = torch.cat(
                [torch.cat([TZ(B, 1, H), nhs[:, :-1, :]], 1), nhs, torch.cat([nhs[:, 1:, :], TZ(B, 1, H)], 1), embs,
                 gh[:, None, :].expand(B, L, H)], 2)
            if self.hyper:
                xx = self.n_fc(ihs)
                og = torch.sigmoid(xx[:, :, :self.nhid])
                uh = torch.tanh(xx[:, :, self.nhid:2 * self.nhid])
                fs = self.n_h_fc(ihs)
                gs = F.softmax(fs.view(embs.size(0), embs.size(1), 5, self.nhid), 2)
            else:
                fs = self.n_fc(ihs)
                og = torch.sigmoid(fs[:, :, :self.nhid])
                uh = torch.tanh(fs[:, :, self.nhid:2 * self.nhid])
                gs = F.softmax(fs[:, :, self.nhid * 2:].view(embs.size(0), embs.size(1), 5, self.nhid), 2)

            ics = torch.stack(
                [torch.cat([TZ(B, 1, H), ncs[:, :-1, :]], 1), ncs, torch.cat([ncs[:, 1:, :], TZ(B, 1, H)], 1),
                 gc[:, None, :].expand(B, L, H), embs], 2)
            n_c = torch.sum(gs * ics, 2)
            n_nhs = og * torch.tanh(n_c)
            return n_nhs, n_c

        def update_g_node(nhs, ncs, gh, gc, mask):
            h_bar = nhs.sum(1) / mask.sum(1)[:, None]
            # h_bar = nhs.mean(1)
            ihs = torch.cat([h_bar[:, None, :], nhs], 1)
            ics = torch.cat([gc[:, None, :], ncs], 1)
            fs = self.g_att_fc(torch.cat([gh[:, None, :].expand(B, L + 1, H), ihs], 2))
            fs = fs + (1. - torch.cat([mask[:, :, None].expand(B, L, H), TZ(B, 1, H) + 1], 1)) * 200.0
            n_gc = torch.sum(F.softmax(fs, 1) * ics, 1)
            n_gh = torch.sigmoid(self.g_out_fc(torch.cat([gh, h_bar], 1))) * torch.tanh(n_gc)
            return n_gh, n_gc

        embs = self.drop1(data)
        embs = self.input(embs)
        # nhs = ncs = TZ(B,L,H)
        nhs = ncs = embs
        # gh = gc = TZ(B,H)
        gh = gc = embs.sum(1) / mask.sum(1)[:, None]
        for i in range(self.MP_ITER):
            n_gh, n_gc = update_g_node(nhs, ncs, gh, gc, mask)
            nhs, ncs = update_nodes(embs, nhs, ncs, gh, gc)
            nhs = mask[:, :, None].expand(B, L, H) * nhs
            ncs = mask[:, :, None].expand(B, L, H) * ncs
            gh, gc = n_gh, n_gc
        nhs = self.drop2(nhs)

        if self._return_all == False:
            return nhs

        rep = torch.cat([nhs, gh[:, None, :]], 1).sum(1) / mask.sum(1)[:, None]
        rep = self.drop2(rep)
        rep = torch.tanh(self.up_fc(rep))
        # rep = self.drop2(rep)
        pred = F.log_softmax(self.fc(rep), 1)
        return pred


class SeqLabelingForSLSTM(nn.Module):
    def __init__(self, config):
        super(SeqLabelingForSLSTM, self).__init__()
        vocab_size = config.vocab_size
        word_emb_dim = config.embed_size
        hidden_dim = config.hidden_size
        num_classes = config.num_tags
        self.word_embed = nn.Embedding(vocab_size, word_emb_dim, padding_idx=0)
        self.Rnn = SLSTM(vocab_size, word_emb_dim, hidden_dim, num_layer=1, return_all=False)
        # self.Linear = getattr(nn, self.cell)(config.embed_size, config.hidden_size // 2, num_layers=config.num_layer, bidirectional=True, batch_first=True)
        self.Crf = decoder.ConditionalRandomField(num_classes)
        self.output_layer = TaggingFNNDecoder(config.hidden_size, config.num_tags, config.tag_pad_idx)
        self.attention = SelfAttention(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=hidden_dim,
                                       dropout_rate=config.dropout)

    def _get_self_attn(self, src, mask):
        attn = self.attention(src, mask)
        attn += src
        return attn

    def forward(self, batch):
        input_ids = batch.input_ids
        tag_mask = batch.tag_mask
        tag_ids = batch.tag_ids
        x = self.word_embed(input_ids)
        x = self.Rnn(x, tag_mask)
        # x = self.Linear(x)
        attn = self._get_self_attn(x, tag_mask)
        x *= attn
        output = self.output_layer(x, tag_mask, tag_ids)
        return output

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
