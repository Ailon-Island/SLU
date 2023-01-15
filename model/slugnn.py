import torch_geometric
import torch
from torch_geometric.nn import MessagePassing, SAGEConv, global_mean_pool, aggr
from model.attention import SelfAttention
from model.slu_baseline_tagging import TaggingFNNDecoder, CRFDecoder
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import scipy
import numpy as np
from torch_geometric.utils import add_self_loops, sort_edge_index
from torch_scatter import scatter
from elmoformanylangs import Embedder


class GNN(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr="mean")
        self.in_channels = in_channels
        self.out_channels = out_channels
        # self.attention = SelfAttention(in_channels, in_channels, in_channels, )
        self.Linear = nn.Linear(in_channels, out_channels)
        self.gru = torch.nn.GRU(self.out_channels, self.out_channels)

    def message(self, x_j):
        return self.Linear(x_j)

    def update(self, h, x):
        x = self.Linear(x)
        output, h_hat = self.gru(torch.stack([h,x],dim=0))
        return torch.squeeze(h_hat)

    def forward(self, x, edge_index, offset=None):
        edge_index, _ = add_self_loops(edge_index._indices(), num_nodes=x.size(0))
        x = self.propagate(edge_index=sort_edge_index(edge_index), size=(x.size(0), x.size(0)), x=x)
        # x = global_mean_pool(x,offset)
        return x

class SLU_GNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        vocab_size = config.vocab_size
        embed_size = config.embed_size
        hidden_size = 2 * config.hidden_size
        output_size = config.hidden_size
        num_tags = config.num_tags
        self.embedder = Embedder('./pretrain/zhs.model')
        # self.word_embed = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.lstm = nn.LSTM(embed_size, hidden_size//2, config.num_layer, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(config.dropout)
        self.gnn = GNN(hidden_size, output_size)
        # self.output_layer = TaggingFNNDecoder(output_size, num_tags, config.tag_pad_idx)
        self.output_layer = CRFDecoder(output_size, num_tags, config.tag_pad_idx)

    def get_edge(self, embed, lengths):
        batch, length, emb = embed.shape
        nodes = []
        edges = []
        offset = [0]
        for i in range(batch):
            lens = lengths[i]
            nodes.append(embed[i,:lens,:])
            edges.append((np.fromfunction(lambda i,j:i==j-1, shape=(lens, lens))
                    + np.fromfunction(lambda i,j:j==i-1, shape=(lens,lens))
                    + np.fromfunction(lambda i,j:j==i-2, shape=(lens,lens))
                    + np.fromfunction(lambda i,j:i==j-2, shape=(lens,lens))).astype(int))
            offset.append(offset[-1]+lens)
        nodes = torch.concat(nodes, dim=0)
        edges = scipy.sparse.block_diag(mats=edges)
        indices = torch.from_numpy(np.vstack((edges.row, edges.col)).astype(np.int64))
        values = torch.from_numpy(edges.data)
        shape = torch.Size(edges.shape)
        edges = torch.sparse.FloatTensor(indices, values, shape)
        offset = torch.tensor(offset)
        return nodes.cuda(), edges.cuda(), offset.cuda()

    def reshape_output(self, gnn_output, length):
        out = []
        begin = 0
        end = 0
        for i in range(len(length)):
            end += length[i]
            out.append(gnn_output[begin:end])
            begin = end
        return  rnn_utils.pad_sequence(out, batch_first=True)

    def forward(self, batch):
        tag_ids = batch.tag_ids
        tag_mask = batch.tag_mask
        input_ids = batch.input_ids
        lengths = batch.lengths
        utt = batch.utt
        # embed = self.word_embed(input_ids)
        utt_length = len(utt[0])
        embed = torch.Tensor(np.stack([np.pad(res, ((0, utt_length - res.shape[0]),(0, 0))) for res in self.embedder.sents2elmo(utt)])).to('cuda')
        packed_inputs = rnn_utils.pack_padded_sequence(embed, lengths, batch_first=True, enforce_sorted=True)
        packed_outputs, _ = self.lstm(packed_inputs)
        lstm_out, _ = rnn_utils.pad_packed_sequence(packed_outputs, batch_first=True)
        hiddens = self.dropout(lstm_out)

        node, edge, offset = self.get_edge(hiddens, lengths)
        gnn_output = self.gnn(node, edge, offset)
        output = self.reshape_output(gnn_output, lengths)
        result = self.output_layer(output, tag_mask, tag_ids)
        return result

    def decode(self, label_vocab, batch):
        batch_size = len(batch)
        labels = batch.labels
        prob, loss = self.forward(batch)
        predictions = []
        for i in range(batch_size):
            pred = prob[i]
            # pred = torch.argmax(prob[i], dim=-1).cpu().tolist()
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