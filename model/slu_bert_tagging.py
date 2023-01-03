from itertools import chain
import re
import numpy as np

import torch
from torch import nn
from torch.nn import Module
from torch.nn.functional import pad

from transformers import AutoTokenizer, AutoModel

from .slu_baseline_tagging import TaggingFNNDecoder
from utils.tensor import dict2device

alphabet_tokens = ['['+chr(i)+']' for i in chain(range(97, 123), range(65, 91))]


class BERTTagging(Module):
    def __init__(self, config):
        super(BERTTagging, self).__init__()
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese",
                                                       additional_special_tokens=['[SPACE]', '[a]', *alphabet_tokens])
        self.model = AutoModel.from_pretrained("bert-base-chinese")
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.dropout_layer = nn.Dropout(p=config.dropout)
        self.output_layer = TaggingFNNDecoder(self.model.config.hidden_size, config.num_tags, config.tag_pad_idx)

    def forward(self, batch):
        tag_ids = batch.tag_ids
        tag_mask = batch.tag_mask
        input_ids = batch.input_ids
        utt = batch.utt
        utt = [re.sub('[a-zA-Z]', lambda x: '['+x.group(0)+']', ut) for ut in utt]
        utt = [re.sub(' ', '[SPACE]', ut) for ut in utt]
        lengths = batch.lengths

        token = self.tokenizer(utt, return_tensors="pt", padding=True, truncation=True)
        token = dict2device(token, self.config.device)
        outputs = self.model(**token)
        out = outputs['last_hidden_state']
        # need to omit bos and eos
        out = out[:, 1:-1, :] * tag_mask.unsqueeze(-1)
        hiddens = self.dropout_layer(out)
        tag_output = self.output_layer(hiddens, tag_mask, tag_ids)

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
