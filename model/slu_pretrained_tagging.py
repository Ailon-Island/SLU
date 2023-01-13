import os.path
from itertools import chain
import re
import numpy as np

import torch
from torch import nn
from torch.nn import Module
from torch.nn.functional import pad

from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM, load_tf_weights_in_electra

from .slu_baseline_tagging import TaggingFNNDecoder
from utils.tensor import dict2device
from utils.vocab import SEP, UNK, PAD

alphabet_tokens = [chr(i) for i in chain(range(97, 123), range(65, 91))]

model_names = {
    'bert': "bert-base-chinese",
    'electra': "hfl/chinese-electra-base-discriminator",
    'electra-180g': "hfl/chinese-electra-180g-base-discriminator"
}


class PretrainedTagging(Module):
    def __init__(self, config):
        super(PretrainedTagging, self).__init__()
        self.config = config
        self.name, self.tokenizer, self.model = self.get_pretrained(config)
        self.dropout_layer = nn.Dropout(p=config.dropout)
        self.output_layer = TaggingFNNDecoder(self.model.config.hidden_size, config.num_tags, config.tag_pad_idx)

    def forward(self, batch):
        tag_ids = batch.tag_ids
        tag_mask = batch.tag_mask
        input_ids = batch.input_ids
        utt = batch.utt
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
        prob, loss = self.forward(batch)

        labels = batch.labels
        utt = batch.utt

        predictions = label_vocab.decode(prob, utt)

        return predictions, labels, loss.cpu().item()

    @staticmethod
    def get_pretrained(config):
        name = model_names[config.pretrained_model.lower()]
        tokenizer = AutoTokenizer.from_pretrained(name,
                                                  sep_token=SEP,
                                                  unk_token=UNK,
                                                  pad_token=PAD,
                                                  additional_special_tokens=[' ', *alphabet_tokens])

        if config.load_pretrained:
            if config.pretrained_framework == 'pytorch':
                model = AutoModel.from_pretrained(name)
                state_dict = torch.load(config.pretrained_path, map_location=torch.device(config.device))
                model.load_state_dict(state_dict)
            elif config.pretrained_framework == 'tensorflow':
                if config.pretrained_model.lower() == 'electra':
                    model = AutoModelForMaskedLM.from_pretrained(name)
                    load_tf_weights_in_electra(
                        model, model.config, config.pretrained_path, discriminator_or_generator='generator'
                    )
                    model = model.electra
                else:
                    raise NotImplementedError
                pytorch_path = config.pretrained_path.replace('.ckpt', '.bin')
                if not os.path.exists(pytorch_path):
                    torch.save(model.state_dict(), pytorch_path)
                print(f"PyTorch checkpoint dumped for {name} at {pytorch_path}.")
            print(f"Local checkpoint loaded for {name} from {config.pretrained_path}.")
        else:
            model = AutoModel.from_pretrained(name)

        model.resize_token_embeddings(len(tokenizer))

        return name, tokenizer, model


