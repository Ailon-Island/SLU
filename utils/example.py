from functools import reduce
from copy import deepcopy
import json

import numpy as np

from utils.vocab import Vocab, LabelVocab, LabelVocabNBI, SEP
from utils.word2vec import Word2vecUtils
from utils.evaluator import Evaluator

class Example():

    @classmethod
    def configuration(cls, root, train_path=None, word2vec_path=None, tag_bi=True, sentence=False):
        cls.evaluator = Evaluator()
        cls.word_vocab = Vocab(padding=True, unk=True, sentence=sentence, filepath=train_path)
        cls.word2vec = Word2vecUtils(word2vec_path)
        cls.label_vocab = LabelVocab(root) if tag_bi else LabelVocabNBI(root)

    @classmethod
    def load_dataset(cls, data_path):
        datas = json.load(open(data_path, 'r'))
        examples = []
        for data in datas:
            for utt in data:
                ex = cls(utt)
                examples.append(ex)
        return examples

    @classmethod
    def load_dialogue_dataset(cls, data_path):
        datas = json.load(open(data_path, 'r'))
        examples = []
        for data in datas:
            dummpy_data = {
                "utt_id": 1,
                "manual_transcript": "",
                "asr_1best": "",
                "semantic": [],
            }
            ex = cls(data[0])
            length = lambda: len(ex.input_idx)
            ex.ex = [ex.ex]
            ex.slot = [ex.slot]
            ex.mask = [1] * length()
            examples.append(deepcopy(ex))
            for utt in data[1:]:
                cur_ex = cls(utt)
                ex.ex += [cur_ex.ex]
                ex.mask = [0] * (length() + 1)
                ex.mask += [1] * len(cur_ex.input_idx)
                ex.slot = cur_ex.slot
                ex.slotvalue = cur_ex.slotvalue
                ex.tags = ['O'] * (length() + 1) + cur_ex.tags
                ex.tag_id = [0] * (length() + 1) + cur_ex.tag_id
                ex.utt += SEP + cur_ex.utt
                ex.input_idx += [Example.word_vocab[SEP]] + cur_ex.input_idx

                examples.append(deepcopy(ex))

        return examples

    def __init__(self, ex: dict):
        super(Example, self).__init__()
        self.ex = ex

        self.utt = ex['asr_1best']
        if self.utt == 'null':
            self.utt == ''
        self.slot = {}
        for label in ex['semantic']:
            act_slot = f'{label[0]}-{label[1]}'
            if len(label) == 3:
                self.slot[act_slot] = label[2]
        self.tags = ['O'] * len(self.utt)
        for slot in self.slot:
            value = self.slot[slot]
            bidx = self.utt.find(value)
            if bidx != -1:
                self.tags[bidx: bidx + len(value)] = [f'I-{slot}'] * len(value)
                self.tags[bidx] = f'B-{slot}'
        self.slotvalue = [f'{slot}-{value}' for slot, value in self.slot.items()]
        self.input_idx = [Example.word_vocab[c] for c in self.utt]
        l = Example.label_vocab
        self.tag_id = [l.convert_tag_to_idx(tag) for tag in self.tags]
