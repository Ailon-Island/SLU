from functools import reduce
from copy import deepcopy
import json

import numpy as np

from utils.vocab import Vocab, LabelVocab, LabelVocabNBI, SEP
from utils.word2vec import Word2vecUtils
from utils.evaluator import Evaluator
import pycorrector
# import jieba

class Example():

    @classmethod
    def configuration(cls, root, train_path=None, word2vec_path=None, tag_bi=True, sentence=False):
        cls.evaluator = Evaluator()
        cls.word_vocab = Vocab(padding=True, unk=True, sentence=sentence, filepath=train_path)
        cls.word2vec = Word2vecUtils(word2vec_path)
        cls.label_vocab = LabelVocab(root) if tag_bi else LabelVocabNBI(root)

    @classmethod
    def load_dataset(cls, data_path, cut=False, correct=False):
        datas = json.load(open(data_path, 'r'))
        examples = []
        for di, data in enumerate(datas):
            for ui, utt in enumerate(data):
                ex = cls(utt, f'{di}-{ui}', cut,)
                examples.append(ex)
        return examples

    @classmethod
    def load_dialogue_dataset(cls, data_path):
        datas = json.load(open(data_path, 'r'))
        examples = []
        for di, data in enumerate(datas):
            ex = cls(data[0], f'{di}-0')
            length = lambda: len(ex.input_idx)
            ex.ex = [ex.ex]
            ex.slot = [ex.slot]
            ex.mask = [1] * length()
            examples.append(deepcopy(ex))
            for ui, utt in enumerate(data[1:], 1):
                cur_ex = cls(utt, f'{di}-{ui}')
                ex.ex += [cur_ex.ex]
                ex.did = cur_ex.did
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
        
    def __init__(self, ex: dict, did):
        for data in datas:
            for utt in data:
                ex = cls(utt, cut, correct)
                # print(ex.slotvalue)
                examples.append(ex)
        return examples

    def __init__(self, ex: dict, did, cut=False, correct=False):
        super(Example, self).__init__()
        self.ex = ex
        self.did = did

        self.utt = ex['asr_1best']
        if self.utt == 'null':
            self.utt == ''
        if correct == True:
            self.utt, _ = pycorrector.correct(self.utt)
            print(self.utt)
        # if cut == True:
        #     self.utt = list(jieba.cut(self.utt))
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
