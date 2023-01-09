from functools import reduce

import json

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
            exps = []
            for utt in data:
                ex = cls(utt)
                exps.append(ex)
            ex.ex = [ex.ex for ex in exps]  # simple cast to list
            ex.slot = [ex.slot for ex in exps]  # simple cast to list
            ex.slotvalue = reduce(lambda x, y: x + y, [ex.slotvalue for ex in exps])
            ex.utt = reduce(lambda x, y: x + SEP + y, [ex.utt for ex in exps])
            ex.input_idx = reduce(lambda x, y: x + [Example.word_vocab[SEP]] + y, [ex.input_idx for ex in exps])
            ex.tags = reduce(lambda x, y: x + ['O'] + y, [ex.tags for ex in exps])
            ex.tag_id = reduce(lambda x, y: x + [Example.label_vocab.convert_tag_to_idx('O')] + y, [ex.tag_id for ex in exps])
            examples.append(ex)

        return examples

    def __init__(self, ex: dict):
        super(Example, self).__init__()
        self.ex = ex

        self.utt = ex['asr_1best']
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
