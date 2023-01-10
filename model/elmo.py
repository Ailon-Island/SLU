import torch
from torch import nn
from elmoformanylangs import Embedder

e = Embedder('./pretrain/zhs.model')
# sents = [["为什么是这样", "好"], ["什", "么"], ["我", "在", "干", "啥"]]
sents = [["这是", "什么"]]

res = e.sents2elmo(sents)
print(res, res[0].shape)

# class Elmo(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.bilstm = nn.LSTM(self.embedding_dim, self.hidden_dim // 2,
#                             num_layers=2, bidirectional=True, batch_first=True)