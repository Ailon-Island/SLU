from torch import nn
import torch
import torch.nn.functional as F
import math


class QKVAttention(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, hidden_dim, output_dim, dropout_rate):
        super(QKVAttention, self).__init__()
        self.__query_dim = query_dim
        self.__key_dim = key_dim
        self.__value_dim = value_dim
        self.__hidden_dim = hidden_dim
        self.__output_dim = output_dim
        self.__dropout_rate = dropout_rate
        self.__query_layer = nn.Linear(self.__query_dim, self.__hidden_dim)
        self.__key_layer = nn.Linear(self.__key_dim, self.__hidden_dim)
        self.__value_layer = nn.Linear(self.__value_dim, self.__output_dim)
        self.__dropout_layer = nn.Dropout(p=self.__dropout_rate)

    def forward(self, input_query, input_key, input_value):
        linear_query = self.__query_layer(input_query)
        linear_key = self.__key_layer(input_key)
        linear_value = self.__value_layer(input_value)

        score_tensor = F.softmax(torch.matmul(
            linear_query,
            linear_key.transpose(-2, -1)
        ) / math.sqrt(self.__hidden_dim), dim=-1)
        forced_tensor = torch.matmul(score_tensor, linear_value)
        forced_tensor = self.__dropout_layer(forced_tensor)
        return forced_tensor


class SelfAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate):
        super(SelfAttention, self).__init__()
        self.__input_dim = input_dim
        self.__hidden_dim = hidden_dim
        self.__output_dim = output_dim
        self.__dropout_rate = dropout_rate
        self.__dropout_layer = nn.Dropout(self.__dropout_rate)
        self.__attention_layer = QKVAttention(
            self.__input_dim, self.__input_dim, self.__input_dim,
            self.__hidden_dim, self.__output_dim, self.__dropout_rate
        )

    def forward(self, input_x, seq_lens, is_flat=False):
        dropout_x = self.__dropout_layer(input_x)
        attention_x = self.__attention_layer(
            dropout_x, dropout_x, dropout_x
        )
        if is_flat:
            flat_x = torch.cat(
                [attention_x[i][:seq_lens[i], :] for
                 i in range(0, len(seq_lens))], dim=0
            )
            return flat_x
        return attention_x
