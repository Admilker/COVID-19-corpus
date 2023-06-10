#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# !/usr/bin/env python
# -*- coding: UTF-8 -*-
import torch
import pickle as pkl
import numpy as np
from importlib import import_module

key = {
    0: 0, #'积极'
    1: 1, #'中立'
    2: 2  #'消极'
}


class Predict:
    def __init__(self, model_name='TextCNN', dataset='text', embedding='embedding_SougouNews.npz', use_word=False):
        if use_word:
            self.tokenizer = lambda x: x.split(' ')  # 以空格隔开，word-level
        else:
            self.tokenizer = lambda x: [y for y in x]  # char-level
        self.x = import_module('models.' + model_name)
        self.config = self.x.Config(dataset, embedding)
        self.vocab = pkl.load(open(self.config.vocab_path, 'rb'))
        self.pad_size = self.config.pad_size
        self.model = self.x.Model(self.config).to('cpu')
        self.model.load_state_dict(torch.load(self.config.save_path, map_location='cuda:0'))

    def build_predict_text(self, texts):
        words_lines = []
        seq_lens = []
        for text in texts:
            words_line = []
            token = self.tokenizer(text)
            seq_len = len(token)
            if self.pad_size:
                if len(token) < self.pad_size:
                    token.extend(['<PAD>'] * (self.pad_size - len(token)))
                else:
                    token = token[:self.pad_size]
                    seq_len = self.pad_size
            # word to id
            for word in token:
                words_line.append(self.vocab.get(word, self.vocab.get('<UNK>')))
            words_lines.append(words_line)
            seq_lens.append(seq_len)

        return torch.LongTensor(words_lines), torch.LongTensor(seq_lens)

    def predict(self, query):
        query = [query]
        # 返回预测的索引
        data = self.build_predict_text(query)
        with torch.no_grad():
            outputs = self.model(data)
            num = torch.argmax(outputs)
        return key[int(num)]

    def predict_list(self, querys):
        # 返回预测的索引
        data = self.build_predict_text(querys)
        with torch.no_grad():
            outputs = self.model(data)
            num = torch.argmax(outputs, dim=1)
            pred = [key[index] for index in list(np.array(num))]
        return pred


if __name__ == "__main__":
    pred = Predict('TextCNN')
    # 预测一条
    query = "疫情结束"
    print(pred.predict(query))
    # 预测一个列表
    querys = ["不能上学了", "一切回归正常吧！魔幻的十一月终将结束！", "我没打过疫苗，现在都还没阳过，所以还是很少去人多的地方，出门戴口罩，回家喷酒精。","快快好起来吧。"]
    print(pred.predict_list(querys))