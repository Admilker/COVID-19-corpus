import torch
from tensorflow import keras
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig, AdamW, WarmupLinearSchedule
import csv
# from main import load_dataset

import json
import numpy as np
import pandas as pd
import scipy.stats

# Bert模型以及相关配置
config = BertConfig.from_pretrained('bert-base-chinese')
config.num_labels = 3
model = BertForSequenceClassification(config=config)
model = BertForSequenceClassification.from_pretrained('bert-base-chinese', config=config)
# model.cuda()

# 加载数据集
def load_dataset(filepath):
    dataset_list = []
    indexs = []
    f = open(filepath, 'r', encoding='gbk')
    r = csv.reader(f, delimiter='\t')
    for item in r:
        items = item[0].split(",")
        dataset_list.append([items[0], items[1]])
        indexs.append([(items[2], items[3]), (items[4], items[5]), (items[6], items[7])])
    # print(dataset_list)
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    for item in dataset_list:
        item[0] = tokenizer.encode(item[0])
        item[0] = str(item[0])
    # print(dataset_list)
    return dataset_list, indexs

model.load_state_dict(torch.load('bert_cla2.ckpt'))
test_dataset, test_indexs = load_dataset('./data5/final.csv')

# model.eval()
#第二标签的处理部分
index = 0
L2 = []
for text, label in test_dataset:
    test_text = list(json.loads(text))
    # print(test_text)
    if label == '1':
        index += 1
        continue
    else:
        with torch.no_grad():
            E = []
            saliency = []
            saliency2 = []
            E_H = []


            # 人类标注的Token的重要性分布
            index1 = list(test_indexs[index][0])
            index2 = list(test_indexs[index][1])
            index3 = list(test_indexs[index][2])

            for i in range(len(test_text)):
                saliency.append(0)
            if index1[0] != '' and index1[1] != '':
                s1 = int(index1[0]) - 1
                e1 = int(index1[1]) - 1
                for i in range(s1, e1):
                    saliency[i] = 1
            if index2[0] != '' and index2[1] != '':
                s2 = int(index2[0]) - 1
                e2 = int(index2[1]) - 1
                # print(s2, e2)
                for i in range(s2, e2):
                    saliency[i] = 1
            if index3[0] != '' and index3[1] != '':
                s3 = int(index3[0]) - 1
                e3 = int(index3[1]) - 1
                for i in range(s3, e3):
                    saliency[i] = 1
            E_H.append(saliency)
            print('第{}个句子的人工标注的重要性分布：{}'.format(index, saliency))

            # 模型预测句子各个token的重要性分布
            for i in range(len(test_text)):
                saliency2.append(0)
            normal_text_tensor = torch.tensor(test_text).unsqueeze(0)
            label_tensor = torch.tensor(int(label))
            outputs = model(normal_text_tensor, labels=label_tensor)
            loss, logits = outputs[:2]
            # print(pre_normal)

            for i in range(len(test_text)):
                i_text = test_text[:i]+test_text[i+1:]
                i_tensor = torch.tensor(i_text).unsqueeze(0)
                i_outputs = model(i_tensor, labels=label_tensor)
                i_loss, i_logits = i_outputs[:2]
                saliency2[i] = (loss.item()-i_loss.item())
            E.append(saliency2)
            print('第{}个句子的模型预测的重要性分布：{}'.format(index, saliency2))

            # 求两个重要性分布的第二范数
            sal = [saliency[i] - saliency2[i] for i in range(len(saliency))]
            l2 = np.linalg.norm(sal)
            print('第{}个句子的两个重要性分布的L2范数是：{}'.format(index, l2))
            L2.append(l2)
            # print(L2)

            print('\n')
    index += 1
print(L2)





