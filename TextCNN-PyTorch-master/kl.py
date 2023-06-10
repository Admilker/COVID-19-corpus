import math
from importlib import import_module

import scipy
import torch
import pandas as pd
import openpyxl
import jieba
import re
import numpy as np
import torch.nn.functional as F

import scipy.stats

import pickle as pkl
from keras_preprocessing.sequence import pad_sequences

from predict import Predict

model = torch.load("model.h5")

with open('./text/data/vocab.pkl', 'rb') as f:
    dictionary = pkl.load(f)


pred = Predict('TextCNN')

data = pd.read_excel('./1900.xlsx')

total = []
x_nor_all =[]
for index, row in data.iterrows():
    # if(index < 1888): continue
    text = str(row['text'])
    label = str(row['label'])
    label = label.split(',')
    #print(str(index) + "  "+text)
    tag = int(label[0])
    true = []
    if tag == 1:
        continue
    if tag == 0:
        true = [1, 0, 0]
    if tag == 2:
        true = [0, 0, 1]
    start_pos = int(label[1])
    end_pos = int(label[2])
    true = torch.FloatTensor(true)
    #
    #     使用jieba进行分词
    #     text = re.sub('([^\u4e00-\u9fa5\u0030-\u0039])', ' ', text)
    tokens = " ".join(jieba.cut(text)).split()
    # label = []
    # 将分词后的句子依次去除词语并转换为向量，作为模型预测输入
    sentence = []
    for j in range(len(tokens)):
        temp = tokens.copy()
        del temp[j]
        word = [dictionary[word] if word in dictionary else 0 for word in temp]  # 将词转换为对应的索引
        sentence.append(word)
    orig = [dictionary[word] if word in dictionary else 0 for word in tokens]  # 完整句子
    sentence.append(orig)
    sentence = pad_sequences(maxlen=320, sequences=sentence, padding='post', value=0)

    # 模型预测结果(根据自己模型所需输入更改)
    predict_label = []
    y = []
    M = sentence.shape[0]
    for start, end in zip(range(0, M, 1), range(1, M + 1, 1)):
        # sentence = [dictionary[i] for i in x[start] if i != 0]
        # y_predict = model.predict(sentence[start:end])

        # 添加batch维度
        input_tensor = torch.tensor(sentence[start:end], dtype=torch.long,device='cuda:0').unsqueeze(0)
        y_predict = model(input_tensor)

        label_predict = np.argmax(y_predict.detach().cpu().numpy()[0])
        predict_label.append(label_predict)
        y.append(y_predict[0])

    # 格式上的转换，y_0是原始文本的分类结果
    y_0 = y.pop()
    y_0 = torch.tensor(y_0).unsqueeze(0)

    true = torch.tensor([predict_label.pop()],device='cuda:0') # torch.FloatTensor(true)
    loss_all = F.cross_entropy(y_0, true)

    # 获取文本分词后向量，用于计算散度
    sum = 0
    get_index = []
    for i in tokens:
        tmp = len(i)
        sum = sum + tmp
        if start_pos < sum <= end_pos:
            get_index.append(1)
        else:
            get_index.append(0)

    # 预测结果转tensor
    y_pre = y
    loss = []
    # y_pre = torch.tensor(y_pre)

    # 求去除单个词后的loss
    for i in y_pre:
        i = torch.tensor(i).unsqueeze(0)
        res = F.cross_entropy(i, true)
        res = res - loss_all
        res = round(abs(res.item()),4) #小数位数
        # res = res.item()
        loss.append(res)
    # print(loss)

    # 求第二范式
    x = np.array(get_index) - np.array(loss)
    x_nor = round(np.linalg.norm(x, axis=None, keepdims=False),4)
    # print(x_nor)

    # kl散度
    KL = round(scipy.stats.entropy(get_index, loss),4)
    # print(KL)
    if math.isnan(KL) or math.isinf(KL):
        print("KL Error: ")
        print(KL)
        print("#No.%s, Text=%s"%(index+1,text))
        print("#分词:")
        print(tokens)
        continue

    total.append(KL)
    x_nor_all.append(x_nor)
    if 1:
        # 打印有效计算的数据信息
        print("************************************************************")
        print("#No.%s           #标记：%s           #标签：%s " %(index+1,label[0],row['label']))
        print("#内容：%s " %text)
        print("#分词:")
        print(tokens)
        print("#交叉熵：")
        print(loss)
        print("#第二范式值：%.4f;            #KL值：%.4f" %(x_nor,KL))
        print("************************************************************")
        print("")

print("")
print("数据总数: %d"%len(data))
print("计算总数: %d"%len(total))
print("----------------------验证结果-----------------------------")
print("第二范式值序列:")
print(x_nor_all)
print("离散度序列:")
print(total)
print("离散度均值:")
print(format(np.mean(total), '.4f'))
# print(format(np.sum(total) / len(total), '.4f'))
print("----------------------------------------------------------")

print("结束。")