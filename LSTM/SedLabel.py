import scipy
import torch
import pandas as pd
import jieba
import re
import numpy as np
import torch.nn.functional as F

import pickle as pkl
from tensorflow import keras
from keras_preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader
from LSTM import lstm

import matplotlib.pyplot as plt

PAD = '<PAD>'

# 保存的模型和构建的词表
model = keras.models.load_model('D:\\Program\\save_model.h5')
print(model)
ckpt_model = torch.load('savemodel/LSTM.ckpt')

# with open('word_dict.pk', 'rb') as f:
with open('dataset/word2index', 'rb') as f:
    dictionary = pkl.load(f)

with open('../dic_dict.pk', 'rb') as f:
    invert = pkl.load(f)

data = pd.read_excel('./LSTM.xlsx')

total = []

kl_res = []
item = []
FS = []
for index, row in data.iterrows():
    if index>1078:
        break
    text = str(row['text'])
    tag = int(row['label'])

    # label = str(row['label'])
    # label = label.split(',')
    # tag = int(label[0])
    true = []
    if tag == 1:
        continue
    if tag == 0:
        true = [1, 0, 0]
    if tag == 2:
        true = [0, 0, 1]
    # start_pos = int(label[1])
    # end_pos = int(label[2])
    start_pos = int(row['start'])
    end_pos = int(row['end'])
    true = torch.FloatTensor(true)
    # item.append()
#
#     使用jieba进行分词
#     text = re.sub('([^\u4e00-\u9fa5\u0030-\u0039])', ' ', text)
    tokens = " ".join(jieba.cut(text)).split()

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


    # 将分词后的句子依次去除词语并转换为向量，作为模型预测输入
    sentence = []
    pad_size = 140
    for j in range(len(tokens)):
        temp = tokens.copy()
        del temp[j]
        if len(temp) < pad_size:
            temp.extend([PAD] * (pad_size - len(temp)))
        else:
            temp = temp[:pad_size]
        word = [dictionary[word] if word in dictionary else 0 for word in temp]
        sentence.append(word)
    if len(tokens)< pad_size:
        tokens.extend([PAD] * (pad_size - len(tokens)))
    else:
        tokens = tokens[:pad_size]
    orig = [dictionary[word] if word in dictionary else 0 for word in tokens]
    sentence.append(orig)
    # sentence = pad_sequences(maxlen=140, sequences=sentence, padding='post', value=0)



    # ckpt模型
    Loss = []
    sentence = TensorDataset(torch.tensor(sentence))
    sentence = DataLoader(sentence)
    model_par = test.LSTM(len(dictionary))
    model_par.load_state_dict(ckpt_model)
    model_par.eval()
    # model_par.to('cuda')
    # sentence = sentence.to('cuda')
    with torch.no_grad():
        for text in sentence:
            # text = text.to('cuda')
            # text = torch.tensor(text).cuda()
            output = model_par(text[0])
            output = output[0][0]
            loss = F.cross_entropy(output, true)
            Loss.append(loss)
    bench = Loss[-1]
    cz = []
    for i in Loss:
        chazhi = abs(i - bench)
        cz.append(chazhi)
    cz.pop()

    # # 模型预测结果(根据自己模型所需输入更改)
    # predict_label = []
    # y = []
    # M = sentence.shape[0]
    # for start, end in zip(range(0, M, 1), range(1, M + 1, 1)):
    #     # sentence = [dictionary[i] for i in x[start] if i != 0]
    #     y_predict = model.predict(sentence[start:end])
    #     label_predict = np.argmax(y_predict[0])
    #     predict_label.append(label_predict)
    #     y.append(y_predict[0])
    #
    #
    # # 格式上的转换，y_0是原始文本的分类结果
    # y_0 = y.pop()
    # y_0 = torch.tensor(y_0)
    # loss_all = F.cross_entropy(y_0, true)



    # # 预测结果转tensor
    # y_pre = y
    # loss = []
    # y_pre = torch.tensor(y_pre)
    #
    # # 求去除单个词后的loss
    # for i in y_pre:
    #     res = F.cross_entropy(i, true)
    #     res = res - loss_all
    #     res = abs(res.item())
    #     # res = res.item()
    #     loss.append(res)
    # print(loss)

    # 求第二范式
    x = np.array(get_index) - np.array(cz)
    x_nor = np.linalg.norm(x, axis=None, keepdims=False)
    FS.append(x_nor)
    print("第%d条的范数：%.4f" %(index, x_nor))

    # kl散度

    # cz = cz / np.sum(cz)
    # get_index = get_index / np.sum(get_index)
    # KL = scipy.stats.entropy(get_index, cz)
    # KL2 = scipy.stats.entropy(cz, get_index)
    # # KL_2 = scipy.stats.entropy(loss, get_index)
    # kl_res.append(KL)
    # print("第%d条的散度：%.4f, %.4f" %(index, KL, KL2))
# print(kl_res)
print(FS)
# plt.plot()

fs = FS
print(list(filter(lambda x: x > 3, fs)))

plt.hist(FS)
plt.show()
plt.hist(kl_res)
plt.show()




