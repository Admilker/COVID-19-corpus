import re
import time
from collections import Counter
from datetime import timedelta

import jieba

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from numpy import shape
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import pickle as pkl
from sklearn import metrics
import pickle


UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号

tokenizer = lambda x: [y for y in x]

# 根据训练集构建词表
def build_vocab(filepath):
    df = pd.read_csv(filepath)
    text = list(df['text'])
    label = list(df['label'])
    words = Counter()
    for i, sentence in enumerate(text):
        words_list = jieba.cut(sentence)  # 将句子进行分词
        # words_list = tokenizer(sentence)  # 将句子进行分词
        words.update(words_list)  # 更新词频列表
        # text[i] = words_list  # 分词后的单词列表存在该列表中
    words = sorted(words, key=words.get, reverse=True)
    print(words[:10])
    words = [PAD] + words
    # 构造词典
    word2idx = {o: i for i, o in enumerate(words)}
    idx2word = {i: o for i, o in enumerate(words)}

    # 词典保存
    # with open('./dataset/word2index', 'wb') as f:
    #     pickle.dump(word2idx, f)
    return word2idx, idx2word, words

# 自己构造词表加载数据
def data_load(filepath, word2idx, batch_size, pad_size = 32):
    df = pd.read_csv(filepath)
    text = list(df['text'])
    label = list(df['label'])
    result = []
    # text转换为词向量
    for i, sentence in enumerate(text):
        # sentence = re.sub('([^\u4e00-\u9fa5\u0030-\u0039])', ' ', sentence)
        lin = " ".join(jieba.cut(sentence)).split()
        # lin = tokenizer(lin)
        seq = len(lin)
        if seq>180:
            label[i] = -1
            continue
        if pad_size:
            if len(lin)<pad_size:
                lin.extend([PAD] * (pad_size - len(lin)))
            else:
                lin = lin[:pad_size]
                seq = pad_size
        lin = [word2idx[word] if word in word2idx else 0 for word in lin]
        result.append(lin)
    label = list(filter(lambda x : x != -1, label))
    data = TensorDataset(torch.tensor(result), torch.tensor(label))
    data = DataLoader(data, shuffle=True, batch_size=batch_size)
    return data

# 搜狗词表加载数据
# def data_load_sogo(filepath, vocab_path, batch_size, pad_size = 32):
#     vocab = pkl.load(open(vocab_path, 'rb'))
#     df = pd.read_csv(filepath)
#     text = list(df['text'])
#     label = list(df['label'])
#     result = []
#     # text转换为词向量
#     for i, sentence in enumerate(text):
#         lin = re.sub('([^\u4e00-\u9fa5\u0030-\u0039])', '', sentence)
#         lin = tokenizer(lin)
#         seq = len(lin)
#         words_line = []
#         if seq>200:
#             label[i] = -1
#             continue
#         if pad_size:
#             if len(lin)<pad_size:
#                 lin.extend([PAD] * (pad_size - len(lin)))
#             else:
#                 lin = lin[:pad_size]
#                 seq = pad_size
#         for word in lin:
#             words_line.append(vocab.get(word, vocab.get(UNK)))
#         # result.append((lin, label[i]))
#         result.append(words_line)
#     label = list(filter(lambda x: x != -1, label))
#     data = TensorDataset(torch.tensor(result), torch.tensor(label))
#     data = DataLoader(data, shuffle=True, batch_size=batch_size)
#     return data

# 构建模型
class LSTM(nn.Module):
    def __init__(self, vocab_size):
        super(LSTM, self).__init__()

        self.n_layer = n_layer = 3
        self.hidden_layer = hidden_layer = 128
        # self.hidden = self.init_hidden()
        embedding_dim = 300
        drop_out = 0.5

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, # 输入的维度
                            hidden_layer, # LSTM输出的hidden_state的维度
                            n_layer, # LSTM的层数
                            dropout=drop_out,
                            bidirectional=True,
                            batch_first=True # 第一个维度是否是batch_size
                            )
        self.fc = nn.Linear(hidden_layer*2, 3)
        # self.softmax = nn.Softmax()
        # self.dropout = nn.Dropout(drop_out)

    def forward(self, x):
        x = x.long()
        embed = self.embedding(x)
        out, (h_0, c_0) = self.lstm(embed)
        # out, self.hidden = self.lstm(embed)
        out = out[:, -1, :]
        out = self.fc(out)
        # out = F.softmax(out, dim=1)
        return out, h_0

# 训练
def train(epoch, train_data, dev_data, lr, class_list):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    loss_train = []
    train_acclist  = []
    setite = 100
    count = 0

    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    for i in range(epoch):
        print('Epoch [{}/{}]'.format(i + 1, epoch))

        # scheduler.step()

        # model.hidden = model.init_hidden()
        for train, label in train_data:
            count += 1
            model.zero_grad()
            trains, labels = train.to(device), label.to(device)
            outputs, hidden = model(trains)
            # outputs = model(trains)
            # optimizer.zero_grad()

            loss = F.cross_entropy(outputs, labels, label_smoothing=0.1)

            loss.backward()
            loss_train.append(loss.cpu().detach().numpy())
            optimizer.step()

            true1 = label.data.cpu()
            predic1 = torch.max(outputs.data, 1)[1].cpu()
            train_acc1 = metrics.accuracy_score(true1, predic1)
            train_acclist.append(train_acc1)

            if count % setite == 0:
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                dev_acc, dev_loss = evaluate(model, dev_data, class_list)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), 'savemodel/test.ckpt')
                    improve = '*'
                else:
                    improve = ''
                print("Step: {}...".format(count),
                      "Loss: {:.6f}...".format(loss.item()),
                      "Acc: {:.6f}...".format(train_acc),
                      "Val Loss:{}".format(dev_loss),
                      "Val acc:{}".format(dev_acc),
                      "improve:{}".format(improve))
                model.train()


    # torch.save(model.state_dict(), 'savemodel/LSTM.ckpt')

    plt.plot(loss_train)
    plt.plot(train_acclist)
    plt.xlabel('train item')
    plt.show()
    test(model, tests, class_list)

# 测试
def test(model, test_iter, class_list):
    # test
    model.load_state_dict(torch.load('./savemodel/test.ckpt'))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = evaluate(model, test_iter, class_list, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


def evaluate(model, data_iter, class_list, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in data_iter:
            texts, labels = texts.to(device), labels.to(device)
            outputs, hidden = model(texts)
            # outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter)

def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


if __name__ == '__main__':
    train_path = './dataset/Train.csv'
    test_path = '../corpus/Test.csv'
    dev_path = './dataset/Dev.csv'
    class_list = ['positive', 'mid', 'negative']
    vocab_sogo = './dataset/vocab.pkl'
    pad_size = 130
    batch_size = 32
    epoch = 100
    lr = 0.001

    # 基于训练集词典
    word2index, idx2word, words = build_vocab(train_path)
    with open('./dataset/word2index', 'rb') as f:
        word2idx = pkl.load(f)
    trains = data_load(train_path, word2idx, batch_size, pad_size)
    tests = data_load(test_path, word2idx, batch_size, pad_size)
    devs = data_load(dev_path, word2idx, batch_size, pad_size)
    model = LSTM(len(words))

    # 基于搜狗词典
    # vocab = pkl.load(open(vocab_sogo, 'rb'))
    # trains = data_load_sogo(train_path, vocab_sogo, batch_size, pad_size)
    # tests = data_load_sogo(test_path, vocab_sogo, batch_size, pad_size)
    # devs = data_load_sogo(dev_path, vocab_sogo, batch_size, pad_size)

    # model = AutoModelForMaskedLM.from_pretrained("bert-base-chinese")
    # model = LSTM(21128)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
    # device = "cpu"
    model.to(device)
    print(model.parameters)
    train(epoch, trains, devs, lr, class_list)
