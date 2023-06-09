import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig, AdamW, WarmupLinearSchedule

import csv
import json
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report


# 超参数
EPOCHS = 10  # 训练的轮数
BATCH_SIZE = 10  # 批大小
MAX_LEN = 200  # 文本最大长度
LR = 1e-5  # 学习率
WARMUP_STEPS = 100  # 热身步骤
T_TOTAL = 1000  # 总步骤

# pytorch的dataset类 重写getitem,len方法
class Custom_dataset(Dataset):
    def __init__(self, dataset_list):
        self.dataset = dataset_list

    def __getitem__(self, item):
        # self.items=self.dataset[item][0].split(",")
        text = self.dataset[item][0]
        label = self.dataset[item][1]

        return text, label

    def __len__(self):
        return len(self.dataset)


# 加载数据集
def load_dataset(filepath, max_len):
    dataset_list = []
    indexs=[]
    f = open(filepath, 'r', encoding='gbk')
    r = csv.reader(f, delimiter='\t')
    for item in r:
        if r.line_num == 1:
            continue
        items=item[0].split(",")
        # print(items)
        dataset_list.append([items[2],items[1]])
        # indexs.append([items[2],items[3]])
    # print(dataset_list)
    # 根据max_len参数进行padding
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    for item in dataset_list:
        # print(item)
        item[0] = item[0].replace(' ','')
        num = max_len - len(item[0])
        if num < 0:
            item[0] = item[0][:max_len]
            item[0] = tokenizer.encode(item[0])
            num_temp = max_len - len(item[0])
            if num_temp > 0:
                for _ in range(num_temp):
                    item[0].append(0)
            # 在开头和结尾加[CLS] [SEP]
            item[0] = [101] + item[0] + [102]
            item[0] = str(item[0])
            continue

        for _ in range(num):
            item[0] = item[0] + '[PAD]'
        item[0] = tokenizer.encode(item[0])
        num_temp = max_len - len(item[0])
        if num_temp > 0:
            for _ in range(num_temp):
                item[0].append(0)
        item[0] = [101] + item[0] + [102]
        item[0] = str(item[0])
    # print(dataset_list)
    return dataset_list


# 计算每个batch的准确率
def  batch_accuracy(pre, label):
    pre = pre.argmax(dim=1)
    correct = torch.eq(pre, label).sum().float().item()
    accuracy = round(correct / float(len(label)), 3)

    return accuracy


if __name__ == "__main__":

    # 生成数据集以及迭代器
    train_dataset = load_dataset('./data3/Train.csv', max_len=MAX_LEN)
    # print(train_dataset)
    test_dataset = load_dataset('./data3/Test.csv', max_len=MAX_LEN)
    # print(test_dataset)
    train_cus = Custom_dataset(train_dataset)
    train_loader = DataLoader(dataset=train_cus, batch_size=BATCH_SIZE, shuffle=True)

    # test_cus = Custom_dataset(test_dataset)
    # test_loader = DataLoader(dataset=test_cus, batch_size=BATCH_SIZE)

    # Bert模型以及相关配置
    config = BertConfig.from_pretrained('bert-base-chinese')
    config.num_labels = 3
    model = BertForSequenceClassification(config = config)
    model = BertForSequenceClassification.from_pretrained('bert-base-chinese', config=config)
    # model.cuda()

 
    optimizer = AdamW(model.parameters(), lr=LR, correct_bias=False)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps = WARMUP_STEPS, t_total = T_TOTAL)


    model.train()
    print('开始训练...')

    for epoch in range(EPOCHS):
        accus = []
        losses = []
        for text, label in train_loader:

            label_list = list(map(json.loads, label))
            text_list = list(map(json.loads, text))
            # print(text_list[0])
            # print(label_list)
            # print(text_list)

            label_tensor = torch.tensor(label_list)
            text_tensor = torch.tensor(text_list)

            outputs = model(text_tensor, labels=label_tensor)
            loss, logits = outputs[:2]
            losses.append(loss)

            optimizer.zero_grad()
            loss.backward()
            scheduler.step()
            optimizer.step()

            acc = batch_accuracy(logits, label_tensor)
            accus.append(acc)

            print('epoch:{} | acc:{} | loss:{}'.format(epoch, acc, loss))

        print("average training loss:", sum(losses) / len(losses))
        print("average training accuracy:", sum(accus) / len(accus))
    torch.save(model.state_dict(), 'bert_cla1.ckpt')
    print('保存训练完成的model...')

    # 测试
    print('开始加载训练完成的model...')
    model.load_state_dict(torch.load('bert_cla1.ckpt'))

    print('开始测试...')
    model.eval()
    test_result = []
    labels=[]

    for text, label in test_dataset:
        labels.append(int(label))
        test_text = list(json.loads(text))
        # print(labels)
        text_tensor = torch.tensor(test_text).unsqueeze(0)

        with torch.no_grad():
            # print('tensor', text_tensor)
            # print('tensor.shape', text_tensor.shape)
            outputs = model(text_tensor, labels=None)
            # print(outputs)
            pre = outputs[0].argmax(dim=1)
            # print(pre)
            test_result.append(pre.item())

    # 分类报告：precision/recall/fi-score/均值/分类个数
    target_names = ['积极', '中性', '消极']
    print("测试集上的结果如下所示：")
    print(classification_report(labels, test_result, target_names=target_names))




