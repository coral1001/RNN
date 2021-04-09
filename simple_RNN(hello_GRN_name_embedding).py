# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 19:17:42 2021

@author: qg
"""

#准备数据

from torch.utils.data import Dataset
import pandas as pd

class NameDataset(Dataset):
    """数据集类"""
    def __init__(self, is_train_set=True):
        filename = './name_data/names_train.csv' if is_train_set else './name_data/names_test.csv'
        data = pd.read_csv(filename, header=None)
        self.names = data[0]
        self.len = len(self.names)
        self.countries = data[1]
        self.country_list = list(sorted(set(self.countries)))
        self.country_dict = self.getCountryDict()
        self.country_num = len(self.country_list)
    
    def __getitem__(self, index):
        return self.names[index], self.country_dict[self.countries[index]]
    
    def __len__(self):
        return self.len

    def idx2country(self, index):
        return self.country_list[index]

    def getCountryDict(self):
        country_dict = dict()
        for idx, country_name in enumerate(self.country_list, 0):
            country_dict[country_name] = idx
        return country_dict

    def getCountriesNum(self):
        return self.country_num

#定义函数

def name2list(name):
    """返回ASCII码表示的姓名列表与列表长度"""
    arr = [ord(c) for c in name]
    return arr, len(arr)


def make_tensors(names, countries):
    # 元组列表，每个元组包含ASCII码表示的姓名列表与列表长度，名字变链表
    sequences_and_lengths = [name2list(name) for name in names]
    # 取出所有的ASCII码表示的姓名列表
    name_sequences = [sl[0] for sl in sequences_and_lengths]
    # 取出所有的列表长度
    seq_lengths = torch.LongTensor([sl[1] for sl in sequences_and_lengths])
    # 将countries转为long型
    countries = countries.long()

    # 接下来每个名字序列补零，使之长度一样。
    # 先初始化一个全为零的tensor，大小为 所有姓名的数量*最长姓名的长度
    seq_tensor = torch.zeros(len(name_sequences), seq_lengths.max()).long()

    # 将姓名序列覆盖到初始化的全零tensor上
    for idx, (seq, seq_len) in enumerate(zip(name_sequences, seq_lengths), 0):
        seq_tensor[idx, :seq_len] = torch.LongTensor(seq)
    # 根据序列长度seq_lengths对补零后tensor进行降序怕排列，方便后面加速计算。
    # 返回排序后的seq_lengths与索引变化列表 sort返回排序的值和列表的ID
    seq_lengths, perm_idx = seq_lengths.sort(dim=0, descending=True)
    # 根据索引变化列表对ASCII码表示的姓名列表进行排序
    seq_tensor = seq_tensor[perm_idx]
    # 根据索引变化列表对countries进行排序，使姓名与国家还是一一对应关系
    # seq_tensor.shape : batch_size*max_seq_lengths,
    # seq_lengths.shape : batch_size
    # countries.shape : batch_size
    countries = countries[perm_idx]
    return seq_tensor, seq_lengths, countries

#定义模型

import torch
from torch.nn.utils.rnn import pack_padded_sequence

class RNNClassifier(torch.nn.Module):
    # input_size=128, hidden_size=100, output_size=18
    def __init__(self, input_size, hidden_size, output_size, n_layers=1, bidirectional=True):
        super(RNNClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_directions = 2 if bidirectional else 1  # 是否双向
        self.embedding = torch.nn.Embedding(input_size, hidden_size)  # 输入大小128，输出大小100。
        # 经过Embedding后input的大小是100，hidden_size的大小也是100，所以形参都是hidden_size。
        self.gru = torch.nn.GRU(hidden_size, hidden_size, n_layers, bidirectional=bidirectional)
        # 如果是双向，会输出两个hidden层，要进行拼接，所以线形成的input大小是 hidden_size * self.n_directions，输出是大小是18，是为18个国家的概率。
        self.fc = torch.nn.Linear(hidden_size * self.n_directions, output_size)
    
    def _init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers * self.n_directions, batch_size, self.hidden_size)
        return hidden

    def forward(self, input, seq_lengths):
        # 先对input进行转置，input shape : batch_size*max_seq_lengths -> max_seq_lengths*batch_size 每一列表示姓名
        input = input.t()#将输入数据转置
        batch_size = input.size(1)  # 总共有多少列，既是batch_size的大小
        hidden = self._init_hidden(batch_size)  # 初始化隐藏层
        embedding = self.embedding(input)  # embedding.shape : max_seq_lengths*batch_size*hidden_size 12*64*100
        # pack_padded_sequence方便批量计算
        gru_input = pack_padded_sequence(embedding, seq_lengths)
        # 进入网络进行计算
        output, hidden = self.gru(gru_input, hidden)

        # 如果是双向的，需要进行拼接
        if self.n_directions == 2:
            hidden_cat = torch.cat([hidden[-1], hidden[-2]], dim=1)

        else:
            hidden_cat = hidden[-1]

        # 线性层输出大小为18
        fc_output = self.fc(hidden_cat)
        return fc_output

#定义训练函数

def time_since(since):
    s = time.time() - since
    m = math.floor(s/60)
    s-= m*60
    return '%dm %ds' % (m, s)


def trainModel():
    total_loss = 0
    for i, (names, countries) in enumerate(trainloader, 1):  # 这里的1意思是 i 从1开始。
        # make_tensors函数返回经过降序排列后的 姓名列表，列表长度，国家
        inputs, seq_lengths, target = make_tensors(names, countries)
        # 输入姓名列表与列表长度向前计算
        output = classifier(inputs, seq_lengths)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if i % 10 == 0:
            print(i)
            print(f'[{time_since(start)}] Epoch {epoch} ', end='')
            print(f'[{i * len(inputs)}/{len(trainset)}] ', end='')
            print(f'loss={total_loss / (i * len(inputs))}')
    return total_loss


#定义测试函数
def testModel():
    correct = 0
    total = len(testset)
    print("evaluating trained model ...")
    with torch.no_grad():
        for i, (names, countries) in enumerate(testloader, 1):
            inputs, seq_lengths, target = make_tensors(names, countries)
            output = classifier(inputs, seq_lengths)
            pred = output.max(dim=1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
        percent = '%.2f' % (100 * correct / total)
        print(f'Test set: Accuracy {correct}/{total} {percent}%')
    return correct / total


#主函数循环

from torch.utils.data import DataLoader
import time
import math

if __name__ == '__main__':
    
    N_EPOCHS = 25  # epoch
    HIDDEN_SIZE = 100  # 隐藏层的大小，也是Embedding后输出的大小
    BATCH_SIZE = 64
    N_COUNTRY = 18  # 总共有18个类别的国家，为RNN后输出的大小
    N_LAYER = 2
    N_CHARS = 128  # 字母字典的大小，Embedding输入的大小

    trainset = NameDataset(is_train_set=True)
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)

    testset = NameDataset(is_train_set=False)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

    # 建立分类模型
    classifier = RNNClassifier(N_CHARS, HIDDEN_SIZE, N_COUNTRY, N_LAYER)

    # 建立损失函数与优化器
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)

    start = time.time()
    print("Training for %d epochs..." % N_EPOCHS)
    acc_list = []
    for epoch in range(1, N_EPOCHS + 1):
        # Train cycle
        trainModel()
        acc = testModel()
        acc_list.append(acc)

print(acc_list)

import numpy as np
import matplotlib.pyplot as plt
#循环，起始是1，列表长度+1是终点。步长是1
epoch = np.arange(1, len(acc_list) + 1, 1)
#将数据变成一个矩阵
acc_list = np.array(acc_list)
#循环，列表
plt.plot(epoch, acc_list)
#x标签
plt.xlabel('Epoch')
#y标签
plt.ylabel('Accuracy')
#绿色
plt.grid()
#展示
plt.show()
