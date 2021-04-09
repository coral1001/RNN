# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 14:50:20 2021

@author: qg
"""

import torch

batch_size = 1
seq_len = 3
input_size = 4
hidden_size = 2
num_layers = 1 #使用自带的RNN,RNN中hidden的层数  

'''
#使用cell，需要自己写循环
cell = torch.nn.RNNCell(input_size= input_size, hidden_size = hidden_size)

dataset = torch.randn(seq_len,batch_size,input_size)
hidden = torch.zeros(batch_size,hidden_size)

for idx , input in enumerate(dataset):
    print('='*20, idx,'='*20)
    print('input size' , input.shape)
    
    hidden = cell(input,hidden)
    print('hidden size:',hidden_size)
    print(hidden)
'''

cell = torch.nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)

inputs = torch.randn(seq_len, batch_size,input_size)
hidden = torch.zeros(num_layers, batch_size,hidden_size)

out,hidden = cell(inputs, hidden)

print('output size:',out.shape)
print('output',out)
print('hidden size:',hidden.shape)
print('hidden:',hidden)