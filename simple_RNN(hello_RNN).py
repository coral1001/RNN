# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 14:50:20 2021

@author: qg
"""
import torch
input_size = 4
hidden_size = 4
batch_size = 1
num_layers = 1
idx2char=['e','h','l','o']#字典
x_data = [1,0,2,2,3]#hello
y_data = [3,1,2,3,2]#ohlol

one_hot_lookup = [
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,0],
    [0,0,0,1]
]
x_one_hot = [one_hot_lookup[x] for x in x_data]
inputs = torch.Tensor(x_one_hot).view(-1,batch_size,input_size)#增加一个batch_size 维度
labels = torch.LongTensor(y_data)#reshape the labels to(seqlen)
class RNNModel(torch.nn.Module):
    def __init__(self,input_size,hidden_size,batch_size,num_layers = 1):
        super(RNNModel, self).__init__()
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = torch.nn.RNN(input_size = self.input_size,hidden_size=self.hidden_size,num_layers=self.num_layers)
    def forward(self,input):
        hidden = torch.zeros(self.num_layers,self.batch_size,self.hidden_size)
        out,_ = self.rnn(input,hidden)
        return out.view(-1,self.hidden_size)#(seqLen*batchSize,hiddenSize)
net = RNNModel(input_size,hidden_size,batch_size,num_layers)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(),lr=0.1)
for epoch in range(50):
    optimizer.zero_grad()
    outputs = net(inputs)
    loss = criterion(outputs,labels)
    loss.backward()
    optimizer.step()
    _,idx = outputs.max(dim=1)
    idx = idx.data.numpy()
    print('Predicted: ',''.join([idx2char[x] for x in idx]),end = ' ')
    print(',Epoch[%d/15] loss=%.4f'%(epoch+1,loss.item()))
