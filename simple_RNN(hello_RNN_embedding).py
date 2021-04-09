# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 14:50:20 2021

@author: qg
"""
import torch
num_class = 4
input_size = 4
hidden_size = 8
embedding_size = 10
num_layers = 2
batch_size = 1
seq_len =5
idx2char=['e','h','l','o']#字典
x_data = [1,0,2,2,3]#hello
y_data = [3,1,2,3,2]#ohlol
'''
one_hot_lookup = [
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,0],
    [0,0,0,1]
]
x_one_hot = [one_hot_lookup[x] for x in x_data]
'''
inputs = torch.LongTensor(x_data).view(batch_size,seq_len)#增加一个batch_size 维度
#inputs = torch.Tensor(x_data).view(-1,batch_size,seq_len)#增加一个batch_size 维度
#inputs = torch.Tensor(x_one_hot).view(-1,batch_size,input_size)
labels = torch.LongTensor(y_data)#reshape the labels to(seqlen)
class RNNModel(torch.nn.Module):
    def __init__(self):
        super(RNNModel, self).__init__()
        self.emb = torch.nn.Embedding(input_size, embedding_size)
        self.rnn = torch.nn.RNN(input_size = embedding_size,
                                hidden_size=hidden_size,
                                num_layers=num_layers,
                                batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, num_class) #全连接层，数量的变化
    def forward(self,x):
        hidden = torch.zeros(num_layers,x.size(0),hidden_size)
        x = self.emb(x)
        x, _ =self.rnn(x,hidden)
        x = self.fc(x)
        return x.view(-1,num_class)#(seqLen*batchSize,hiddenSize)
net = RNNModel()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(),lr=0.05)
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
