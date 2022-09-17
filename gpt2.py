import random


def select_top_k(predictions, k=10):
    predicted_index = random.choice(
        predictions[0, -1, :].sort(descending=True)[1][:10]).item()
    return predicted_index
import torch
from transformers import GPT2Tokenizer

import logging
logging.basicConfig(level=logging.INFO)

# 载入预训练模型的分词器
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

from transformers import GPT2LMHeadModel

# 读取 GPT-2 预训练模型
model = GPT2LMHeadModel.from_pretrained("gpt2")

with open('/home/nrj-summer/gpt2justforhomework/X.txt', 'r', encoding='utf-8', errors='ignore') as f:
    dataset = f.read()
# print(len(dataset))

indexed_text = tokenizer.encode(dataset)
del(dataset)

dataset_cut = []
for i in range(len(indexed_text)//1024):
    # 将字符串分段成长度为 1024
    dataset_cut.append(indexed_text[i*1024:i*1024+1024])
del(indexed_text)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset_tensor = torch.tensor(dataset_cut)
dataset_tensor.shape
from torch.utils.data import DataLoader, TensorDataset

# 构建数据集和数据迭代器，设定 batch_size 大小为 2
train_set = TensorDataset(dataset_tensor,
                          dataset_tensor)  # 标签与样本数据相同
train_loader = DataLoader(dataset=train_set,
                          batch_size=1,
                          shuffle=False)
# print(train_loader)
from torch import nn
from torch.autograd import Variable
import time

pre = time.time()

epoch = 1  # 循环学习 30 次

model.to(device)
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)  # 定义优化器

for i in range(epoch):
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data).to(device), Variable(
            target).to(device)

        optimizer.zero_grad()

#         loss, logits, _ = model(data, labels=target)
        loss = model(data, labels=target).loss
        logits = model(data, labels=target).logits

        total_loss += float(loss)

        loss.backward()
        optimizer.step()

        if batch_idx == len(train_loader)-1:
            # 在每个 Epoch 的最后输出一下结果
            print('average loss:', total_loss/len(train_loader))

print('训练时间：', time.time()-pre)
