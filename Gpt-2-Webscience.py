import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from transformers import GPT2Tokenizer,GPT2Model,TextDataset, DataCollatorForLanguageModeling, GPT2Config, GPT2LMHeadModel
from pathlib import Path
# tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# use like this: encoded_input = tokenizer("Do not meddle in the affairs of wizards, for they are subtle and quick to anger.")
# path = r'C:\Users\nrj1224\Desktop\HPC-summer\lab5\data\WOS\WOS5736'
# with gzip.open(path, 'rb') as lbpath:
    # labels = np.frombuffer(lbpath.read(), np.uint8)
# train_dataset = GPT_2Dataset(r'C:\Users\nrj1224\Desktop\HPC-summer\lab5\data\WOS\Meta-data', transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1037,), (0.3081,))]))
# test_dataset = GPT_2Dataset(r'C:\Users\nrj1224\Desktop\HPC-summer\lab5\data\WOS\Meta-data', train=False, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1037,), (0.3081,))]))
# train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
# test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

# TRAIN_BASE =False

#select top k numbers
import random
 
def select_top_k(predictions, k=10):
    predicted_index = random.choice(
        predictions[0, -1, :].sort(descending=True)[1][:10]).item()
    return predicted_index

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

import logging
logging.basicConfig(level = logging.INFO)

text = "Yesterday, "
indexed_tokens = tokenizer.encode(text)
tokens_tensor = torch.tensor([indexed_tokens])
print(tokenizer)

print('Words in vocabulary: ', tokenizer.vocab_size)

# from pytorch_transformers import GPT2LMHeadModel
 
# 读取 GPT-2 预训练模型
# model = GPT2LMHeadModel.from_pretrained("gpt2")
# model.eval()
 
# total_predicted_text = text
# n = 100  # 预测过程的循环次数
# for _ in range(n):
#     with torch.no_grad():
#         outputs = model(tokens_tensor)
#         predictions = outputs[0]
 
#     predicted_index = select_top_k(predictions, k=10)
#     predicted_text = tokenizer.decode(indexed_tokens + [predicted_index])
#     total_predicted_text += tokenizer.decode(predicted_index)
 
#     if '<|endoftext|>' in total_predicted_text:
#         # 如果出现文本结束标志，就结束文本生成
#         break
 
#     indexed_tokens += [predicted_index]
#     tokens_tensor = torch.tensor([indexed_tokens])
 
# print(total_predicted_text)

# inputs = '2^30%'
# print(tokenizer.encode("Hello world"))
# paths = ["X.txt"]

# config = GPT2Config(
#     vocab_size = tokenizer.vocab_size,
#     bos_token = tokenizer.bos_token_id,
#     eos_token = tokenizer.eos_token_id
# )
# config = GPT2Config(vocab_size=15_000,
#                          n_positions=1024,
#                          n_head=2,
#                          n_layer=2,
#                          n_embd=256)
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.eval()

# from datasets import load_dataset
# dataset = load_dataset("text", data_files=paths)
with open('C:\\Users\\nrj1224\Desktop\\HPC-summer\\lab5\\X.txt', 'r', encoding='utf-8', errors='ignore') as f:
    dataset = f.read()
print(len(dataset))


# def encode(lines):
#     return tokenizer(lines['text'], add_special_tokens = True, truncation = True, max_length = 1024)
indexed_text = tokenizer.encode(dataset)
del(dataset)

dataset_cut = []
for i in range(len(indexed_text)//1024):
    dataset_cut.append(indexed_text[i*1024:i*1024+1024])
del(indexed_text)

dataset_tensor = torch.tensor(dataset_cut)
print(dataset_tensor)

from torch.utils.data import DataLoader, TensorDataset
 
# 构建数据集和数据迭代器，设定 batch_size 大小为 2
train_set = TensorDataset(dataset_tensor,
                          dataset_tensor)  # 标签与样本数据相同
train_loader = DataLoader(dataset=train_set,
                          batch_size=2,
                          shuffle=False)
print(train_loader)
from torch import nn
from torch.autograd import Variable
import time
 
pre = time.time()
 
epoch = 30  # 循环学习 30 次
 
model.to(device)
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)  # 定义优化器
 
for i in range(epoch):
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data).to(device), Variable(
            target).to(device)
 
        optimizer.zero_grad()
 
        loss, logits, _ = model(data, labels=target)
 
        total_loss += loss
 
        loss.backward()
        optimizer.step()
 
        if batch_idx == len(train_loader)-1:
            # 在每个 Epoch 的最后输出一下结果
            print('average loss:', total_loss/len(train_loader))
 
print('训练时间：', time.time()-pre)
dataset = dataset['train']
# print(len(dataset))

datacollator = DataCollatorForLanguageModeling(tokenizer = tokenizer, mlm = False, mlm_probability = 0.15)
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="gpt_2",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=10,# 64
    save_steps=100,
    save_total_limit=2,
    prediction_loss_only=True,
)
print("model:")
print(model)
print("datacollator")
print(datacollator)
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=datacollator,
    train_dataset=dataset,
)
trainer.train()
trainer.save_model("gpt_2")

indexed_text = tokenizer.encode(dataset)
del(dataset)
 
dataset_cut = []
for i in range(len(indexed_text)//512):
    # 将字符串分段成长度为 512
    dataset_cut.append(indexed_text[i*512:i*512+512])
del(indexed_text)
 
dataset_tensor = torch.tensor(dataset_cut)
print(dataset_tensor.shape)
from torch.utils.data import DataLoader, TensorDataset
 
# 构建数据集和数据迭代器，设定 batch_size 大小为 2
train_set = TensorDataset(dataset_tensor,
                          dataset_tensor)  # 标签与样本数据相同
train_loader = DataLoader(dataset=train_set,
                          batch_size=2,
                          shuffle=False)
print(train_loader)
from torch import nn
from torch.autograd import Variable
import time
 
pre = time.time()
 
epoch = 30  # 循环学习 30 次
 
model.to(device)
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)  # 定义优化器
 
for i in range(epoch):
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data).to(device), Variable(
            target).to(device)
 
        optimizer.zero_grad()
 
        loss, logits, _ = model(data, labels=target)
 
        total_loss += loss
 
        loss.backward()
        optimizer.step()
 
        if batch_idx == len(train_loader)-1:
            # 在每个 Epoch 的最后输出一下结果
            print('average loss:', total_loss/len(train_loader))
 
print('训练时间：', time.time()-pre)














# training_args = TrainingArguments()
# config = GPT2Config(vocab_size=15_000,
#                          n_positions=512,
#                          n_head=2,
#                          n_layer=2,
#                          n_embd=256,)
# tokensizer = GPT2Tokenizer.from_pretrained('gpt2')


# criterion = torch.nn.CrossEntropyLoss()
# model = GPT2Model(config)
# loss = model.compute_loss
# dataset = TextDataset(
#     tokenizer=tokensizer,
#     file_path="r'C:\Users\nrj1224\Desktop\HPC-summer\lab5\data\WOS\Meta-data\X.txt'",
#     block_size=128
# )
# optimizer = torch.optim.Adam(model.parameters(), lr = 0.00001)

# from transformers import DataCollatorForLanguageModeling

# data_collator = DataCollatorForLanguageModeling(
#     tokenizer=tokensizer, mlm=False
# )
# from transformers import Trainer, TrainingArguments

# training_args = TrainingArguments(
#     # output_dir="./"+model_dir,
#     overwrite_output_dir=True,
#     num_train_epochs=100,
#     per_gpu_train_batch_size=64,
#     save_steps=20_000,
#     save_total_limit=2,
#     logging_steps=100
# )

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     data_collator=data_collator,
#     train_dataset=dataset,
#     prediction_loss_only=True,
# )

# trainer.train()
# # trainer.save_model("./"+model_dir)
