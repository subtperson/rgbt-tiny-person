import torch
import copy
import os
import numpy as np
import random


c1 = torch.nn.Conv2d(in_channels=2, out_channels=2, kernel_size=2, stride=2, padding=1)
c2 = torch.nn.Conv2d(in_channels=2, out_channels=2, kernel_size=3, stride=1, padding=1)


    
# random a input with 2*2*128*160
input = torch.rand(2, 2, 128, 160)
print(input.shape)
x = c1(input)
print(x.shape)










# class Model(torch.nn.Module):
#     def __init__(self):
#         super(Model, self).__init__()
#         self.l1 = torch.nn.Linear(out_features=10)
#         self.l2 = torch.nn.Linear(10, 10)
#         self.l3 = torch.nn.Linear(10, 10)
#         self.l4 = torch.nn.Linear(10, 10)
#         self.l5 = torch.nn.Linear(10, 10)

#     def forward(self, x):
#         x = self.l1(x)
#         x = self.l2(x)
#         # print(x)
#         x = self.l3(x)
#         # print(x)
#         a = x.detach()

#         a = self.l4(a)
#         a = self.l5(a)


#         return x, a
    
# model = Model()
# model = model.cuda()
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# input = torch.tensor([1., 2, 3, 4, 5, 6, 7, 8, 9, 10], requires_grad=True).cuda()

# for i in range(4):
#     output, a = model(input)
#     loss = output.sum()
#     lossa = a.sum()-6
#     optimizer.zero_grad()
    
#     total_loss = loss + lossa
#     total_loss.backward()
#     optimizer.step()
# print(output)


