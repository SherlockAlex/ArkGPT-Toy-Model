import torch
import torch.nn as nn

a = torch.tensor([[[1,2,3,4,5,6,7]],[[7,8,9,4,5,6,1]]])
b = torch.tensor([[[7,6,5,4,1,2,3]],[[7,8,4,4,5,6,1]]])
c = torch.tensor([[[7,6,5,4,1,2,3]],[[7,8,4,4,5,6,1]]])

x = torch.stack([a,b,c])

y = torch.transpose(x,0,1)  # 第一指标与第二指标进行交换

B,T,X,D = y.shape
y = y.view(B,T,D)

print(y)