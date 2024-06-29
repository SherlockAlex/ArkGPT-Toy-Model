from ArkGPT import ArkGPT
import torch
import torch.nn.functional as F

# 设置打印选项
torch.set_printoptions(sci_mode=False)

a = torch.randn(6,8)
b = torch.randn(6,8)
s = torch.softmax(torch.matmul(a,b.transpose(-2,-1)),dim=-1)

print(s)

q = F.elu(a) + 1
k = F.elu(b) + 1

# q = q^n,n越高，注意力越容易集中，而n越低，注意力越涣散
q = q*q*q*q
k = k*k*k*k

y = torch.matmul(q,k.transpose(-2,-1))
scale = y.sum(dim=-1).unsqueeze(-1)
s = y/scale
print(s)