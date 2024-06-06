import torch
import torch.nn as nn
import layers

class MultiLinearAttention(nn.Module):
    def __init__(self,num_heads,in_features,out_features,device = 'cpu',*args, **kwargs) -> None:
        super(MultiLinearAttention,self).__init__(*args, **kwargs)
        self.device = device
        self.num_heads = num_heads
        self.head_size = in_features//num_heads
        self.heads = nn.ModuleList([layers.LinearAttention(in_features=self.head_size,out_features=out_features,device=self.device) for i in range(self.num_heads)])
        self.projection = nn.Linear(in_features=self.num_heads*out_features,out_features=out_features,bias=False,device=device)
        pass

    def forward(self,x):
        inputs = torch.split(x,self.head_size,dim=-1)
        outputs = [head(input) for head,input in zip(self.heads,inputs)]
        y = torch.concat(outputs,dim=-1)
        y = self.projection(y)
        return y
    
    def forget(self):
        for x in self.heads:
            x.forget()

    pass
