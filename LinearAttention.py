import torch
import torch.nn as nn
from LinearAttentionCell import LinearAttentionCell

class LinearAttention(nn.Module):
    def __init__(self,d_model,in_features,out_features,device,*args, **kwargs) -> None:
        super(LinearAttention,self).__init__(*args, **kwargs)
        self.cell = LinearAttentionCell(in_features=in_features,out_features=d_model,device = device).to(device)
        self.projection = nn.Linear(in_features=d_model,out_features=out_features,bias=False,device=device)

    def forward(self,x):
        B,T,S = x.shape
        x = x.view(B,T,1,S)
        y = torch.stack([self.cell(x[:,i,:,:]) for i in range(T)],dim=0)
        y = torch.transpose(y,0,1)  # 第一指标与第二指标进行交换

        B,T,X,S = y.shape
        y = y.view(B,T,S)
        y = self.projection(y)

        return y

    def forget(self):
        self.cell.forget()

    pass