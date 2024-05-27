import torch
import torch.nn as nn
from MemoryCell import MemoryCell

class Memory(nn.Module):
    def __init__(self,in_features,out_features, *args, **kwargs) -> None:
        super(Memory,self).__init__(*args, **kwargs)
        self.cell = MemoryCell(in_features=in_features,out_features=out_features)

    def forward(self,x):
        shape = x.shape
        inputs = x.reshape(shape[0],1,shape[-1])
        y = [self.cell(inputs[i]) for i in range(shape[0])]
        y = torch.stack(y,dim=0)
        shape = y.shape
        y = y.reshape(shape[0],shape[-1])
        return y

    def forget(self):
        self.cell.forget()

    pass