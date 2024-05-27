import torch
import torch.nn as nn

class RMSNormalize(nn.Module):
    def __init__(self,epsilon = 1e-12, *args, **kwargs) -> None:
        super(RMSNormalize,self).__init__(*args, **kwargs)
        self.epsilon = epsilon
        self.scale = torch.tensor(1,dtype=torch.float32)

    def forward(self,x):
        rms = torch.sqrt(torch.mean(torch.square(x),dim=0,keepdim=True))
        return x / (rms + self.epsilon)*self.scale
        pass
    pass