import torch.nn as nn
import torch

class MemoryCell(nn.Module):
    def __init__(self,in_features,out_features) -> None:
        super(MemoryCell,self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        self.wq = nn.Linear(in_features=in_features,out_features=out_features,bias=False)
        self.wk = nn.Linear(in_features=in_features,out_features=out_features,bias=False)
        self.wv = nn.Linear(in_features=in_features,out_features=out_features,bias=False)

        self.memory = 0
        self.zeta = 0
        self.pos = 0

        self.dk = 1.0/out_features

    def get_cosine(self,pos,d_model):
        indices = torch.arange(d_model//2,dtype=torch.float32)
        thetas = 10000**(-2*indices)
        cosines = torch.cos(pos*thetas)
        cosines = cosines.repeat_interleave(2)
        return cosines
    
    def get_sine(self,pos,d_model):
        indices = torch.arange(d_model//2,dtype=torch.float32)
        thetas = 10000**(-2*indices)
        sine = torch.sin(pos*thetas)
        sine = sine.repeat_interleave(2)
        return sine

    def forward(self,x):
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        sine = self.get_sine(self.pos,self.out_features)
        cosine = self.get_cosine(self.pos,self.out_features)
        self.pos = self.pos + 1

        q = torch.concat([q[1:,:],q[:1,:]],dim=0)*cosine + torch.concat([-q[1:,:],q[:1,:]],dim=0)*sine
        k = torch.concat([k[1:,:],k[:1,:]],dim=0)*cosine + torch.concat([-k[1:,:],k[:1,:]],dim=0)*sine

        query = torch.sigmoid(q*self.dk)
        key = torch.sigmoid(k*self.dk)

        self.memory = self.memory + torch.matmul(torch.transpose(key,-1,0),v)
        self.zeta = self.zeta + torch.transpose(key,-1,0)

        scale = torch.matmul(query,self.zeta)
        memory = torch.matmul(query,self.memory)

        y = memory/scale
        return y
    
    def forget(self):
        self.memory = self.memory*0
        self.zeta = self.zeta*0
        self.pos = self.pos*0
        pass
    
    pass


