import torch
import torch.nn as nn
import torch.functional as F
import layers

class Forward(nn.Module):
    def __init__(self,in_features,out_features,*args, **kwargs) -> None:
        super(Forward,self).__init__(*args, **kwargs)
        
        self.normalize = layers.RMSNormalize()
        
        self.linear = nn.Linear(in_features,128)
        self.dense1 = nn.Linear(128,128)
        self.dense2 = nn.Linear(128,128)

    def forward(self,x):
        x = self.linear(x)
        x = self.normalize(x)

        x = torch.sigmoid(self.dense1(x)) + x
        x = torch.relu(self.dense2(x)) + x

        return x

        pass
        
    pass

class Block(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(Block,self).__init__(*args, **kwargs)
        self.memory = layers.Memory(128,128)
        self.feedforward = Forward(128,128)
        self.normalize = layers.RMSNormalize()

    def forward(self,x):
        x = self.normalize(x)
        x = self.memory(x) + x
        x = self.normalize(x)

        x = self.feedforward(x) + x
        return x
        
    def forget(self):
        pass

    pass

class Model(nn.Module):
    def __init__(self,vocab_count,d_model,*args, **kwargs) -> None:
        super(Model,self).__init__(*args, **kwargs)
        self.embedding = nn.Embedding(vocab_count,d_model)
        self.linear = nn.Linear(d_model,128,bias=False)
        self.memory_blocks = [Block() for i in range(8)]
        self.block = nn.Sequential(*(self.memory_blocks+[layers.RMSNormalize()]))
        self.logits = nn.Linear(128,vocab_count,bias=False)
        pass

    def forward(self,x):
        x = self.embedding(x)
        x = self.linear(x)
        x = self.block(x)
        logits = self.logits(x)
        return logits
    
    def forget(self):
        for x in self.memory_blocks:
            x.forget()

    pass