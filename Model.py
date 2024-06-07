import torch
import torch.nn as nn
from torch.nn import functional as F
import layers

class FeedForward(nn.Module):
    def __init__(self,in_features,out_features,device,*args, **kwargs) -> None:
        super(FeedForward,self).__init__(*args, **kwargs)
        
        self.normalize = layers.RMSNormalize().to(device)
        
        self.dense1 = nn.Linear(in_features,4*out_features).to(device)
        self.dropout1 = nn.Dropout(0.1).to(device)
        self.dense2 = nn.Linear(4*out_features,out_features).to(device)
        self.dropout2 = nn.Dropout(0.1).to(device)

    def forward(self,x):
        x = torch.sigmoid(self.dense1(x))
        x = self.dropout1(x)
        x = torch.sigmoid(self.dense2(x))
        x = self.dropout2(x)

        return x

        pass
        
    pass

class Block(nn.Module):
    def __init__(self,units, device,*args, **kwargs) -> None:
        super(Block,self).__init__(*args, **kwargs)
        self.memory = layers.LinearAttention(units,units,units,device).to(device)
        self.moe = layers.MoELayer(num_experts=8,in_features=units,out_features=4*units,device = device).to(device)
        self.feedforward = FeedForward(4*units,units,device).to(device)
        self.normalize = layers.RMSNormalize().to(device)

    def forward(self,x):
        input = x
        x = self.memory(self.normalize(x)) + x
        x = self.moe(self.normalize(x))
        x = self.feedforward(self.normalize(x))
        return x + input
        
    def forget(self):
        self.memory.forget()
        pass

    pass

class Model(nn.Module):
    def __init__(self,vocab_count,d_model,units,device,*args, **kwargs) -> None:
        super(Model,self).__init__(*args, **kwargs)
        self.embedding = nn.Embedding(vocab_count,d_model).to(device)
        self.linear = nn.Linear(d_model,units,bias=False).to(device)
        self.memory_blocks = [Block(units,device).to(device) for i in range(16)]
        self.block = nn.Sequential(*(self.memory_blocks+[layers.RMSNormalize().to(device)]))
        self.normalize = layers.RMSNormalize()
        self.logits = nn.Linear(units,vocab_count,bias=False).to(device)
        pass

    def forward(self,x,labels = None):
        x = self.embedding(x)
        x = self.linear(x)
        x = self.block(x) + x
        x = self.normalize(x)
        logits = self.logits(x)
        if labels is not None:
            # 训练模式
            B,T,C = logits.shape
            logits_reshaped = logits.view(B * T, C)
            labels_reshaped = labels.view(B * T)
            loss = F.cross_entropy(input=logits_reshaped, target=labels_reshaped)
            self.forget()
            return logits,loss
            pass
        return logits
    
    def forget(self):
        for x in self.memory_blocks:
            x.forget()

    pass

