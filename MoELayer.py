import torch
import torch.nn as nn

class MoELayer(nn.Module):
    def __init__(self,num_experts,in_features,out_features,device,*args, **kwargs) -> None:
        super(MoELayer,self).__init__(*args, **kwargs)
        self.d_model = in_features // num_experts
        self.experts = nn.ModuleList([nn.Linear(in_features=self.d_model,out_features=out_features).to(device) for i in range(num_experts)])
        self.router = nn.Linear(in_features=in_features,out_features=num_experts).to(device)
        self.dropout = nn.Dropout(0.1).to(device)
        self.projection = nn.Linear(in_features=in_features,out_features=out_features)
        

    def forward(self,x):
        logits = self.router(x)
        router = torch.softmax(logits,dim=-1)
        
        inputs = torch.split(x,self.d_model,dim=-1)
        experts = torch.stack([self.dropout(torch.tanh(expert(input))) for input,expert in zip(inputs,self.experts)],dim=0)
        experts_output = torch.einsum('btx,xbts->bts',router,experts)
        output = self.dropout(torch.tanh(self.projection(x))) + experts_output
        return output
        

        pass
    pass
