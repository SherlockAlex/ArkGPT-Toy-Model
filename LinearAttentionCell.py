import torch.nn as nn
import torch

class LinearAttentionCell(nn.Module):
    def __init__(self,in_features,out_features,device) -> None:
        super(LinearAttentionCell,self).__init__()
        
        self.device = device

        self.in_features = in_features
        self.out_features = out_features
        
        self.rotatry = torch.tensor([1 if i % 2 == 0 else -1 for i in range(out_features)],device=device)
        self.swap_indices = torch.tensor([i+1 if i % 2 == 0 else i-1 for i in range(out_features)], dtype=torch.long,device=device)

        self.wq = nn.Linear(in_features=in_features,out_features=out_features).to(device)
        self.wk = nn.Linear(in_features=in_features,out_features=out_features).to(device)
        self.wv = nn.Linear(in_features=in_features,out_features=out_features,bias=False).to(device)

        self.memory = 0
        self.zeta = 0
        self.pos = 0

        self.dk = torch.sqrt(torch.tensor(1.0/out_features,dtype=torch.float32,device=device))

        indices = torch.arange(self.out_features//2,dtype=torch.float32,device=self.device)
        self.thetas = 10000**(-2*indices)


    @torch.no_grad
    def get_cosine(self,pos):
        cosines = torch.cos(pos*self.thetas)
        cosines = cosines.repeat_interleave(2)
        return cosines
    
    @torch.no_grad
    def get_sine(self,pos):
        sine = torch.sin(pos*self.thetas)
        sine = sine.repeat_interleave(2)
        return sine

    def forward(self,x):
        
        # 自注意力机制
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)
        
        with torch.no_grad():
            
            cosines = self.get_cosine(self.pos)
            sines = self.get_sine(self.pos)

            
            B,T,C = q.shape

            q_x = q
            q_y = ((q*self.rotatry).squeeze().index_select(-1,self.swap_indices).unsqueeze(1))
            q_y = q_y.view(B,T,C)
            q = q_x*cosines + q_y*sines

            k_x = k
            k_y = ((k*self.rotatry).squeeze().index_select(-1,self.swap_indices).unsqueeze(1))
            k_y = k_y.view(B,T,C)
            k = k_x*cosines + k_y*sines

        q = nn.functional.elu(q*self.dk) + 1
        k = nn.functional.elu(k*self.dk) + 1

        with torch.no_grad():
            
            self.memory = self.memory + torch.matmul(torch.transpose(k,-2,-1),v)
            self.zeta = self.zeta + torch.transpose(k,-2,-1)
        
        # 检索模型中记忆相关的部分
        scale = torch.matmul(q,self.zeta)
        memory = torch.matmul(q,self.memory)

        y = (memory/scale)

        with torch.no_grad():
            self.pos = self.pos + 1

        return y
    
    def forget(self):
        with torch.no_grad():
            self.memory = 0
            self.zeta = 0
            self.pos = 0
            #self.cell_state = 0
        pass
    
    pass


