import torch
import torch.nn as nn
from torch.nn import functional as F

eps = 1e-8

class Attention(nn.Module):
    def __init__(self, d_model, device):
        super(Attention, self).__init__()
        self.d_model = d_model
        self.device = device
        self.dk = torch.sqrt(torch.tensor(1.0 / d_model, dtype=torch.float32, device=device))
        self.is_use_rope = False

    def forward(self, query, key, value, mask=None):
        
        if self.is_use_rope:

            B,T,C = query.shape
            
            positions = torch.arange(T, dtype=torch.float32, device=self.device).unsqueeze(1)
            pos_thetas = positions * self.thetas
            cosines = torch.cos(pos_thetas).repeat_interleave(2, dim=-1)
            sines = torch.sin(pos_thetas).repeat_interleave(2, dim=-1)

            q_x = query
            q_y = ((query*self.rotatry).squeeze().index_select(-1,self.swap_indices).unsqueeze(1))
            q_y = q_y.view(B,T,C)
            query = q_x*cosines + q_y*sines

            k_x = key
            k_y = ((key*self.rotatry).squeeze().index_select(-1,self.swap_indices).unsqueeze(1))
            k_y = k_y.view(B,T,C)
            key = k_x*cosines + k_y*sines
        
        attn_logits = torch.matmul(query, key.transpose(-2, -1)) / self.dk
        if mask is not None:
            attn_logits += (mask * -1e9)  # Applying mask to the attention logits

        score = torch.softmax(attn_logits, dim=-1)
        attention = torch.matmul(score, value)
        return attention

    def use_rope(self, state: bool,npos_max = 64):
        self.is_use_rope = state
        if self.is_use_rope:
            self.rotatry = torch.tensor([1 if i % 2 == 0 else -1 for i in range(self.d_model)],device=self.device)
            self.swap_indices = torch.tensor([i+1 if i % 2 == 0 else i-1 for i in range(self.d_model)], dtype=torch.long,device=self.device)

            indices = torch.arange(self.d_model//2, dtype=torch.float32, device=self.device)/self.d_model
            self.thetas = 10000**(-2*indices)

class LinearAttentionCell(nn.Module):
    def __init__(self,d_model,device) -> None:
        super(LinearAttentionCell,self).__init__()
        self.device = device
        self.d_model = d_model
        self.memory = 0
        self.zeta = 0
        self.pos = 0
        self.forget_rate = 1-10*eps     # 防止记忆数值爆炸
        self.dk = torch.tensor(1.0/self.d_model,dtype=torch.float32,device=self.device)
        self.use_rope = False
        pass

    @torch.no_grad
    def get_cosine(self,theta:torch.Tensor):
        cosines = torch.cos(theta)
        cosines = cosines.repeat_interleave(2,dim=-1)
        return cosines
    
    @torch.no_grad
    def get_sine(self,theta:torch.Tensor):
        sine = torch.sin(theta)
        sine = sine.repeat_interleave(2,dim=-1)
        return sine

    def forward(self,query,key,value):

        q = query
        k = key

        if self.use_rope:
            with torch.no_grad():
                theta = self.pos*self.thetas
                cosines = self.get_cosine(theta)
                sines = self.get_sine(theta)

            B,T,C = q.shape
            q_x = q
            q_y = ((q*self.rotatry).squeeze().index_select(-1,self.swap_indices).unsqueeze(1))
            q_y = q_y.view(B,T,C)

            k_x = k
            k_y = ((k*self.rotatry).squeeze().index_select(-1,self.swap_indices).unsqueeze(1))
            k_y = k_y.view(B,T,C)
            
            q = q_x*cosines + q_y*sines
            k = k_x*cosines + k_y*sines

        q:torch.Tensor = torch.nn.functional.elu(q) + 1
        k:torch.Tensor = torch.nn.functional.elu(k) + 1

        # (1,s)
        q = torch.pow(q,5)
        k = torch.pow(k,5)

        delta_zeta = k.transpose(-2,-1)
        delta_memory = torch.matmul(delta_zeta,value)
        memory = self.forget_rate*self.memory + delta_memory
        zeta = self.forget_rate*self.zeta + delta_zeta

        scale = torch.matmul(q,zeta) + eps
        score = torch.matmul(q,memory)
        attention = score/scale

        self.apply_update(memory=memory,zeta=zeta)
        return attention
    
    def apply_update(self,memory:torch.Tensor,zeta:torch.Tensor):
        with torch.no_grad():
            self.memory = memory
            self.zeta = zeta
            self.pos = self.pos + 1
        pass

    def is_use_rope(self,value):
        self.use_rope = value
        if self.use_rope:
            with torch.no_grad():
                self.rotatry = torch.tensor([1 if i % 2 == 0 else -1 for i in range(self.d_model)],device=self.device)
                self.swap_indices = torch.tensor([i+1 if i % 2 == 0 else i-1 for i in range(self.d_model)], dtype=torch.long,device=self.device)
                indices = (torch.arange(self.d_model//2,dtype=torch.float32) + 1)/self.d_model
                self.thetas = 10000**(-2*indices).to(device=self.device)
                self.thetas.requires_grad=False
    
    def forget(self):
        with torch.no_grad():
            self.memory = 0
            self.zeta = 0
            self.pos = 0
        pass
    
    pass


class LinearAttention(nn.Module):
    def __init__(self,d_model,out_features,device,*args, **kwargs) -> None:
        super(LinearAttention,self).__init__(*args, **kwargs)
        self.cell = LinearAttentionCell(d_model=d_model,device = device).to(device)
        self.projection = nn.Linear(in_features=d_model,out_features=out_features,bias=False,device=device)
        self.dropout = nn.Dropout(0.2)
        pass

    def forward(self,query:torch.Tensor,key:torch.Tensor,value:torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            B,T,S = query.shape
            query = query.view(B,T,1,S)

            B,T,S = key.shape
            key = key.view(B,T,1,S)

            B,T,S = value.shape
            value = value.view(B,T,1,S)

        y = torch.stack([self.cell(query[:,i,:,:],key[:,i,:,:],value[:,i,:,:]) for i in range(T)],dim=0)
        y = torch.transpose(y,0,1)  # 第一指标与第二指标进行交换
        with torch.no_grad():
            B,T,X,S = y.shape
            y = y.view(B,T,S)
        y = self.dropout(self.projection(y))
        return y

    def forget(self):
        self.cell.forget()
        pass

    def is_use_rope(self,value:bool):
        self.cell.is_use_rope(value=value)

    pass

class SelfLinearAttention(nn.Module):
    def __init__(self,d_model,in_features,out_features,device,*args, **kwargs) -> None:
        super(SelfLinearAttention,self).__init__(*args, **kwargs)
        self.wq = nn.Linear(in_features=in_features,out_features=d_model,bias=False).to(device)
        self.wk = nn.Linear(in_features=in_features,out_features=d_model,bias=False).to(device)
        self.wv = nn.Linear(in_features=in_features,out_features=d_model,bias=False).to(device)
        
        self.linear_attention = LinearAttention(d_model=d_model,out_features=out_features,device=device)
        pass

    def forward(self,x):
        query = self.wq(x)
        key = self.wk(x)
        value = self.wv(x)

        attention = self.linear_attention(query,key,value)
        return attention
        pass

    def forget(self):
        self.linear_attention.forget()

    def is_use_rope(self,value:bool):
        self.linear_attention.is_use_rope(value)
    pass

class SelfAttention(nn.Module):
    def __init__(self,in_features,out_features,d_model,device, *args, **kwargs) -> None:
        super(SelfAttention,self).__init__(*args, **kwargs)
        self.attention = Attention(d_model=d_model,device=device)
        self.projection = nn.Linear(in_features=d_model,out_features=out_features,bias=False).to(device)
        self.wq = nn.Linear(in_features=in_features,out_features=d_model,bias=False).to(device)
        self.wk = nn.Linear(in_features=in_features,out_features=d_model,bias=False).to(device)
        self.wv = nn.Linear(in_features=in_features,out_features=d_model,bias=False).to(device)
    
    def forward(self,x:torch.Tensor):
        query = self.wq(x)
        key = self.wk(x)
        value = self.wv(x)
        x = self.attention(query,key,value)
        y = self.projection(x)
        return y
        pass
    pass

class FeedForward(nn.Module):
    def __init__(self,in_features,out_features,device,*args, **kwargs) -> None:
        super(FeedForward,self).__init__(*args, **kwargs)
        self.dense = nn.Linear(in_features,4*out_features).to(device)
        self.linear = nn.Linear(4*out_features,out_features).to(device)
        self.dropout = nn.Dropout(0.3).to(device)
        self.gelu = nn.GELU()
        #self.relu = nn.ReLU()
        pass

    def forward(self,x:torch.Tensor) -> torch.Tensor:
        x = self.gelu(self.dense(x))
        x = self.dropout(self.linear(x))
        return x

    pass

class MoELayer(nn.Module):
    # 专家层，强化局域信息
    def __init__(self,num_experts,in_features,out_features,device,*args, **kwargs) -> None:
        super(MoELayer,self).__init__(*args, **kwargs)
        self.head_size = in_features // num_experts
        self.experts = nn.ModuleList([FeedForward(in_features=self.head_size,out_features=out_features,device = device) for i in range(num_experts)])
        self.router = nn.Linear(in_features=in_features,out_features=num_experts).to(device)
        self.norm = nn.LayerNorm(normalized_shape=num_experts).to(device)
        pass

    def forward(self,x:torch.Tensor)->torch.Tensor:
        logits = self.router(x)
        router = torch.softmax(self.norm(logits),dim=-1)
        inputs = torch.split(x,self.head_size,dim=-1)
        experts = torch.stack([expert(input) for input,expert in zip(inputs,self.experts)],dim=0)
        y = torch.einsum('btx,xbts->bts',router,experts)
        return y
        pass

    pass

class DecoderOnlyBlock(nn.Module):
    def __init__(self,units, device,*args, **kwargs) -> None:
        super(DecoderOnlyBlock,self).__init__(*args, **kwargs)
        self.attention = SelfLinearAttention(64,units,units,device).to(device)
        self.feed_forward = FeedForward(in_features=units,out_features=units,device=device)
        self.norm_1 = nn.LayerNorm(normalized_shape=units).to(device)
        self.norm_2 = nn.LayerNorm(normalized_shape=units).to(device)
        self.norm_3 = nn.LayerNorm(normalized_shape=units).to(device)
        pass

    def forward(self,x):
        input = x
        x = self.norm_1(x)
        x = self.norm_2(self.attention(x)+x)
        x = self.norm_3(self.feed_forward(x) + x)
        return x + input
        
    def forget(self):
        self.attention.forget()
        pass

    pass


class DecoderOnly(nn.Module):
    '''
    Decoder层\n
    采用线性注意力机制实现\n
    可以手动清除模型记忆\n
    根据当前token输入以及记忆机制，预测下一token的输出\n
    '''
    def __init__(self,in_features,out_features,units = 512,num_block = 12,device = 'cpu',*args, **kwargs) -> None:
        super(DecoderOnly,self).__init__(*args, **kwargs)
        self.linear = nn.Linear(in_features=in_features,out_features=units,bias=False).to(device)
        self.decoder_blocks = [DecoderOnlyBlock(units,device).to(device) for i in range(num_block)]
        self.decoder_blocks[0].attention.is_use_rope(True)
        self.decoder = nn.Sequential(*(self.decoder_blocks))
        self.norm_1 = nn.LayerNorm(normalized_shape=units)
        self.norm_2 = nn.LayerNorm(normalized_shape=units)
        self.moe_2 = MoELayer(num_experts=8,in_features=units,out_features=units,device = device).to(device)
        self.logits = nn.Linear(units,out_features,bias=False).to(device)
        pass

    def forward(self,x):
        x = self.linear(x)
        x = self.norm_1(self.decoder(x) + x)
        x = self.norm_2(self.moe_2(x) + x)
        logits = self.logits(x)
        return logits
    
    def forget(self):
        for x in self.decoder_blocks:
            x.forget()
        pass
    pass

class LinearGPT(nn.Module):
    '''
    DecoderOnly架构语言模型\n
    采用线性注意力机制实现\n
    实现无限上下文计算\n
    '''
    def __init__(self,vocab_count,d_model,units = 512,num_block = 12,device = 'cpu', *args, **kwargs) -> None:
        super(LinearGPT,self).__init__(*args, **kwargs)
        self.embedding = nn.Embedding(vocab_count,d_model).to(device)
        self.decoder = DecoderOnly(in_features=d_model,out_features=vocab_count,units=units,num_block=num_block,device=device,*args,**kwargs)
        pass

    def forward(self,x,labels=None):
        x = self.embedding(x)
        logits = self.decoder(x)
        if labels is not None:
            # 训练模式
            B,T,C = logits.shape
            logits_reshaped = logits.view(B * T, C)
            labels_reshaped = labels.view(B * T)
            loss = F.cross_entropy(input=logits_reshaped, target=labels_reshaped)
            # 计算模型正确率
            predictions = torch.argmax(logits,dim=-1).view(B * T)
            correct = (predictions == labels_reshaped).sum().item()
            total = labels_reshaped.numel()
            accuracy = correct / total
            # 清空模型的记忆
            self.forget()
            return loss,accuracy
            pass
        return logits[:,-1,:]
        pass
    
    def forget(self):
        self.decoder.forget()
    pass

class EncoderBlock(nn.Module):
    def __init__(self,d_model,units,device, *args, **kwargs) -> None:
        super(EncoderBlock,self).__init__(*args, **kwargs)
        self.norm_1 = nn.LayerNorm(units).to(device)
        self.norm_2 = nn.LayerNorm(units).to(device)
        self.attention = SelfAttention(in_features=units,out_features=units,d_model=d_model,device=device)
        self.moe = MoELayer(num_experts=8,in_features=units,out_features=units,device=device)

    def forward(self,x):
        x = self.norm_1(x)
        x = self.norm_2(self.attention(x)+x)
        x = self.moe(x)+x
        return x
        pass

    def use_rope(self,state,max_pos = 64):
        self.attention.attention.use_rope(state,max_pos)
        pass

    pass

class EmbeddingBlock(nn.Module):
    def __init__(self,in_features,out_features,units,num_blocks,device, *args, **kwargs) -> None:
        super(EmbeddingBlock,self).__init__(*args, **kwargs)
        self.linear = nn.Linear(in_features=in_features,out_features=units,bias=False).to(device)
        self.encoder = EncoderBlock(32,units,device)
        self.encoder.use_rope(True)
        self.decoder = DecoderOnlyBlock(units,device=device)
        self.norm_1 = nn.LayerNorm(units).to(device)
        self.norm_2 = nn.LayerNorm(units).to(device)
        self.logits = nn.Linear(units,out_features,bias=False,device=device)
        pass

    def forward(self,x,label = None):
        x:torch.Tensor = self.linear(x)
        x:torch.Tensor = self.norm_1(self.encoder(x)+x)
        if label is None:
            key = torch.mean(x,dim=-2)
            return key 
            pass
        x = self.norm_2(self.decoder(x)+x)
        logits = self.logits(x)
        return logits
        pass

# 让模型自己预测自己
class EmbeddingModel(nn.Module):
    def __init__(self,vocab_count,d_model,units = 512,num_block = 12,device = 'cpu', *args, **kwargs) -> None:
        super(EmbeddingModel,self).__init__(*args, **kwargs)
        self.encoder = EmbeddingBlock(d_model,vocab_count,units,num_block,device=device)
        self.embedding = nn.Embedding(vocab_count,d_model).to(device)
        self.embedding.weight.data[44819] = torch.zeros(d_model)
        self.embedding.weight.data[44819].requires_grad = False

    def forward(self,x,labels=None):
        x = self.embedding(x)

        logits = self.encoder(x,labels)
        # 清空模型的记忆
        self.forget()
        if labels is not None:
            # 训练模式
            with torch.no_grad():
                B,T,C = logits.shape
                logits_reshaped = logits.view(B * T, C)
                labels_reshaped = labels.view(B * T)
            loss = F.cross_entropy(input=logits_reshaped, target=labels_reshaped)
            # 计算模型正确率
            predictions = torch.argmax(logits,dim=-1).view(B * T)
            correct = (predictions == labels_reshaped).sum().item()
            total = labels_reshaped.numel()
            accuracy = correct / total
            return loss,accuracy
            pass
        return logits
        pass

    def forget(self):
        self.encoder.decoder.forget()

    pass