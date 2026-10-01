import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self,embedding_dim, block_size,head_size):
        super().__init__()
        self.key=nn.Linear(embedding_dim, head_size,bias=False)
        self.query=nn.Linear(embedding_dim, head_size,bias=False)
        self.value=nn.Linear(embedding_dim, head_size,bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B,T,C =x.shape
        k=self.key(x)
        q=self.query(x)
        w=q@k.transpose(-2,-1)/C**0.5
        w=w.masked_fill(self.tril[:T,:T]==0,float('-inf'))
        w=F.softmax(w,dim=-1)
        v=self.value(x)
        output=w@v
        return output

class MultiheadAttention(nn.Module):
    def __init__(self, embedding_dim, block_size, num_heads):
        super().__init__()
        head_size=embedding_dim // num_heads
        self.heads=nn.ModuleList([SelfAttention(embedding_dim, block_size,head_size) for _ in range(num_heads)])
        self.proj=nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.proj(out)

class FeedForward(nn.Module):
    def __init__(self, n_emdd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_emdd, 4 * n_emdd),
            nn.ReLU(),
            nn.Linear(4 * n_emdd, n_emdd)
        )
    def forward(self,x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, embedding_dim, block_size, n_heads):
        super().__init__()
        self.sa=MultiheadAttention(embedding_dim, block_size, n_heads)
        self.ff=FeedForward(embedding_dim)
        self.ln1=nn.LayerNorm(embedding_dim)
        self.ln2=nn.LayerNorm(embedding_dim)

    def forward(self,x):
        x=x+self.sa(self.ln1(x))
        x=x+self.ff(self.ln2(x))
        return x

