import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from transformer import Block

# print("Using PyTorch version", torch.__version__, "  and CUDA", torch.version.cuda)

corpus = [
    "hello friends how are you",
    "the tea is very hot",
    "my name is Aarohi",
    "the roads of Delhi are busy",
    "it is raining in Mumbai",
    "the train is late again",
    "i love eating samosas and drinking tea",
    "holi is my favorite festival",
    "diwali brings lights and sweets",
    "india won the cricket match"
]

corpus=[s + " <END>" for s in corpus]
text=" ".join(corpus)

words=list(set(text.split()))
# print(words)

vocab_size=len(words)
# print(vocab_size)

word2indx={w:i for i,w in enumerate(words)}
# print(word2indx)

idx2word = {i: w for w, i in word2indx.items()} 
#print("idx2word : ", idx2word) #


data=torch.tensor([word2indx[w] for w in text.split()],dtype=torch.long)
# print(data)

block_size=6    #Model can see 6 tokens in the past to predict the next token or num of  Input token
embedding_dim=32
n_heads=2       # 2 multihead attention heads
n_layers=2      # 2 Transformer block 
lr=1e-3
epochs=1000

# Divide the data into batches of size 16

def get_batch(batch_size=16):
    
    ix = torch.randint(len(data) - block_size,(batch_size,))# (62-6=56),16  # Randomly select 16 starting indices for the batches
    x=torch.stack([data[i:i+block_size] for i in ix])
    y=torch.stack([data[i+1:i+block_size+1] for i in ix])           # Actual prediction
    return x,y

class SmallGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table=nn.Embedding(vocab_size,embedding_dim)
        self.position_embedding_table=nn.Embedding(block_size,embedding_dim)
        self.Block=nn.Sequential(*[Block(embedding_dim, block_size, n_heads) for _ in range(n_layers)]) 
        self.linear=nn.LayerNorm(embedding_dim)
        self.head=nn.Linear(embedding_dim,vocab_size)
    def forward(self,indx,targets=None):
        batch_size,block_size=indx.shape # (16,6)
        token_embeddings = self.token_embedding_table(indx) # (16,6,32)
        position_embeddings = self.position_embedding_table(
            torch.arange(block_size, device=indx.device)
        )

        x=token_embeddings+position_embeddings # (16,6,32)
        x=self.Block(x) # (16,6,32)
        x=self.linear(x) # (16,6,32)
        logits=self.head(x) # (16,6,10)
        loss=None

        if targets is not None:
            batch_size,block_size,vocab_size=logits.shape #(16,6,42)  logits=raw outpput
            loss=F.cross_entropy(logits.view(batch_size*block_size,vocab_size),targets.view(batch_size*block_size))
        return logits,loss

    def generate(self,indx,max_new_token):
        for _ in range(max_new_token):
            indx_cond=indx[:,-block_size:]
            logits,_=self(indx_cond) # (16,6,10)
            logits=logits[:,-1,:] # (16,10) raw output
            probability=F.softmax(logits,dim=-1)
            next_token=torch.multinomial(probability,num_samples=1) # (16,1)
            indx=torch.cat((indx,next_token),dim=1) # (16,7)
        return indx

model = SmallGPT()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

for step in range(epochs):
    xb, yb = get_batch() 
    logits, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 300 == 0:
        print(f"Step {step}, loss={loss.item():.4f}")

context = torch.tensor([[word2indx["hello"]]], dtype=torch.long)
out = model.generate(context, max_new_token=15)

print("\nGenerated text:\n")
print(" ".join(idx2word[int(i)] for i in out[0]))
