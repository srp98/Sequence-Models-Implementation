import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 64                                         # how many independent sequences will we process in parallel?
block_size = 256                                        # what is the maximum context length for predictions?
max_iters = 3000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embed = 384
n_head = 6
n_layer = 6
dropout = 0.2

torch.manual_seed(1337)

with open('data/tiny_shakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]                 # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l])        # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))                                  # the first 90% will be trained, rest val
train_data = data[:n]
val_data = data[n:]


# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size, ))
    x = torch.stack([data[i: i+block_size] for i in ix])
    y = torch.stack([data[i+1: i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)

    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()

    return out


class Head(nn.Module):
    """ Self-Attention Head"""
    def __init__(self, head_size):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(n_embed, self.head_size, bias=False)
        self.query = nn.Linear(n_embed, self.head_size, bias=False)
        self.value = nn.Linear(n_embed, self.head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        # Compute Attention Scores
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        # Perform weighted aggregation of values
        v = self.value(x)
        out = wei @ v

        return out


class MultiHeadAttention(nn.Module):
    """Multiple Heads of self-attention in parallel"""
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.heads = nn.ModuleList([Head(self.head_size) for _ in range(self.num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))

        return out


class FeedForward(nn.Module):
    """A Simple Linear Layer followed by non-linearity"""
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_embed, 4 * n_embed),
                                 nn.ReLU(),
                                 nn.Linear(4 * n_embed, n_embed),    # Projection Layer going back to residual pathway
                                 nn.Dropout(dropout)
                                 )

    def forward(self, x):
        return self.net(x)


class LayerNorm1d:  # (used to be BatchNorm1d)

    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)

    def __call__(self, x):
        # calculate the forward pass
        xmean = x.mean(1, keepdim=True)                             # layer mean
        xvar = x.var(1, keepdim=True)                               # layer variance
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)            # normalize to unit variance
        self.out = self.gamma * xhat + self.beta

        return self.out

    def parameters(self):
        return [self.gamma, self.beta]


class Block(nn.Module):
    """Transformer Block: Communication followed by computation"""
    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embed)
        # self.ln1 = LayerNorm1d(n_embed)
        # self.ln2 = LayerNorm1d(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))              # Residual connections
        x = x + self.ffwd(self.ln2(x))

        return x


# simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.pos_embedding = nn.Embedding(block_size, n_embed)          # Positional Embedding of input
        # self.sa_head = Head(n_embed)
        # self.ma_heads = MultiHeadAttention(4, n_embed//4)             # 4 heads of 8-dimensional self-attention
        # self.ffwd = FeedForward(n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, n_head=n_head) for _ in range(n_layer)])
        # self.block = nn.Sequential(Block(n_embed, n_head=4),
        #                            Block(n_embed, n_head=4),
        #                            Block(n_embed, n_head=4),
        #                            Block(n_embed, n_head=4),
        #                            nn.LayerNorm(n_embed)
        #                            )
        self.ln_f = nn.LayerNorm(n_embed)                               # Final LayerNorm
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):

        B, T = idx.shape
        # idx and targets are both (B, T) tensor of integers
        token_embed = self.token_embedding_table(idx)                   # (B, T, C)
        pos_embed = self.pos_embedding(torch.arange(T, device=device))  # (T, C)
        x = token_embed + pos_embed                                     # (B, T, C)
        # x = self.sa_head(x)                                           # Single Attention Head
        # x = self.ma_heads(x)                                          # Multi Attention Heads
        # x = self.ffwd(x)                                              # Feedforward Network to collect information
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)                                        # (B, T, C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens, so it won't go over blocksize
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]                                   # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)                           # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)          # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)              # (B, T+1)

        return idx


model = BigramLanguageModel()
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
