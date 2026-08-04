import torch
import torch.nn as nn

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        # Q, K, V projections, all same size, this is normal not laziness
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.n_heads = n_heads
        self.d_head = d_model // n_heads  # if this isn't a clean int everything breaks
        self.d_model = d_model

    def forward(self, x):
        B, T, d_model = x.shape

        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)

        # split into heads, forget the transpose here and you'll debug shapes for an hour
        Q = Q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        K = K.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        V = V.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # Q @ K^T, scaled by sqrt(d_head) or softmax saturates and gradients die
        scores = Q @ K.transpose(-2, -1) / (self.d_head ** 0.5)

        # causal mask, no peeking at future tokens
        mask = torch.triu(torch.ones(T, T, device = x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))  # -inf becomes 0 after softmax

        weights = torch.softmax(scores, dim=-1)
        out = weights @ V

        # merge heads back into one vector
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return out


class MLP(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.fc1 = nn.Linear(d_model, 4 * d_model)  # expand, gives the model room to think
        self.fc2 = nn.Linear(4 * d_model, d_model)  # shrink back down
        self.gelu = nn.GELU()  # without this fc1+fc2 collapse into one linear layer, pointless

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        return x


class Block(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)  # pre-norm, before attention
        self.ln2 = nn.LayerNorm(d_model)  # pre-norm, before mlp
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.mlp = MLP(d_model)

    def forward(self, x):
        # residual connections, x + Block(x), never just Block(x)
        # drop the "x +" and a bad early layer wrecks everything downstream
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, d_model, n_heads, n_layer, vocab_size, block_size):
        super().__init__()
        # separate blocks, own weights each, not copies of each other
        self.blocks = nn.ModuleList([Block(d_model, n_heads) for i in range(n_layer)])
        self.token_embed = nn.Embedding(vocab_size, d_model)  # token id -> vector
        self.pos_embed = nn.Embedding(block_size, d_model)    # position -> vector, attention has no sense of order on its own
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)  # project to a score per vocab token

    def forward(self, x):
        B, T = x.shape  # x is still raw token ids here, (B, T), no d_model yet
        tok_emb = self.token_embed(x)
        positions = torch.arange(T, device = x.device)
        pos_emb = self.pos_embed(positions)
        x = tok_emb + pos_emb

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.head(x)
        return logits


# quick shape sanity checks

attn = CausalSelfAttention(d_model=8, n_heads=2)
test = torch.randn(2, 5, 8)
result = attn(test)
print(result.shape)

mlp = MLP(d_model=8)
test2 = torch.randn(2, 5, 8)
result2 = mlp(test2)
print(result2.shape)

block = Block(d_model=8, n_heads=2)
test3 = torch.randn(2, 5, 8)
result3 = block(test3)
print(result3.shape)

gpt = GPT(d_model=8, n_heads=2, n_layer=2, vocab_size=100, block_size=16)
test4 = torch.randint(0, 100, (2, 5))
result4 = gpt(test4)
print(result4.shape)
