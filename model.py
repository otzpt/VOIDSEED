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

    def forward(self, x, kv_cache = None):
        B, T, d_model = x.shape

        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)

        # split into heads, forget the transpose here and you'll debug shapes for an hour
        Q = Q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        K = K.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        V = V.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # kv_cache optimization
        if kv_cache is not None:
            past_K, past_V = kv_cache
            K = torch.cat([past_K, K], dim = 2) #dim = 2 is the dimension T
            V = torch.cat([past_V,V], dim = 2)

        new_kv_cache = (K, V) #stores updated K, V
        # Q @ K^T, scaled by sqrt(d_head) or softmax saturates and gradients die
        # shape is (B, n_heads, T, T_total), and T_total > T once there's a cache
        scores = Q @ K.transpose(-2, -1) / (self.d_head ** 0.5)

        # only needed when several new tokens are scored against each other at
        # once (training, or the first pass over a prompt). With a cache we're
        # doing one token at a time and it's always the newest, so there's no
        # future for it to peek at -- no mask needed.
        if kv_cache is None:
            mask = torch.triu(torch.ones(T, T, device = x.device), diagonal=1).bool()
            scores = scores.masked_fill(mask, float('-inf'))  # -inf becomes 0 after softmax

        weights = torch.softmax(scores, dim=-1)
        out = weights @ V

        # merge heads back into one vector
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return out, new_kv_cache


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

    def forward(self, x, kv_cache = None):
        # residual connections, x + Block(x), never just Block(x)
        # drop the "x +" and a bad early layer wrecks everything downstream
        attn_out, new_kv_cache = self.attn(self.ln1(x), kv_cache)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, new_kv_cache


class GPT(nn.Module):
    def __init__(self, d_model, n_heads, n_layer, vocab_size, block_size):
        super().__init__()
        # separate blocks, own weights each, not copies of each other
        self.blocks = nn.ModuleList([Block(d_model, n_heads) for i in range(n_layer)])
        self.token_embed = nn.Embedding(vocab_size, d_model)  # token id -> vector
        self.pos_embed = nn.Embedding(block_size, d_model)    # position -> vector, attention has no sense of order on its own
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)  # project to a score per vocab token

    def forward(self, x, kv_cache = None):
        B, T = x.shape  # x is still raw token ids here, (B, T), no d_model yet
        tok_emb = self.token_embed(x)

        # positions have to be absolute -- with a cache x is only the new
        # tokens, so index 0 here isn't position 0 of the sequence
        past_length = kv_cache[0][0].shape[2] if kv_cache is not None else 0
        positions = torch.arange(past_length, past_length + T, device = x.device)
        pos_emb = self.pos_embed(positions)
        x = tok_emb + pos_emb

        layer_caches = kv_cache if kv_cache is not None else [None] * len(self.blocks)
        new_kv_cache = []
        for block, layer_cache in zip(self.blocks, layer_caches):
            x, updated_cache = block(x, layer_cache)
            new_kv_cache.append(updated_cache)

        x = self.ln_f(x)
        logits = self.head(x)
        return logits, new_kv_cache


# quick shape sanity checks. behind __main__ so importing GPT doesn't run them
if __name__ == "__main__":
    attn = CausalSelfAttention(d_model=8, n_heads=2)
    test = torch.randn(2, 5, 8)
    result, _ = attn(test)
    print(result.shape)

    mlp = MLP(d_model=8)
    test2 = torch.randn(2, 5, 8)
    result2 = mlp(test2)
    print(result2.shape)

    block = Block(d_model=8, n_heads=2)
    test3 = torch.randn(2, 5, 8)
    result3, _ = block(test3)
    print(result3.shape)

    gpt = GPT(d_model=8, n_heads=2, n_layer=2, vocab_size=100, block_size=16)
    test4 = torch.randint(0, 100, (2, 5))
    result4, _ = gpt(test4)
    print(result4.shape)

    # whole sequence at once must give the same logits as one token at a time
    # through the cache. if this fails the cache is silently wrong
    gpt.eval()
    with torch.no_grad():
        tokens = torch.randint(0, 100, (1, 6))

        full_logits, _ = gpt(tokens)

        cached_logits = []
        kv_cache = None
        for t in range(tokens.shape[1]):
            step_logits, kv_cache = gpt(tokens[:, t:t + 1], kv_cache)
            cached_logits.append(step_logits)
        cached_logits = torch.cat(cached_logits, dim=1)

        assert torch.allclose(full_logits, cached_logits, atol=1e-5), \
            "kv_cache path diverged from full-sequence path"
        print("kv_cache equivalence check passed")
