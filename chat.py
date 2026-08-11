import torch
import tiktoken
from model import GPT
from search_index import search

device = 'cuda' if torch.cuda.is_available() else 'cpu'
enc = tiktoken.get_encoding("gpt2")

BLOCK_SIZE = 256
MAX_NEW_TOKENS = 100

model = GPT(d_model = 768, n_heads = 12, n_layer = 12, vocab_size = 50257, block_size = 256)
model = model.to(device)

checkpoint = torch.load('checkpoints/976000.pt', map_location = device)
state_dict = checkpoint['model']
state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)
model.eval()

while True:
    prompt = input("\nPrompt(or 'exit' to leave): ")

    if prompt == "exit":
        break

    else:

        results = search(prompt, k = 3)

        context = "\n\n".join([r.text for r in results])
        full_prompt = f"{context}\n\nQuestion: {prompt}\nAnswer: "

        # leave room for MAX_NEW_TOKENS: with a kv_cache, positions only grow,
        # there's no more sliding the window each step like before
        ids = enc.encode(full_prompt)[-(BLOCK_SIZE - MAX_NEW_TOKENS):]
        print(ids)

        ids_tensor = torch.tensor([ids], device = device)

        with torch.no_grad():
            logits, kv_cache = model(ids_tensor)  # prefill: whole prompt at once
            for _ in range(MAX_NEW_TOKENS):
                last_logits = logits[:, -1, :]
                probs = torch.softmax(last_logits, dim = -1)
                next_id = torch.multinomial(probs, num_samples = 1)
                ids_tensor = torch.cat([ids_tensor, next_id], dim = 1)
                logits, kv_cache = model(next_id, kv_cache)  # decode: just the new token

        output_ids = ids_tensor[0].tolist()
        print(enc.decode(output_ids))
