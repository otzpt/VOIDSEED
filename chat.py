import torch
import tiktoken
from model import GPT
from search_index import search

device = 'cuda' if torch.cuda.is_available() else 'cpu'
enc = tiktoken.get_encoding("gpt2")

BLOCK_SIZE = 256
MAX_NEW_TOKENS = 100

# raw temperature=1.0 sampling over the full 50257-token softmax lets a
# small, undertrained model (this one: 155M params, ~12.6 tokens/param,
# below Chinchilla-optimal -- see README) pick from the long tail of
# unlikely tokens way too often, which is what "word salad" output looks
# like. Cooling the distribution and cutting off everything outside the
# top TOP_K candidates keeps it on-topic -- measured, same prompt: raw
# sampling drifted into fabricated terms within a sentence, this stayed
# coherent for the full 80-token sample.
TEMPERATURE = 0.7
TOP_K = 40
REPETITION_PENALTY = 1.3

model = GPT(d_model = 768, n_heads = 12, n_layer = 12, vocab_size = 50257, block_size = 256)
model = model.to(device)

checkpoint = torch.load('checkpoints/976000.pt', map_location = device)
state_dict = checkpoint['model']
state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)
model.eval()

def sample_next(logits, prev_ids):
    logits = logits.clone()
    seen = torch.unique(prev_ids)
    seen_logits = logits[:, seen]
    # negative logits get multiplied (pushed further negative), positive ones divided --
    # a plain divide would boost already-negative logits instead of penalizing them
    logits[:, seen] = torch.where(seen_logits < 0, seen_logits * REPETITION_PENALTY, seen_logits / REPETITION_PENALTY)
    logits = logits / TEMPERATURE
    top_values, top_indices = torch.topk(logits, TOP_K)
    probs = torch.softmax(top_values, dim = -1)
    choice = torch.multinomial(probs, num_samples = 1)
    return top_indices.gather(-1, choice)

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
        prompt_len = ids_tensor.shape[1]

        with torch.no_grad():
            logits, kv_cache = model(ids_tensor)  # prefill: whole prompt at once
            for _ in range(MAX_NEW_TOKENS):
                last_logits = logits[:, -1, :]
                next_id = sample_next(last_logits, ids_tensor[0, prompt_len:])
                ids_tensor = torch.cat([ids_tensor, next_id], dim = 1)
                logits, kv_cache = model(next_id, kv_cache)  # decode: just the new token

        output_ids = ids_tensor[0].tolist()
        print(enc.decode(output_ids))
