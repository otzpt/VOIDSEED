import re
import torch
import tiktoken
from model import GPT
from search_index import search

device = 'cuda' if torch.cuda.is_available() else 'cpu'
enc = tiktoken.get_encoding("gpt2")

BLOCK_SIZE = 256
MAX_NEW_TOKENS = 100

# sampling straight from the full vocab gives word salad on a model this
# small. top-k keeps it on topic, and the penalty stops it locking onto one
# token and repeating it forever (that showed up as blank replies).
# careful: lowering temperature alone makes the repeating worse, not better
TEMPERATURE = 0.7
TOP_K = 40
REPETITION_PENALTY = 1.3
EOT_TOKEN = 50256  # gpt-2 <|endoftext|>

model = GPT(d_model = 768, n_heads = 12, n_layer = 12, vocab_size = 50257, block_size = 256)
model = model.to(device)

checkpoint = torch.load('checkpoints/976000.pt', map_location = device)
state_dict = checkpoint['model']
state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)
model.eval()

LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]*\)")
URL_RE = re.compile(r"https?://\S+")

def useful_context(passages):
    # a chunk that's mostly markdown links is a table of contents: no answer in
    # it, and the model copies the format and starts inventing real-looking URLs
    keep = []
    for p in passages:
        lines = [l for l in p.text.split("\n") if l.strip()]
        if lines and sum(1 for l in lines if "](" in l or "http" in l) / len(lines) > 0.5:
            continue
        text = URL_RE.sub("", LINK_RE.sub(r"\1", p.text))
        keep.append(re.sub(r"[ \t]+", " ", text).strip())
    if not keep:  # a weak chunk still beats no context at all
        keep = [p.text for p in passages]
    return "\n\n".join(keep)

def trim_answer(text):
    # the corpus is full of Q/A text, so after answering the model just writes
    # the next question. cut whatever it invents past the answer
    for marker in ("Question:", "<|endoftext|>"):
        text = text.split(marker)[0]
    # any url that survives here is invented -- a fake link that looks official
    # is worse than no link
    return re.sub(r"</?a\b[^>]*>", "", URL_RE.sub("", text)).strip()

def sample_next(logits, prev_ids):
    logits = logits.clone()
    seen = torch.unique(prev_ids)
    seen_logits = logits[:, seen]
    # multiply if negative, divide if positive -- dividing a negative logit
    # moves it towards zero, which would reward repeats instead of punishing them
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

        context = useful_context(results)
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
                if next_id.item() == EOT_TOKEN:
                    break
                ids_tensor = torch.cat([ids_tensor, next_id], dim = 1)
                logits, kv_cache = model(next_id, kv_cache)  # decode: just the new token

        output_ids = ids_tensor[0, prompt_len:].tolist()
        print(trim_answer(enc.decode(output_ids)))
