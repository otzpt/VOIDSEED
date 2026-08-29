import sys
sys.path.append('/home/otzpt/Documentos/tinyllm2')
import re
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import tiktoken
from model import GPT
from search_index import search

# if anyone decides to maintain this code
# good luck with that it is hard to read
# or maybe its a skill issue its my first time
# doing something like this

app = Flask(__name__)
CORS(app)

@app.after_request
def add_ai_generated_header(response):
    # Art. 50(2): machine-readable marking that a reply came from an AI system.
    response.headers['X-AI-Generated'] = 'true'
    return response
# loads the model
def load_model():
    # creates GPT and loads model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = GPT(d_model = 768, n_heads = 12, n_layer = 12, vocab_size = 50257, block_size = 256)
    model = model.to(device)

    #loads the checkpoint
    checkpoint = torch.load('/home/otzpt/Documentos/tinyllm2/checkpoints/976000.pt', map_location = device)
    state_dict = checkpoint['model']
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()

    #returns the values
    return model, device

model, device = load_model()  # runs once when server starts

BLOCK_SIZE = 256
MAX_NEW_TOKENS = 60

# sampling straight from the full vocab gives word salad on a model this
# small. top-k keeps it on topic, and the penalty stops it locking onto one
# token and repeating it forever (that showed up as blank replies).
# careful: lowering temperature alone makes the repeating worse, not better
# 0.4/20 measured better than 0.7/40: answers stay anchored on the retrieved
# text instead of drifting (20% of answer words found in the context -> 29%)
TEMPERATURE = 0.4
TOP_K = 20
REPETITION_PENALTY = 1.3
EOT_TOKEN = 50256  # gpt-2 <|endoftext|>

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

def generate_response(prompt):
    # searchs for relevant passages
    enc = tiktoken.get_encoding("gpt2")

    # builds the prompt
    results = search(prompt, k = 3)
    context = useful_context(results)
    full_prompt = f"{context}\n\nQuestion: {prompt}\nAnswer: "

    # generates tokens same loop as chat.py, now with kv_cache
    # leave room for MAX_NEW_TOKENS: positions only grow with a cache, no
    # more sliding the window each step
    ids = enc.encode(full_prompt)[-(BLOCK_SIZE - MAX_NEW_TOKENS):]
    ids_tensor = torch.tensor([ids], device = device)
    prompt_len = len(ids) # stores how many tokens origin promt was/is

    with torch.no_grad():
        logits, kv_cache = model(ids_tensor)  # prefill: whole prompt at once
        for _ in range(MAX_NEW_TOKENS):
            # decodes and returns answer(text)
            last_logits = logits[:, -1, :]
            next_id = sample_next(last_logits, ids_tensor[0, prompt_len:])
            if next_id.item() == EOT_TOKEN:
                break
            ids_tensor = torch.cat([ids_tensor, next_id], dim = 1)
            logits, kv_cache = model(next_id, kv_cache)  # decode: just the new token
    # formating
    # this should fix the answer being more than 100 words of encoding
    # and decoding
    generated_ids = ids_tensor[0][prompt_len:].tolist()  # only new tokens
    answer = trim_answer(enc.decode(generated_ids))

    #cuts last phrase completly if there is a '.'
    last_period = answer.rfind('.')
    if last_period != -1:
        answer = answer[:last_period + 1]

    return answer

@app.route('/chat', methods=['POST'])
def chat():
    # reads message
    message = request.json.get('message')
    if not message:
        # no message = nothing to search or generate on, don't let it hit search()
        return jsonify({'response': '', 'ai_generated': True}), 400
    response = generate_response(message) # calls generate response
    # returns output
    return jsonify({'response': response, 'ai_generated': True})

if __name__ == '__main__':
    app.run(port=5000)
